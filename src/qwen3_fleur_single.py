"""
FLEUR for Qwen3-Omni (Single file mode without vLLM)

This script implements FLEUR metric evaluation for audio captions using Qwen3-Omni models
with transformers directly, without vLLM. This is optimized for single audio-caption pair
evaluation.

Reference: FLEUR: An Explainable Reference-Free Evaluation Metric for Image Captioning
           Using a Large Multimodal Model (ACL 2024)
           https://github.com/Yebin46/FLEUR
"""

import os
import re
import argparse

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info


# Default model path - can be overridden via environment variable or argument
DEFAULT_MODEL_PATH = os.environ.get("QWEN3_OMNI_MODEL_PATH", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
DEFAULT_THINKING_MODEL_PATH = os.environ.get("QWEN3_OMNI_THINKING_MODEL_PATH", "Qwen/Qwen3-Omni-30B-A3B-Thinking")


def make_fleur_prompt(audio_path: str, pred_caption: str) -> list:
    """
    Create FLEUR evaluation prompt for Qwen3-Omni

    Args:
        audio_path: Path to the audio file
        pred_caption: The caption to evaluate

    Returns:
        Messages list for Qwen3-Omni
    """
    # FLEUR instruction: evaluate caption quality on 0.0-1.0 scale
    eval_prompt = f"""Your task is to evaluate and rate the caption on a scale of 0.0 to 1.0 based on the given Grading Criteria. (Print Real Number Score ONLY)

Grading Criteria:

0.0: The caption does not describe the audio at all.
1.0: The caption accurately and clearly describes the audio.

Caption: {pred_caption}

Score(Choose a rating from 0.0 to 1.0):"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": eval_prompt},
            ],
        }
    ]

    return messages


def calculate_smoothed_score(
    output_text: str,
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    rate2token: dict,
    processor
) -> tuple:
    """
    Calculate score smoothing based on token probability distributions.

    This implements the FLEUR score smoothing algorithm which uses the probability
    distribution over digit tokens (0-9) to compute a weighted average score,
    making the evaluation more robust to minor variations in model output.

    Args:
        output_text: The raw output text from the model
        logits: The logits tensor from model generation
        generated_ids: The generated token IDs
        rate2token: Dictionary mapping digits (0-9) to their token IDs
        processor: The processor for decoding

    Returns:
        Smoothed FLEUR score (0.0 to 1.0)
    """
    # Parse the score from output text
    # Remove non-numeric characters except dots
    dotsnumbersdots = re.sub(r'[^\d\.]', '', output_text)
    # Remove leading dots
    numbersdots = re.sub(r'^\.+', '', dotsnumbersdots)
    # Remove trailing dots
    numbers = re.sub(r'\.+$', '', numbersdots)

    if not numbers:
        return None, None

    try:
        score_check = float(numbers)
    except ValueError:
        return None, None

    # Validate score range
    if score_check < 0 or score_check > 1:
        return score_check, None

    # Convert generated_ids to list for searching
    token_ids_list = generated_ids.tolist()

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)

    if score_check < 1.0:
        # Find the first decimal digit position in the score string
        # e.g., "0.7" -> find position of '7' which is index 2 (after "0.")
        score_str = str(score_check)
        num_index_in_score = score_str.index('.') + 1
        find_num = int(score_str[num_index_in_score])

        # Find the token position in the output
        target_token = rate2token[find_num]
        matching_indices = [i for i, tid in enumerate(token_ids_list) if tid == target_token]

        if not matching_indices:
            # Fallback to raw score if token not found
            return score_check, score_check

        # Handle duplicates: for 0.0 select second 0 (after dot), for others select first
        if find_num == 0 and len(matching_indices) > 1:
            num_index_in_token = matching_indices[1]
        else:
            num_index_in_token = matching_indices[0]

        # Get probability distribution at this position
        if num_index_in_token >= len(probs):
            return score_check, score_check

        probs_at_pos = probs[num_index_in_token]

        # Calculate weighted score using probabilities for digits 0-9
        # Each digit d contributes: P(d) * d * 0.1 (for first decimal place)
        score = 0.0
        for rate, token in rate2token.items():
            prob = probs_at_pos[token].item()
            score += prob * rate * 0.1

        # Handle second decimal place if present (e.g., 0.75)
        if len(score_str) > 3:
            num2_index_in_score = score_str.index('.') + 2
            if num2_index_in_score < len(score_str):
                find_num2 = int(score_str[num2_index_in_score])

                target_token2 = rate2token[find_num2]
                matching_indices2 = [i for i, tid in enumerate(token_ids_list) if tid == target_token2]

                if matching_indices2:
                    # For second decimal, choose the second occurrence
                    if len(matching_indices2) > 1:
                        num2_index_in_token = matching_indices2[1]
                    else:
                        num2_index_in_token = matching_indices2[0]

                    if num2_index_in_token < len(probs):
                        probs_at_pos2 = probs[num2_index_in_token]

                        # Add contribution from second decimal place
                        for rate, token in rate2token.items():
                            prob = probs_at_pos2[token].item()
                            score += prob * rate * 0.01

    else:
        # score_check == 1.0 case
        # Find position of '1' token
        target_token = rate2token[1]
        matching_indices = [i for i, tid in enumerate(token_ids_list) if tid == target_token]

        if not matching_indices:
            return score_check, score_check

        num_index_in_token = matching_indices[0]

        if num_index_in_token >= len(probs):
            return score_check, score_check

        probs_at_pos = probs[num_index_in_token]

        # For 1.0, calculate as: 0.9 * P(0) + 1.0 * P(1)
        score = 0.0
        prob_0 = probs_at_pos[rate2token[0]].item()
        score += 0.9 * prob_0
        prob_1 = probs_at_pos[rate2token[1]].item()
        score += prob_1

    return score_check, score


def load_model(args):
    """
    Load Qwen3-Omni model with transformers (without vLLM).

    Args:
        args: Command line arguments containing:
            - model_path: Path to the Qwen3-Omni model (optional)
            - use_think_mode: Whether to use thinking model variant

    Returns:
        Initialized model, processor, and rate2token mapping
    """
    # Set model path
    model_path = getattr(args, 'model_path', None) or DEFAULT_MODEL_PATH
    if args.use_think_mode:
        model_path = getattr(args, 'thinking_model_path', None) or DEFAULT_THINKING_MODEL_PATH
        print("Using thinking model variant")

    print(f"Loading model from: {model_path}")

    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    # Create mapping from digits (0-9) to token IDs for score smoothing
    rate2token = {}
    for digit in range(10):
        encoded = processor.tokenizer.encode(str(digit), add_special_tokens=False)
        rate2token[digit] = encoded[-1]  # Use the last token ID

    return model, processor, rate2token


def get_fleur(model, processor, rate2token, caption, audio):
    """
    Compute FLEUR score for a single audio-caption pair.

    Args:
        model: Qwen3-Omni model
        processor: Qwen3OmniMoeProcessor
        rate2token: Dictionary mapping digits (0-9) to their token IDs
        caption: The caption to evaluate
        audio: Path to the audio file

    Returns:
        raw_score: The raw score parsed from model output
        score: The smoothed FLEUR score
    """
    # Create FLEUR evaluation prompt
    messages = make_fleur_prompt(audio, caption)

    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    audio, _, _ = process_mm_info(messages, use_audio_in_video=False)
    # Process inputs with audio
    inputs = processor(
        text=text,
        audio=audio,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False
    )
    inputs = inputs.to(model.device).to(model.dtype)

    try:
        # Generate with output scores for probability calculation
        with torch.no_grad():
            outputs, _ = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,  # Deterministic for evaluation
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Get generated tokens (excluding input)
        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs.sequences[0, input_len:]

        # Decode output text
        output_text = processor.batch_decode(
            outputs.sequences[:, input_len:],
            skip_special_tokens=True
        )[0].strip()

        # Handle thinking mode output
        if "</think>" in output_text:
            output_text = output_text.split("</think>")[-1].strip()

        # Stack scores to get logits for each generated token
        if outputs.scores:
            logits = torch.stack(outputs.scores, dim=0).squeeze(1)

            # Calculate smoothed score
            raw_score, score = calculate_smoothed_score(
                output_text,
                logits,
                generated_ids,
                rate2token,
                processor
            )
        else:
            # Fallback: parse raw score without smoothing
            dotsnumbersdots = re.sub(r'[^\d\.]', '', output_text)
            numbersdots = re.sub(r'^\.+', '', dotsnumbersdots)
            numbers = re.sub(r'\.+$', '', numbersdots)
            try:
                raw_score = float(numbers) if numbers else None
                score = raw_score
            except ValueError:
                raw_score = None
                score = None

    except Exception as e:
        print(f"Error in generation: {e}")
        score = None
        raw_score = None

    return raw_score, score


def main(args):
    # Load model
    model, processor, rate2token = load_model(args)

    # Compute FLEUR score
    raw_score, fleur_score = get_fleur(
        model,
        processor,
        rate2token,
        args.pred_caption,
        args.audio_path
    )

    print(f"Final FLEUR Score: {fleur_score}, Raw Score: {raw_score}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute FLEUR scores for audio captions using Qwen3-Omni (single file mode)"
    )
    parser.add_argument(
        '--audio_path',
        type=str,
        required=True,
        help='Audio file path to evaluate'
    )
    parser.add_argument(
        '--pred_caption',
        type=str,
        required=True,
        help='Caption to evaluate against the audio'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to Qwen3-Omni model (default: from environment or HuggingFace)'
    )
    parser.add_argument(
        '--thinking_model_path',
        type=str,
        default=None,
        help='Path to Qwen3-Omni thinking model (default: from environment or HuggingFace)'
    )
    parser.add_argument(
        '--use_think_mode',
        action='store_true',
        default=False,
        help='Whether to use thinking model variant'
    )

    args = parser.parse_args()

    print('Computing FLEUR scores with Qwen3-Omni (single file mode)...')
    main(args)
