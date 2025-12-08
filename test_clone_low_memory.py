#!/usr/bin/env python
"""
Low-memory version of test_clone.py using on-the-fly dequantization.

This version reduces peak GPU memory from ~12.8GB to ~3.5GB by using
on-the-fly dequantization instead of caching decompressed weights.

Usage:
    python test_clone_low_memory.py --model-path /data/models/step-audio-edix \
                                    --prompt-audio ./01_raw.wav
"""

import os
import argparse
import torch
import logging
import soundfile as sf

# ============================================================================
# IMPORTANT: Enable on-the-fly BEFORE importing model-related modules
# ============================================================================
from on_the_fly_dequant import enable_on_the_fly
enable_on_the_fly()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project imports (after enabling on-the-fly)
from tokenizer import StepAudioTokenizer  # noqa: E402
from tts import StepAudioTTS  # noqa: E402
from model_loader import ModelSource  # noqa: E402


def get_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def main():
    parser = argparse.ArgumentParser(description="Test Step-Audio-EditX Clone (Low Memory)")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/models/step-audio-edix",
        help="Model path",
    )
    parser.add_argument(
        "--prompt-audio",
        type=str,
        help="Path to prompt audio file",
        default="./01_raw.wav",
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="太棒了！恭喜你啊！我就知道你一定可以的！今晚必须庆祝一下！",
        help="Text content of the prompt audio",
    )
    parser.add_argument(
        "--target-text",
        type=str,
        default="先去订个餐厅，好好吃一顿",
        help="Text to synthesize with cloned voice",
    )
    parser.add_argument(
        "--output", type=str, default="output_clone_low_mem.wav", help="Output audio file path"
    )

    args = parser.parse_args()

    if not os.path.exists(args.prompt_audio):
        logger.error(f"Prompt audio file not found: {args.prompt_audio}")
        return

    logger.info("=" * 60)
    logger.info("🎵 Step-Audio-EditX Clone Test (Low Memory Mode)")
    logger.info("=" * 60)
    logger.info("⚡ Using on-the-fly dequantization to reduce memory usage")
    logger.info(f"Initial GPU memory: {get_memory_mb():.1f} MB")
    logger.info("=" * 60)

    # Reset peak stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Initialize models
    logger.info("Loading StepAudioTokenizer...")
    encoder = StepAudioTokenizer(
        os.path.join(args.model_path, "Step-Audio-Tokenizer"),
        model_source=ModelSource.LOCAL,
        funasr_model_id="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online",
    )
    logger.info(f"✓ StepAudioTokenizer loaded. Memory: {get_memory_mb():.1f} MB")

    # Load TTS model
    tts_model_path = os.path.join(args.model_path, "Step-Audio-EditX-AWQ-4bit")
    
    logger.info(f"Loading StepAudioTTS from {tts_model_path}...")
    tts_engine = StepAudioTTS(
        tts_model_path,
        encoder,
        model_source=ModelSource.LOCAL,
        quantization_config="awq-4bit",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    logger.info(f"✓ StepAudioTTS loaded. Memory: {get_memory_mb():.1f} MB")

    # Perform clone
    logger.info("Starting voice cloning...")
    try:
        output_audio, output_sr = tts_engine.clone(
            args.prompt_audio, args.prompt_text, args.target_text
        )

        # Get peak memory
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        current_mem = get_memory_mb()

        # Convert tensor to numpy if needed
        if isinstance(output_audio, torch.Tensor):
            audio_numpy = output_audio.cpu().numpy().squeeze()
        else:
            audio_numpy = output_audio

        # Save output audio
        sf.write(args.output, audio_numpy, output_sr)
        
        logger.info("=" * 60)
        logger.info(f"✓ Clone completed! Output saved to: {args.output}")
        logger.info(f"  Sample rate: {output_sr}")
        logger.info(f"  Duration: {len(audio_numpy) / output_sr:.2f}s")
        logger.info(f"  Current GPU memory: {current_mem:.1f} MB")
        logger.info(f"  Peak GPU memory: {peak_mem:.1f} MB")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Clone failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

