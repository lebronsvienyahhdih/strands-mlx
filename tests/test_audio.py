#!/usr/bin/env python3
"""Test audio analysis with MLXVisionModel."""
import pytest
import os
from strands import Agent
from strands_mlx import MLXVisionModel


def test_audio_transcription():
    """Test basic audio transcription"""
    # Check if test media exists
    audio_path = "./test_media/audio_task_completed.wav"
    if not os.path.exists(audio_path):
        pytest.skip(f"Test media not found: {audio_path}")

    # Use Gemma3n - audio-capable model
    model = MLXVisionModel(
        model_id="mlx-community/gemma-3n-E2B-it-5bit",
        params={"temperature": 0.7, "max_tokens": 1000},
    )

    agent = Agent(model=model)

    print("\n" + "=" * 50)
    print("🎵 AUDIO TRANSCRIPTION TEST")
    print("=" * 50)
    print(f"Model: gemma-3n-E2B-it-5bit")
    print(f"Audio: {audio_path}\n")

    # Test audio transcription
    result = agent(f"Transcribe this audio: <audio>{audio_path}</audio>")
    print(f"\n🎵 Audio transcription result:\n{result}\n")

    # Assertions
    assert result is not None, "Result should not be None"
    result_text = str(result).lower()
    assert len(result_text) > 0, "Result should have content"

    # Audio transcriptions should contain some text
    assert len(result_text) > 10, "Transcription should be substantial"

    # Check for expected transcription content
    assert (
        "task" in result_text and "completed" in result_text
    ), "Transcription should contain 'task completed successfully'"

    print("=" * 50)
    print("✅ Audio test completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
