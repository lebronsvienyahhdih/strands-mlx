#!/usr/bin/env python3
"""Test vision analysis using mlx_vision_invoke as a tool."""
import pytest
import os
from strands import Agent
from strands_mlx import MLXModel, mlx_vision_invoke


def test_vision_invoke_direct():
    """Test direct mlx_vision_invoke tool call"""
    # Check if test media exists
    image_path = "./test_media/sample_image.jpg"
    if not os.path.exists(image_path):
        pytest.skip(f"Test media not found: {image_path}")

    # Use text model with vision tool
    model = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit")
    agent = Agent(model=model, tools=[mlx_vision_invoke])

    print("\n" + "=" * 50)
    print("🔧 DIRECT VISION INVOKE TEST")
    print("=" * 50)
    print(f"Text Model: Qwen3-1.7B-4bit")
    print(f"Vision Tool: mlx_vision_invoke")
    print(f"Image: {image_path}\n")

    # Test direct tool call
    result = agent.tool.mlx_vision_invoke(
        prompt="Describe this image in detail",
        image_path=image_path,
        model_id="mlx-community/Qwen2-VL-2B-Instruct-4bit",
    )

    print(f"\n🖼️ Vision invoke result:\n{result}\n")

    # Assertions
    assert result is not None, "Result should not be None"
    result_text = str(result).lower()
    assert len(result_text) > 0, "Result should have content"
    assert len(result_text) > 20, "Description should be substantial"

    # Check for HTTP status code keywords
    assert any(
        keyword in result_text for keyword in ["200", "404", "status", "http"]
    ), "Vision analysis should identify HTTP status codes"

    print("=" * 50)
    print("✅ Direct vision invoke test passed!")
    print("=" * 50)


def test_agent_using_vision_invoke():
    """Test agent naturally using mlx_vision_invoke tool"""
    # Check if test media exists
    image_path = "./test_media/sample_image.jpg"
    if not os.path.exists(image_path):
        pytest.skip(f"Test media not found: {image_path}")

    # Use text model with vision tool
    model = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit")
    agent = Agent(model=model, tools=[mlx_vision_invoke])

    print("\n" + "=" * 50)
    print("🤖 AGENT USING VISION INVOKE TEST")
    print("=" * 50)
    print(f"Text Model: Qwen3-1.7B-4bit")
    print(f"Vision Tool: mlx_vision_invoke (available to agent)")
    print(f"Image: {image_path}\n")

    # Test agent calling vision tool naturally
    result = agent(f"What do you see in this image: {image_path}?")

    print(f"\n🖼️ Agent response:\n{result}\n")

    # Assertions
    assert result is not None, "Result should not be None"
    result_text = str(result).lower()
    assert len(result_text) > 0, "Result should have content"
    assert len(result_text) > 20, "Description should be substantial"

    # Check for HTTP status code keywords
    assert any(
        keyword in result_text for keyword in ["200", "404", "status", "http", "ok", "not found"]
    ), "Vision analysis should identify HTTP status codes"

    print("=" * 50)
    print("✅ Agent vision invoke test passed!")
    print("=" * 50)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
