"""Vision tests for strands-mlx"""

import pytest
from pathlib import Path

# Test media path
TEST_IMAGE = Path("./test_media/sample_image.jpg")


def test_vision_basic():
    """Test basic vision model functionality"""
    from strands import Agent
    from strands_mlx import MLXVisionModel

    model = MLXVisionModel(model_id="mlx-community/Qwen2-VL-2B-Instruct-4bit")
    agent = Agent(model=model)

    response = agent(f"Describe this image: <image>{TEST_IMAGE}</image>")
    print(f"\n🖼️ Basic vision response:\n{response}\n")

    # Assertions
    assert response is not None
    assert len(str(response)) > 0

    # Check for expected content (HTTP status codes)
    response_text = str(response).lower()
    assert "200" in response_text or "ok" in response_text
    assert "404" in response_text or "not found" in response_text


def test_vision_detailed():
    """Test detailed vision analysis"""
    from strands import Agent
    from strands_mlx import MLXVisionModel

    model = MLXVisionModel(model_id="mlx-community/Qwen2-VL-2B-Instruct-4bit")
    agent = Agent(model=model)

    response = agent(f"Analyze this image in detail: <image>{TEST_IMAGE}</image>")
    print(f"\n🔍 Detailed vision response:\n{response}\n")

    # Assertions
    assert response is not None
    assert len(str(response)) > 50  # Detailed response should be substantial

    # Check for HTTP status code references
    response_text = str(response).lower()
    assert "status" in response_text or "http" in response_text
    assert "200" in response_text or "404" in response_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
