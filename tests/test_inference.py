"""Inference tests for strands-mlx"""

import pytest
from strands import Agent
from strands_mlx import MLXModel
from strands_tools import calculator


def test_basic_agent():
    """Test basic agent functionality"""
    model = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit")
    agent = Agent(model=model, tools=[calculator])

    response = agent("What is 29 * 42?")
    assert response is not None
    assert "1218" in str(response)


def test_mlx_invoke():
    """Test mlx_invoke tool"""
    from strands_mlx.tools.mlx_invoke import mlx_invoke

    result = mlx_invoke(
        prompt="What is the capital of France?", model_id="mlx-community/Qwen3-1.7B-4bit"
    )

    assert result["status"] == "success"
    assert len(result["content"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
