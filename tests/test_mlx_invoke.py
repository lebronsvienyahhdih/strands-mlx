"""Test mlx_invoke tool - invoke MLX models with custom settings."""

import pytest
from strands import Agent
from strands_tools import calculator
from strands_mlx import mlx_invoke


def test_basic_mlx_invoke():
    """Test basic mlx_invoke functionality"""
    agent = Agent(tools=[mlx_invoke, calculator])

    result = agent.tool.mlx_invoke(
        prompt="What is 29*42?",
        system_prompt="You are a helpful math assistant.",
        model_id="mlx-community/Qwen3-1.7B-4bit",
        agent=agent,
    )
    print(f"\n🧮 Basic mlx_invoke result:\n{result}\n")

    # Assertions
    assert result is not None
    assert "status" in result
    assert result["status"] == "success"
    result_text = str(result).lower()
    assert "1218" in result_text  # Should have the answer


def test_mlx_invoke_with_calculator():
    """Test mlx_invoke with calculator tool"""
    agent = Agent(tools=[mlx_invoke, calculator])

    result = agent.tool.mlx_invoke(
        prompt="Calculate 15 * 7 and explain the result",
        system_prompt="You are a math tutor.",
        model_id="mlx-community/Qwen3-1.7B-4bit",
        tools=["calculator"],
        params={"temperature": 0.7, "max_tokens": 2000},
        agent=agent,
    )
    print(f"\n🔢 mlx_invoke with calculator:\n{result}\n")

    # Assertions
    assert result is not None
    assert "status" in result
    assert result["status"] == "success"
    result_text = str(result).lower()
    assert "105" in result_text  # 15 * 7 = 105


def test_agent_using_mlx_invoke():
    """Test agent naturally using mlx_invoke"""
    agent = Agent(tools=[mlx_invoke, calculator])

    response = agent(
        "Use mlx_invoke to ask the Qwen3-1.7B-4bit model to calculate 25 * 4 with the calculator tool"
    )
    print(f"\n🤖 Agent using mlx_invoke naturally:\n{response}\n")

    # Assertions
    assert response is not None
    response_text = str(response).lower()
    assert "100" in response_text  # 25 * 4 = 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
