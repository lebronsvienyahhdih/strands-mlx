"""Training pipeline tests for strands-mlx"""

import pytest
import tempfile
from pathlib import Path
from strands import Agent
from strands_mlx import MLXModel, MLXSessionManager
from strands_mlx.tools.validate_training_data import validate_training_data
from strands_mlx.tools.dataset_splitter import dataset_splitter
from strands_tools import calculator


@pytest.fixture
def temp_dir():
    """Create temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_data_collection(temp_dir):
    """Test training data collection with MLXSessionManager"""
    session_id = "test_session"
    session = MLXSessionManager(session_id=session_id, storage_dir=temp_dir)
    model = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit")
    agent = Agent(model=model, tools=[calculator], session_manager=session)

    # Generate some training data
    agent("What is 5 + 3?")

    # Check file was created
    session_file = Path(temp_dir) / f"{session_id}.jsonl"
    assert session_file.exists()

    # Check file has content
    content = session_file.read_text()
    assert len(content) > 0


def test_validation(temp_dir):
    """Test training data validation"""
    # Create sample training data
    session_id = "validation_test"
    session = MLXSessionManager(session_id=session_id, storage_dir=temp_dir)
    model = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit")
    agent = Agent(model=model, tools=[calculator], session_manager=session)

    agent("Calculate 7 * 6")

    # Validate the generated data
    jsonl_path = Path(temp_dir) / f"{session_id}.jsonl"
    result = validate_training_data(jsonl_path=str(jsonl_path), max_examples=1)

    assert result["status"] == "success"


def test_dataset_splitter(temp_dir):
    """Test dataset splitting functionality"""
    # Create sample training data
    session_id = "split_test"
    session = MLXSessionManager(session_id=session_id, storage_dir=temp_dir)
    model = MLXModel(model_id="mlx-community/Qwen3-1.7B-4bit")
    agent = Agent(model=model, tools=[calculator], session_manager=session)

    # Generate multiple examples
    for i in range(5):
        agent(f"What is {i} + {i+1}?")

    # Split dataset
    jsonl_path = Path(temp_dir) / f"{session_id}.jsonl"
    result = dataset_splitter(input_path=str(jsonl_path), output_dir=temp_dir)

    assert result["status"] == "success"

    # Check split files were created
    assert (Path(temp_dir) / "train.jsonl").exists()
    assert (Path(temp_dir) / "valid.jsonl").exists()
    assert (Path(temp_dir) / "test.jsonl").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
