"""MLX Session Manager - Model-agnostic training data export.

Uses tokenizer's native chat template for formatting - compatible with mlx-lm training.
"""

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from strands.agent.agent import Agent
    from strands.types.content import Message

from strands.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


class MLXSessionManager(SessionManager):
    """MLX-LM compatible session manager - uses tokenizer's chat template."""

    def __init__(
        self,
        session_id: str,
        tokenizer: Any = None,
        model_id: Optional[str] = None,
        storage_dir: Optional[str] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize MLX session manager.

        Args:
            session_id: Unique identifier for this session
            tokenizer: Optional tokenizer (will be captured from agent if not provided)
            model_id: Optional model ID to load tokenizer from (e.g., "mlx-community/Qwen3-1.7B-4bit")
            storage_dir: Directory to store training data (default: ~/.strands/mlx_training_data)
            add_generation_prompt: Whether to add generation prompt in template
            tokenize: Whether to tokenize (False for text output)
            system_prompt: Optional system prompt (will be captured from agent if not provided)
            **kwargs: Additional args passed to apply_chat_template
        """
        self.session_id = session_id
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.add_generation_prompt = add_generation_prompt
        self.tokenize = tokenize
        self.system_prompt = system_prompt
        self.template_kwargs = kwargs

        # Storage directory
        self.storage_dir = storage_dir or os.path.expanduser("~/.strands/mlx_training_data")
        os.makedirs(self.storage_dir, exist_ok=True)

        # JSONL file path
        self.jsonl_path = os.path.join(self.storage_dir, f"{session_id}.jsonl")

        # Agent reference (captured during initialize)
        self.agent = None

        # Track last message count to detect new conversations
        self.last_message_count = 0

        logger.info(f"MLXSessionManager: session_id={session_id}, output={self.jsonl_path}")

    def _convert_strands_messages_to_chat_format(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Convert Strands messages to standard chat format.

        Args:
            messages: List of Strands Message objects or dicts

        Returns:
            List of dicts in format: {"role": "...", "content": "..."}
        """
        chat_messages = []
        tools = []

        for msg in messages:
            role = msg.get("role") if isinstance(msg, dict) else msg.role
            content = msg.get("content") if isinstance(msg, dict) else msg.content

            if role == "user":
                # Handle user messages
                message = {"role": "user", "content": ""}

                if isinstance(content, list):
                    # Collect text and tool results separately
                    text_parts = []
                    tool_results = []

                    for item in content:
                        if isinstance(item, dict):
                            if "text" in item:
                                text_parts.append(item["text"])
                            elif "toolResult" in item:
                                tool_results.append(item["toolResult"])

                    # Combine text
                    message["content"] = "".join(text_parts)

                # Add user message first
                if message["content"]:
                    chat_messages.append(message)

                # Add tool results as separate "tool" role messages (ChatML format)
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "toolResult" in item:
                            tool_result = item["toolResult"]
                            result_content = tool_result.get("content", [])
                            result_text = []
                            for result_item in result_content:
                                if isinstance(result_item, dict) and "text" in result_item:
                                    result_text.append(result_item["text"])

                            if result_text:
                                # Use proper "tool" role - let tokenizer's chat template handle formatting
                                chat_messages.append(
                                    {
                                        "role": "tool",
                                        "content": "".join(result_text),
                                        # Include tool_call_id if available
                                        "tool_call_id": tool_result.get("toolUseId", ""),
                                    }
                                )

            elif role == "assistant":
                # Handle assistant messages
                message = {"role": "assistant", "content": ""}
                tool_calls = []

                if isinstance(content, list):
                    text_parts = []

                    for item in content:
                        if isinstance(item, dict):
                            if "text" in item:
                                text_parts.append(item["text"])
                            elif "toolUse" in item:
                                # Extract tool call with proper ID for ChatML
                                tool_use = item["toolUse"]
                                tool_call = {
                                    "id": tool_use.get("toolUseId", ""),  # Include ID
                                    "type": "function",  # OpenAI format
                                    "function": {
                                        "name": tool_use.get("name"),
                                        "arguments": json.dumps(
                                            tool_use.get("input", {})
                                        ),  # JSON string
                                    },
                                }
                                tool_calls.append(tool_call)

                    message["content"] = "".join(text_parts)

                # Add tool calls if present
                if tool_calls:
                    message["tool_calls"] = tool_calls

                chat_messages.append(message)

        return chat_messages

    def _extract_tool_specs(self, agent: "Agent") -> Optional[List[Dict[str, Any]]]:
        """Extract tool specifications from agent in OpenAI format.

        Args:
            agent: The Strands agent

        Returns:
            List of tool specs in OpenAI format, or None if no tools
        """
        if not hasattr(agent, "tools") or not agent.tools:
            return None

        tools = []
        for tool in agent.tools:
            try:
                tool_spec = tool.to_strands_tool()
                if isinstance(tool_spec, dict) and "toolSpec" in tool_spec:
                    spec = tool_spec["toolSpec"]
                    # Convert to OpenAI format
                    tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": spec.get("name", "unknown"),
                                "description": spec.get("description", ""),
                                "parameters": spec.get("inputSchema", {}).get("json", {}),
                            },
                        }
                    )
            except Exception as e:
                logger.warning(f"Could not serialize tool: {e}")

        return tools if tools else None

    def _format_conversation(self, agent: "Agent") -> str:
        """Format agent's message history using tokenizer's chat template.

        Args:
            agent: The Strands agent

        Returns:
            Formatted text string ready for training
        """
        # Get current system prompt from agent (captures dynamic prompts)
        current_system_prompt = None
        if hasattr(agent, "system_prompt") and agent.system_prompt:
            current_system_prompt = agent.system_prompt
        elif self.system_prompt:
            current_system_prompt = self.system_prompt
        else:
            current_system_prompt = "You are a helpful AI assistant."

        # Convert Strands messages to chat format
        chat_messages = self._convert_strands_messages_to_chat_format(agent.messages)

        # Prepend system message - matches mlx-lm pattern
        if current_system_prompt:
            chat_messages.insert(0, {"role": "system", "content": current_system_prompt})

        # Extract tools if available
        tools = self._extract_tool_specs(agent)

        # Use tokenizer's native chat template
        try:
            formatted_text = self.tokenizer.apply_chat_template(
                chat_messages,
                tools=tools,
                tokenize=self.tokenize,
                add_generation_prompt=self.add_generation_prompt,
                **self.template_kwargs,
            )

            # If tokenized, decode back to text
            if self.tokenize:
                formatted_text = self.tokenizer.decode(formatted_text)

            return formatted_text

        except Exception as e:
            logger.error(f"Error applying chat template: {e}", exc_info=True)
            raise

    def initialize(self, agent: "Agent", **kwargs: Any) -> None:
        """Initialize with agent reference.

        Args:
            agent: The Strands agent
            **kwargs: Additional initialization arguments
        """
        self.agent = agent

        # Capture system prompt if not provided
        if not self.system_prompt and hasattr(agent, "system_prompt"):
            self.system_prompt = agent.system_prompt or "You are a helpful AI assistant."

        # Capture tokenizer if not provided
        if self.tokenizer is None:
            # Try loading from model_id first
            if self.model_id:
                try:
                    from transformers import AutoTokenizer

                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                    logger.info(f"Loaded tokenizer from model_id: {self.model_id}")
                except Exception as e:
                    logger.warning(f"Failed to load tokenizer from model_id: {e}")
                    # Continue to try agent.model.tokenizer

            # Fallback to agent's model tokenizer
            if self.tokenizer is None:
                if hasattr(agent, "model") and hasattr(agent.model, "tokenizer"):
                    self.tokenizer = agent.model.tokenizer
                    logger.info("Captured tokenizer from agent.model.tokenizer")
                else:
                    raise ValueError(
                        "No tokenizer available. Provide one of:\n"
                        "  1. tokenizer parameter in __init__\n"
                        "  2. model_id parameter to load tokenizer\n"
                        "  3. Agent with model.tokenizer attribute"
                    )

        # Verify tokenizer has chat template
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise ValueError(
                "Tokenizer does not have apply_chat_template method. "
                "Use a HuggingFace tokenizer or mlx-lm TokenizerWrapper."
            )

        logger.info(f"Initialized MLX session for agent {agent.agent_id}")

    def append_message(self, message: "Message", agent: "Agent", **kwargs: Any) -> None:
        """Hook called when message is added - we use sync_agent instead."""
        pass

    def sync_agent(self, agent: "Agent", **kwargs: Any) -> None:
        """Save complete conversation ONLY after final assistant response.

        Args:
            agent: The Strands agent
            **kwargs: Additional sync arguments
        """
        # Check if we have new messages
        current_message_count = len(agent.messages)

        if current_message_count <= self.last_message_count:
            return  # No new messages

        # Check if last message is from assistant with actual text content
        if current_message_count > 0:
            last_msg = agent.messages[-1]
            last_role = last_msg.get("role") if isinstance(last_msg, dict) else last_msg.role

            # Only save if last message is from assistant
            if last_role != "assistant":
                return  # Wait for assistant response

            # Check if assistant message has text content (not just tool call)
            last_content = (
                last_msg.get("content") if isinstance(last_msg, dict) else last_msg.content
            )

            has_text = False
            has_only_tool_use = True

            if isinstance(last_content, list):
                for item in last_content:
                    if isinstance(item, dict):
                        if "text" in item and item["text"].strip():
                            has_text = True
                            has_only_tool_use = False
                        elif "toolUse" not in item:
                            has_only_tool_use = False

            # Skip if no text or only tool calls (wait for final response)
            if not has_text or has_only_tool_use:
                logger.debug(
                    f"Skipping save - waiting for final assistant response "
                    f"(has_text={has_text}, only_tool={has_only_tool_use})"
                )
                return

        # We have complete exchange - format and save
        try:
            formatted_text = self._format_conversation(agent)

            # Create JSONL entry (mlx-lm training format)
            jsonl_entry = {"text": formatted_text}

            # Append to file
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                json.dump(jsonl_entry, f, ensure_ascii=False)
                f.write("\n")

            logger.info(
                f"✅ Saved conversation: {len(formatted_text)} chars, "
                f"{current_message_count} messages"
            )

            # Update last message count
            self.last_message_count = current_message_count

        except Exception as e:
            logger.error(f"Error saving conversation: {e}", exc_info=True)

    def redact_latest_message(
        self, redact_message: "Message", agent: "Agent", **kwargs: Any
    ) -> None:
        """Redaction not supported for training data."""
        logger.warning("Redaction not supported for MLX training data export")

    def get_jsonl_path(self) -> str:
        """Get path to JSONL file.

        Returns:
            Path to the JSONL training data file
        """
        return self.jsonl_path

    def get_example_count(self) -> int:
        """Get number of examples saved.

        Returns:
            Number of training examples in the JSONL file
        """
        if not os.path.exists(self.jsonl_path):
            return 0
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MLXSessionManager(session_id='{self.session_id}', "
            f"examples={self.get_example_count()})"
        )
