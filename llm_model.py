"""LLM model integration.

Uses AsyncOpenAI for streaming responses and tool execution.
Requires OPENAI_API_KEY in the environment.
"""

import asyncio
import json
import os

from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI

from llm_tool_inventory import ToolInventory

MODEL_NAME = "gpt-5.2"
MAX_TOOL_ROUNDS = 3
GLOBAL_TOOL_EXECUTION_TIMEOUT_IN_SECS = 15


class LLMModel:
    """LLM integration using OpenAI model."""

    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.tool_inventory = ToolInventory()

    async def stream_response(
        self,
        new_prompt: str,
        conversation_history: List[Dict[str, Any]],
    ) -> AsyncIterator[str]:
        """Stream model output for the new user prompt.

        Tool call execution handled internally, including tool-call announcement and tool status messages.
        """
        model_input = conversation_history + [{"role": "user", "content": new_prompt}]
        previous_response_id: Optional[str] = None
        iteration = 0

        while True:
            iteration += 1

            if iteration > MAX_TOOL_ROUNDS:
                raise RuntimeError(
                    "Too many tool-call rounds."
                )  # Safeguard against infinite loops

            response_stream = await self.client.responses.create(
                model=MODEL_NAME,
                input=model_input,
                tools=self.tool_inventory.tools_list,
                tool_choice="auto",
                previous_response_id=previous_response_id,
                stream=True,
            )
            pending_function_calls: Dict[str, object] = {}
            current_response_id: Optional[str] = None

            try:
                async for event in response_stream:
                    event_type = getattr(event, "type", None)

                    # Response creation event
                    if event_type == "response.created":
                        resp = getattr(event, "response", None)
                        current_response_id = (
                            getattr(resp, "id", None) if resp else None
                        )
                        if current_response_id is None:
                            current_response_id = getattr(event, "response_id", None)

                    # Text response event
                    if event_type == "response.output_text.delta":
                        yield event.delta

                    # Function call events
                    elif event_type == "response.output_item.added":
                        item = getattr(event, "item", None)
                        if item and getattr(item, "type", None) == "function_call":
                            func_name = getattr(item, "name", None)
                            yield f"[calls {func_name}]\n"

                    elif event_type == "response.output_item.done":
                        item = getattr(event, "item", None)
                        if item and getattr(item, "type", None) == "function_call":
                            pending_function_calls[item.call_id] = item

                    # Error event
                    elif event_type == "response.error":
                        msg = (
                            getattr(event, "error_message", None)
                            or "OpenAI streaming error for llm_model " + MODEL_NAME
                        )
                        raise RuntimeError(msg)

            except asyncio.CancelledError:
                raise

            finally:
                # if aclose() exists close it.
                aclose = getattr(response_stream, "aclose", None)
                if callable(aclose):
                    try:
                        await aclose()
                    except Exception:
                        pass

            if not pending_function_calls:
                return

            # Must have the response id that created those calls
            if not current_response_id:
                raise RuntimeError(
                    "Did not receive response_id; cannot submit tool outputs."
                )

            previous_response_id = current_response_id
            function_outputs = []

            for func_call in pending_function_calls.values():
                raw_args = func_call.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}

                yield self.tool_inventory.get_tool_pre_execute_msg(func_call.name, args)
                try:
                    result = await asyncio.wait_for(
                        self.tool_inventory.execute_tool(func_call.name, args),
                        timeout=GLOBAL_TOOL_EXECUTION_TIMEOUT_IN_SECS,
                    )
                except asyncio.CancelledError:
                    # Task cancelled due to user interruption (Ctrl+C) during tool execution
                    yield self.tool_inventory.get_tool_cancel_msg(func_call.name, args)
                    raise
                except asyncio.TimeoutError:
                    result = {"error": f"{func_call.name} timed out"}
                except Exception as e:
                    result = {"error": str(e)}

                function_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": func_call.call_id,
                        "output": json.dumps(result),
                    }
                )

            # Provide function outputs as new input to the model
            model_input = function_outputs
