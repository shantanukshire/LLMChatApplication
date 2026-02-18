"""LLM chat application with tool support.

Runs an interactive loop that streams LLM responses and supports
Ctrl+C cancellation.
"""

import asyncio
import os
import sys
from typing import Any, Dict, List

from llm_model import LLMModel


async def get_user_input() -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, "You: ")


async def stream_llm_response(
    llm_model: LLMModel,
    user_query: str,
    conversation_history: List[Dict[str, Any]],
) -> None:
    """Stream the LLM response for the given query and append to conversation history.

    Prints chunks as they arrive. Mutates conversation_history in place by
    appending the user message and assistant response on success.
    If cancelled then history is not updated.
    """
    assistant_text_chunks = []
    print("Assistant: ", end="", flush=True)

    async for chunk in llm_model.stream_response(user_query, conversation_history):
        print(chunk, end="", flush=True)
        assistant_text_chunks.append(chunk)

    # In case of cancellation, conversation history will not be updated which is intentional
    assistant_text = "".join(assistant_text_chunks)
    conversation_history.append({"role": "user", "content": user_query})
    conversation_history.append({"role": "assistant", "content": assistant_text})


def check_required_env() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        print("Missing required environment variable: OPENAI_API_KEY")
        print("Set it before running the program.")
        sys.exit(1)

    if "ELYOS_AI_API_KEY" not in os.environ:
        print("Missing required environment variable: ELYOS_AI_API_KEY")
        print("Set it before running the program.")
        sys.exit(1)


async def main() -> None:
    llm_model = LLMModel()
    conversation_history = []

    while True:
        query = await get_user_input()
        if query.lower() in ["quit", "exit", "q"]:
            break

        stream_task = asyncio.create_task(
            stream_llm_response(llm_model, query, conversation_history)
        )

        try:
            await stream_task
        except asyncio.CancelledError:
            # User pressed Ctrl+C during streaming or tool execution
            print("\nOperation cancelled.", flush=True)
            # Ensure task is cancelled
            if not stream_task.done():
                stream_task.cancel()
                try:
                    await stream_task
                except asyncio.CancelledError:
                    pass
        except KeyboardInterrupt:
            # Handle Ctrl+C during async operations
            print("\n\nCancelling...", flush=True)
            stream_task.cancel()
            try:
                await stream_task
            except (asyncio.CancelledError, KeyboardInterrupt):
                pass
            print("Operation cancelled.", flush=True)

        print("\n")


if __name__ == "__main__":
    check_required_env()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
