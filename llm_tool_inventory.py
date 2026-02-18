"""LLM Tool definitions and execution.

Supports get_weather and research_topic tools.
Requires ELYOS_AI_API_KEY in the environment.
"""

import asyncio
import os
from typing import Any, Dict, List

import httpx

ELYOS_AI_DOMAIN_URL = "https://elyos-interview-907656039105.europe-west2.run.app"
API_GET_WEATHER_TIMEOUT_IN_SECS = 2
API_RESEARCH_TOPIC_TIMEOUT_IN_SECS = 10


class ToolInventory:
    """Tool inventory which LLM model can call"""

    def __init__(self) -> None:
        self.tools_list: List[Dict[str, Any]] = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get current weather for a city. Fast response.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, e.g. London, Tokyo",
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "research_topic",
                "description": "Research a topic in depth. Takes 3-8 seconds. Use for questions requiring detailed research.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Topic to research, e.g. 'solar energy', 'climate change'",
                        }
                    },
                    "required": ["topic"],
                    "additionalProperties": False,
                },
            },
        ]

    def get_tool_pre_execute_msg(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Return a short message to show before executing tool"""
        if tool_name == "get_weather":
            return ""  # Since tool is fast and has low timeout, no pre-execute message required
        elif tool_name == "research_topic":
            topic = args.get("topic", "Some topic")
            return f"Researching {topic}... (Ctrl+C to cancel)\n"
        return ""

    def get_tool_cancel_msg(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Message to show when the tool run is explicitly cancelled by the user."""
        if tool_name == "get_weather":
            return "\nWeather tool cancelled."
        elif tool_name == "research_topic":
            return "\nResearch tool cancelled."
        return ""

    async def execute_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        if tool_name == "get_weather":
            return await _get_weather(**args)
        elif tool_name == "research_topic":
            return await _research_topic(**args)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")


# Tools implementation


async def _get_weather(location: str) -> Dict[str, Any]:
    return await _get_http_request(
        url=f"{ELYOS_AI_DOMAIN_URL}/weather",
        headers={"X-API-Key": os.environ["ELYOS_AI_API_KEY"]},
        params={"location": location},
        timeout_in_secs=API_GET_WEATHER_TIMEOUT_IN_SECS,
        source_function_name="get_weather",
    )


async def _research_topic(topic: str) -> Dict[str, Any]:
    return await _get_http_request(
        url=f"{ELYOS_AI_DOMAIN_URL}/research",
        headers={"X-API-Key": os.environ["ELYOS_AI_API_KEY"]},
        params={"topic": topic},
        timeout_in_secs=API_RESEARCH_TOPIC_TIMEOUT_IN_SECS,
        source_function_name="research_topic",
    )


async def _get_http_request(
    url: str,
    headers: Dict[str, str],
    params: Dict[str, str],
    timeout_in_secs: float,
    source_function_name: str,
) -> Dict[str, Any]:
    """Perform a HTTP GET request; returns JSON body as dict or an error dict."""
    try:
        async with httpx.AsyncClient(timeout=timeout_in_secs) as client:
            r = await client.get(
                url=url,
                params=params,
                headers=headers,
            )

            if r.status_code >= 400:
                return {
                    "error": "request_failed",
                    "status": r.status_code,
                    "body": r.text,
                }

            json_resp = r.json()
            if isinstance(json_resp, dict) and json_resp.get("status") == "throttled":
                return {
                    "error": "request_throttled",
                    "status": 429,
                    "body": r.text,
                }

            return json_resp

    except asyncio.CancelledError:
        raise

    except httpx.TimeoutException:
        return {
            "error": "request_timeout",
            "detail": f"Request timed out for {source_function_name}",
        }

    except Exception as e:
        return {"error": "request_failed", "detail": str(e)}
