"""Unit tests for tool_inventory."""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

from tool_inventory import ToolInventory


class TestToolInventory(unittest.TestCase):
    """Tests for ToolInventory"""

    def test_tools_list_has_two_tools(self) -> None:
        inv = ToolInventory()
        self.assertEqual(len(inv.tools_list), 2)

    def test_tools_list_contains_get_weather(self) -> None:
        inv = ToolInventory()
        names = [t["name"] for t in inv.tools_list]
        self.assertIn("get_weather", names)

    def test_tools_list_contains_research_topic(self) -> None:
        inv = ToolInventory()
        names = [t["name"] for t in inv.tools_list]
        self.assertIn("research_topic", names)

    def test_get_weather_returns_empty(self) -> None:
        inv = ToolInventory()
        msg = inv.get_tool_pre_execute_msg("get_weather", {"location": "London"})
        self.assertEqual(msg, "")

    def test_research_topic_returns_message_with_topic(self) -> None:
        inv = ToolInventory()
        msg = inv.get_tool_pre_execute_msg(
            "research_topic", {"topic": "quantum computing"}
        )
        self.assertIn("Researching", msg)
        self.assertIn("quantum computing", msg)
        self.assertIn("Ctrl+C to cancel", msg)

    def test_get_weather_cancel_msg(self) -> None:
        inv = ToolInventory()
        msg = inv.get_tool_cancel_msg("get_weather", {"location": "Paris"})
        self.assertIn("Weather", msg)
        self.assertIn("cancelled", msg)

    def test_research_topic_cancel_msg(self) -> None:
        inv = ToolInventory()
        msg = inv.get_tool_cancel_msg("research_topic", {"topic": "AI"})
        self.assertIn("Research", msg)
        self.assertIn("cancelled", msg)

    def test_unknown_tool_raises_value_error(self) -> None:
        inv = ToolInventory()

        async def run() -> None:
            with self.assertRaises(ValueError) as ctx:
                await inv.execute_tool("unknown_tool", {})
            self.assertIn("Unknown tool", str(ctx.exception))

        asyncio.run(run())

    @patch.dict(os.environ, {"ELYOS_AI_API_KEY": "test-key"})
    @patch("tool_inventory.httpx.AsyncClient")
    def test_get_weather_success(self, mock_client_class: unittest.mock.MagicMock) -> None:
        mock_response = unittest.mock.MagicMock()
        mock_response.status_code = 200
        mock_response.text = "{}"
        mock_response.json.return_value = {
            "location": "London",
            "temperature_c": 5.4,
            "condition": "Partly cloudy",
            "humidity": 93
        }
        mock_client = unittest.mock.MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)

        inv = ToolInventory()

        async def run() -> None:
            result = await inv.execute_tool("get_weather", {"location": "London"})
            self.assertEqual(result["location"], "London")
            self.assertEqual(result["temperature_c"], 5.4)
            self.assertEqual(result["condition"], "Partly cloudy")
            self.assertEqual(result["humidity"], 93)

        asyncio.run(run())

    @patch.dict(os.environ, {"ELYOS_AI_API_KEY": "test-key"})
    @patch("tool_inventory.httpx.AsyncClient")
    def test_get_weather_http_error_returns_error_dict(
        self, mock_client_class: unittest.mock.MagicMock
    ) -> None:
        mock_response = unittest.mock.MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_client = unittest.mock.MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)

        inv = ToolInventory()

        async def run() -> None:
            result = await inv.execute_tool("get_weather", {"location": "London"})
            self.assertEqual(result["error"], "request_failed")
            self.assertEqual(result["status"], 500)

        asyncio.run(run())

    @patch.dict(os.environ, {"ELYOS_AI_API_KEY": "test-key"})
    @patch("tool_inventory.httpx.AsyncClient")
    def test_get_weather_throttled_returns_error_dict(
        self, mock_client_class: unittest.mock.MagicMock
    ) -> None:
        mock_response = unittest.mock.MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"status": "throttled"}'
        mock_response.json.return_value = {"status": "throttled"}
        mock_client = unittest.mock.MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)

        inv = ToolInventory()

        async def run() -> None:
            result = await inv.execute_tool("get_weather", {"location": "London"})
            self.assertEqual(result["error"], "request_throttled")
            self.assertEqual(result["status"], 429)

        asyncio.run(run())

    @patch.dict(os.environ, {"ELYOS_AI_API_KEY": "test-key"})
    @patch("tool_inventory.httpx.AsyncClient")
    def test_get_weather_timeout_returns_error_dict(
        self, mock_client_class: unittest.mock.MagicMock
    ) -> None:
        import httpx

        mock_client = unittest.mock.MagicMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)

        inv = ToolInventory()

        async def run() -> None:
            result = await inv.execute_tool("get_weather", {"location": "London"})
            self.assertEqual(result["error"], "request_timeout")
            self.assertIn("get_weather", result["detail"])

        asyncio.run(run())

    @patch.dict(os.environ, {"ELYOS_AI_API_KEY": "test-key"})
    @patch("tool_inventory.httpx.AsyncClient")
    def test_research_topic_success(self, mock_client_class: unittest.mock.MagicMock) -> None:
        mock_response = unittest.mock.MagicMock()
        mock_response.status_code = 200
        mock_response.text = "{}"
        mock_response.json.return_value = {    
            "topic": "quantum computing",    
            "summary": "Research summary for 'quantum computing'.",    
            "sources": [        
                "nature.com",        
                "sciencedirect.com",        
                "arxiv.org"    
                ],    
            "generated_at": "2026-02-19"
        }
        mock_client = unittest.mock.MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)

        inv = ToolInventory()

        async def run() -> None:
            result = await inv.execute_tool(
                "research_topic", {"topic": "quantum computing"}
            )
            self.assertEqual(result["topic"], "quantum computing")
            self.assertEqual(result["summary"], "Research summary for 'quantum computing'.")
            self.assertEqual(result["sources"], ["nature.com", "sciencedirect.com", "arxiv.org"])
            self.assertEqual(result["generated_at"], "2026-02-19")

        asyncio.run(run())

    @patch.dict(os.environ, {"ELYOS_AI_API_KEY": "test-key"})
    @patch("tool_inventory.httpx.AsyncClient")
    def test_research_topic_timeout_returns_error_dict(
        self, mock_client_class: unittest.mock.MagicMock
    ) -> None:
        import httpx

        mock_client = unittest.mock.MagicMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)

        inv = ToolInventory()

        async def run() -> None:
            result = await inv.execute_tool(
                "research_topic", {"topic": "AI"}
            )
            self.assertEqual(result["error"], "request_timeout")
            self.assertIn("research_topic", result["detail"])

        asyncio.run(run())

    @patch.dict(os.environ, {"ELYOS_AI_API_KEY": "test-key"})
    @patch("tool_inventory.httpx.AsyncClient")
    def test_execute_tool_cancelled_error_propagates(
        self, mock_client_class: unittest.mock.MagicMock
    ) -> None:
        mock_client = unittest.mock.MagicMock()
        mock_client.get = AsyncMock(side_effect=asyncio.CancelledError())
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)

        inv = ToolInventory()

        async def run() -> None:
            with self.assertRaises(asyncio.CancelledError):
                await inv.execute_tool("get_weather", {"location": "London"})

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
