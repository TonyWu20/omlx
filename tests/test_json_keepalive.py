# SPDX-License-Identifier: Apache-2.0
"""Tests for _with_json_keepalive non-streaming wrapper."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from omlx.server import _with_json_keepalive


async def _collect(gen):
    """Collect all chunks from an async generator."""
    chunks = []
    async for chunk in gen:
        chunks.append(chunk)
    return chunks


class TestJsonKeepaliveDisconnectDebounce:
    """Tests for debounce behavior in _with_json_keepalive.

    The debounce invariant: disconnect is only confirmed after
    _DISCONNECT_CONFIRM (3) consecutive True returns from is_disconnected().
    A single False resets the counter.
    """

    @pytest.mark.asyncio
    async def test_single_true_then_false_not_cancelled(self):
        """Single True followed by False should NOT cancel — debounce requires 3 consecutive."""
        mock_request = AsyncMock()
        # True once, then False, then coroutine completes
        mock_request.is_disconnected = AsyncMock(side_effect=[True, False])

        async def slow_coro():
            await asyncio.sleep(0.05)
            return '{"result": "ok"}'

        chunks = await _collect(
            _with_json_keepalive(mock_request, slow_coro(), disconnect_poll=0.01)
        )

        # The final JSON must appear — generator was NOT cancelled
        full_response = "".join(chunks)
        assert '{"result": "ok"}' in full_response, (
            f"Expected JSON result in response (not cancelled); got: {full_response!r}"
        )

    @pytest.mark.asyncio
    async def test_three_consecutive_true_is_cancelled(self):
        """Three consecutive True returns must cancel the coroutine."""
        mock_request = AsyncMock()
        # Three True in a row — debounce threshold reached
        mock_request.is_disconnected = AsyncMock(side_effect=[True, True, True])

        async def slow_coro():
            await asyncio.sleep(999)
            return "unreachable"

        chunks = await _collect(
            _with_json_keepalive(mock_request, slow_coro(), disconnect_poll=0.01)
        )

        # is_disconnected must have been called at least 3 times before cancel
        assert mock_request.is_disconnected.call_count >= 3, (
            f"Expected is_disconnected called >=3 times for debounce, "
            f"got {mock_request.is_disconnected.call_count}"
        )
        # The unreachable return value must NOT appear
        full_response = "".join(chunks)
        assert "unreachable" not in full_response

    @pytest.mark.asyncio
    async def test_false_resets_counter_then_three_true_cancels(self):
        """A False resets the debounce counter; cancellation requires 3 NEW consecutive True.

        Sequence: True, True, False, True, True, True
        - After first two True: counter=2, NOT cancelled (needs 3)
        - False: counter resets to 0
        - Final three True: counter reaches 3, cancelled
        """
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(
            side_effect=[True, True, False, True, True, True]
        )

        async def slow_coro():
            await asyncio.sleep(999)
            return "unreachable"

        chunks = await _collect(
            _with_json_keepalive(mock_request, slow_coro(), disconnect_poll=0.01)
        )

        # Must have consumed all 6 side_effect values before cancelling
        assert mock_request.is_disconnected.call_count == 6, (
            f"Expected exactly 6 is_disconnected calls (True,True,False,True,True,True), "
            f"got {mock_request.is_disconnected.call_count}"
        )
        full_response = "".join(chunks)
        assert "unreachable" not in full_response
