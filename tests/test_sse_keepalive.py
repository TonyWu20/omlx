# SPDX-License-Identifier: Apache-2.0
"""Tests for _with_sse_keepalive SSE wrapper."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from omlx.server import _with_sse_keepalive


async def _collect(gen):
    """Collect all items from an async generator."""
    items = []
    async for item in gen:
        items.append(item)
    return items


class TestSSEKeepaliveExceptionHandling:
    """Tests for exception handling in _with_sse_keepalive."""

    @pytest.mark.asyncio
    async def test_normal_generator_passes_through(self):
        """Normal generator items should pass through unchanged."""

        async def gen():
            yield "data: chunk1\n\n"
            yield "data: chunk2\n\n"

        items = await _collect(_with_sse_keepalive(gen()))
        # First item is always the initial keepalive
        assert items[0] == ": keep-alive\n\n"
        assert "data: chunk1\n\n" in items
        assert "data: chunk2\n\n" in items

    @pytest.mark.asyncio
    async def test_generator_exception_yields_error_sse(self):
        """When inner generator raises, keepalive wrapper should yield
        error SSE data and [DONE] instead of propagating the exception."""

        async def gen():
            yield "data: first_chunk\n\n"
            raise RuntimeError("Memory limit exceeded during prefill")

        items = await _collect(_with_sse_keepalive(gen()))

        # Should contain initial keepalive + first chunk + error + done
        assert items[0] == ": keep-alive\n\n"
        assert "data: first_chunk\n\n" in items

        # Find the error SSE event
        error_items = [i for i in items if i.startswith("data: {")]
        assert len(error_items) == 1
        error_data = json.loads(error_items[0].removeprefix("data: ").strip())
        assert "error" in error_data
        assert "Memory limit exceeded during prefill" in error_data["error"]["message"]
        assert error_data["error"]["type"] == "server_error"

        # Must end with [DONE]
        assert "data: [DONE]\n\n" in items

    @pytest.mark.asyncio
    async def test_generator_exception_before_any_yield(self):
        """Exception on first iteration should still produce error SSE."""

        async def gen():
            if True:
                raise ValueError("Block allocation failed")
            yield  # unreachable, but makes this an async generator

        items = await _collect(_with_sse_keepalive(gen()))

        assert items[0] == ": keep-alive\n\n"

        error_items = [i for i in items if i.startswith("data: {")]
        assert len(error_items) == 1
        error_data = json.loads(error_items[0].removeprefix("data: ").strip())
        assert "Block allocation failed" in error_data["error"]["message"]
        assert "data: [DONE]\n\n" in items

    @pytest.mark.asyncio
    async def test_empty_generator_completes_cleanly(self):
        """Empty generator should complete without errors."""

        async def gen():
            return
            yield  # make it an async generator

        items = await _collect(_with_sse_keepalive(gen()))
        assert items[0] == ": keep-alive\n\n"
        # No error items
        error_items = [i for i in items if i.startswith("data: {")]
        assert len(error_items) == 0


class TestSSEKeepaliveDisconnectDebounce:
    """Tests for debounce behavior in _with_sse_keepalive.

    The debounce invariant: disconnect is only confirmed after
    _DISCONNECT_CONFIRM (3) consecutive True returns from is_disconnected().
    A single False resets the counter.
    """

    @pytest.mark.asyncio
    async def test_single_true_then_false_not_cancelled(self):
        """Single True followed by False should NOT cancel — debounce requires 3 consecutive."""
        mock_request = AsyncMock()
        # True once, then False, then the generator completes
        mock_request.is_disconnected = AsyncMock(side_effect=[True, False])

        completed = False

        async def slow_gen():
            nonlocal completed
            # Yield after a brief delay so the disconnect poll fires at least once
            await asyncio.sleep(0.05)
            completed = True
            yield "data: done\n\n"

        items = await _collect(
            _with_sse_keepalive(
                slow_gen(),
                http_request=mock_request,
                disconnect_poll=0.01,
            )
        )

        # Generator must have completed and its item must appear — not cancelled
        assert completed, "Generator should have completed (not been cancelled)"
        assert any("done" in item for item in items), (
            "Expected generator output in items; got cancelled instead"
        )

    @pytest.mark.asyncio
    async def test_three_consecutive_true_is_cancelled(self):
        """Three consecutive True returns must cancel the generator."""
        mock_request = AsyncMock()
        # Three True in a row — debounce threshold reached
        mock_request.is_disconnected = AsyncMock(side_effect=[True, True, True])

        async def slow_gen():
            event = asyncio.Event()
            await event.wait()  # never resolves during test
            yield "unreachable"  # pragma: no cover

        items = await _collect(
            _with_sse_keepalive(
                slow_gen(),
                http_request=mock_request,
                disconnect_poll=0.01,
            )
        )

        # is_disconnected must have been called at least 3 times before cancel
        assert mock_request.is_disconnected.call_count >= 3, (
            f"Expected is_disconnected called >=3 times for debounce, "
            f"got {mock_request.is_disconnected.call_count}"
        )
        # The unreachable yield must NOT appear
        assert not any("unreachable" in item for item in items)

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

        async def slow_gen():
            event = asyncio.Event()
            await event.wait()  # never resolves during test
            yield "unreachable"  # pragma: no cover

        items = await _collect(
            _with_sse_keepalive(
                slow_gen(),
                http_request=mock_request,
                disconnect_poll=0.01,
            )
        )

        # Must have consumed all 6 side_effect values before cancelling
        assert mock_request.is_disconnected.call_count == 6, (
            f"Expected exactly 6 is_disconnected calls (True,True,False,True,True,True), "
            f"got {mock_request.is_disconnected.call_count}"
        )
        assert not any("unreachable" in item for item in items)
