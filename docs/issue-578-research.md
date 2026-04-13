# Research: Issue #578 — Long-context agent streaming aborts (GeneratorExit / prefill interrupted)

## Context

Issue jundot/omlx#578 reports agentic client (Claude Code) sessions ending mid-task:
- Repeated `[vlm_stream_generate] GeneratorExit for request ...` → `Prefill interrupted`
- Or: session "finishes" around the 5-minute mark claiming task done with no actual file changes

**Bisect:** `0.3.5.dev2` (commit `3ed1057`, local observation) regresses. Commit `b8a9c5a` (upstream HEAD) works.
The 3 commits between them (`ce1e517`, `512c21b`, `b8a9c5a`) are what fixed it.

A secondary, fully reproducible issue: `/v1/messages` returns 422 for Anthropic built-in tool types.

---

## Root Cause Analysis

### There are TWO distinct failure modes in this issue

---

### Failure Mode A — "Session ends ~5 min, claims done, no file changes" (PRIMARY for user's experience)

**Introduced by:** `7316ffa` ("fix(vlm): use mlx-lm decode model for batch=1, 2x VLM generation speed")
**Fixed by:** `512c21b` ("feat(vlm): per-request mRoPE position tracking for batched VLM decode")

**Root cause:**

`7316ffa` changed `omlx/models/vlm.py` to enable the fast `decode_model` path for all batch sizes
(`input_ids.shape[0] > 1` → `>= 1`). This is correct for standard RoPE models but catastrophically
wrong for **mRoPE models** (Qwen3.5-35B, Qwen3.5-27B, Qwen3-VL, GLM-4.6V).

mRoPE models use 3D position IDs (temporal/height/width). The mlx-lm `decode_model` uses standard
1D RoPE — it reads `cache.offset` as a scalar and produces wrong positions for all tokens after
the prefill. This causes the model to generate **semantically garbage tokens**: coherent-looking
text that makes no logical sense relative to the context.

In an agentic workflow:
- Claude Code sends a long subagent task (100k+ token context)
- The model generates garbage during decode → the output looks like a plausible but wrong "task complete" message
- Claude Code's outer loop sees a finished response and moves on — no `GeneratorExit`, no server error
- The session terminates around 5 minutes (when long-context decode of ~131k window completes)
  with no actual work done

This also explains the `is_disconnected()` / `GeneratorExit` seen in server logs: the decode output
was nonsense, Claude Code gave up and closed the connection during generation, triggering the abort.

**`512c21b` fix:**
Added `self._uses_mrope` detection at `VLMModelAdapter` init (`omlx/models/vlm.py:~163`).
Changed the fast decode path condition to `if self._decode_model is not None and not self._uses_mrope`.
Added a full per-request mRoPE decode path using `_batch_rope_deltas` for correct 3D positions.

**Why `b8a9c5a` works:** It includes `512c21b` which restores correct decode for mRoPE models.

**Why `0.3.5.dev2` breaks:** It includes `7316ffa` (introduced the bug) but not `512c21b` (the fix).

---

### Failure Mode B — Spurious `GeneratorExit` / `Prefill interrupted` (SECONDARY, non-deterministic)

**Present in all versions including `b8a9c5a`**

`_with_sse_keepalive` (`server.py:1262–1272`) polls `is_disconnected()` every 2 s.
Starlette's `Request.is_disconnected()` is known to return `True` spuriously when uvicorn's
internal receive channel is drained (encode/starlette#1677).

One false positive immediately fires `task.cancel()` — no debouncing, no grace period.
During 2–4 min prefill (110k tokens), ~60–120 polls occur. A single false positive kills the request.

Same pattern in `_run_with_disconnect_guard` (1 s poll) and `_with_json_keepalive` (2 s poll).

This explains the non-determinism observed in the issue report and explains why even with
`b8a9c5a` some users may still see occasional `GeneratorExit` aborts.

---

### Secondary Issue (reproducible): 422 on `/v1/messages` with Anthropic built-in tools

**File:** `omlx/api/anthropic_models.py:114–120`

Anthropic built-in tool types (e.g. `web_search_20250305`) omit `input_schema` and include a
`type` field. `AnthropicTool.input_schema` is required with no default → Pydantic `ValidationError`
→ FastAPI 422. Present in all versions.

---

## Proposed Fixes

### Fix 1 (Failure Mode A — CRITICAL): Already fixed in `512c21b`

The mRoPE decode corruption is fully fixed upstream by `512c21b`. Users on `0.3.5.dev2` or
the `v0.3.5.dev1` DMG need to update to the tip that includes `512c21b` and `b8a9c5a`.

No code change needed in this repo — it is already at `b8a9c5a`.

**Affected models:** Any model with `"mrope_section"` in `text_config.rope_scaling` or
`text_config.rope_parameters`. Confirmed: Qwen3.5-35B, Qwen3.5-27B, Qwen3-VL, GLM-4.6V.

### Fix 2 (Failure Mode B): Add disconnect confirmation counter

**Target files:**
- `omlx/server.py:1262–1272` (`_with_sse_keepalive`)
- `omlx/server.py:1315–1325` (`_run_with_disconnect_guard`)
- `omlx/server.py:1357–1367` (`_with_json_keepalive`)

Replace zero-debounce single-poll cancel with a consecutive-confirmation counter
(require 3 consecutive `True` results, any `False` resets the counter):

```python
# Add above the is_disconnected block:
disconnect_count = 0
_DISCONNECT_CONFIRM = 3

# Replace the existing disconnected check:
if disconnected:
    disconnect_count += 1
    if disconnect_count >= _DISCONNECT_CONFIRM:
        logger.info("Client disconnected (confirmed), cancelling")
        task.cancel()
        ...
        return
    else:
        logger.debug("is_disconnected() True (%d/%d), waiting for confirmation",
                     disconnect_count, _DISCONNECT_CONFIRM)
else:
    disconnect_count = 0
```

---

## Relationship to Issue #548 (SpecPrefill + tool call / file edit failures)

### Short answer

**Yes, #548 is the same mRoPE incompatibility class of bug, but via a different code path —
and the simplest fix (skip SpecPrefill for mRoPE models) does disable SpecPrefill entirely
for Qwen3.5 series.**

### How #548 differs from Failure Mode A

| | #578 Failure Mode A | #548 |
|---|---|---|
| Trigger | `0.3.5.dev2` + mRoPE model + any request | SpecPrefill ON + mRoPE model + long prompt |
| Mechanism | `decode_model` path passes wrong 1D offset to mRoPE cache | `_PositionMappedRoPE` in `sparse_prefill()` applies standard 1D `manual_rope()` to mRoPE attention layers |
| Corruption scope | Per-request decode only | **Global**: `_OffsetAdjustedRoPE` stays installed on model's attention layers across requests until `cleanup_rope()` is called |
| Onset | Immediately on first decode token | After 3-4 messages once SpecPrefill threshold triggers |
| Fixed by `512c21b`? | Yes | **No** |

### Root cause of #548 in detail

`sparse_prefill()` (`omlx/patches/specprefill.py:~585`) patches all full-attention layers
with `_PositionMappedRoPE`, which calls:

```python
manual_rope(x, positions, self._dims, base=self._base, scale=self._scale)
```

This bypasses the mRoPE module entirely and applies standard 1D sinusoidal RoPE using
pre-computed scalar positions. For Qwen3.5's `Qwen3NextAttention`, the real `self.rope`
module internally handles `mrope_section = [11, 11, 10]` splits to produce 3D-correct
embeddings. `manual_rope` does not — it applies identical rotations across all 3 sections,
producing wrong key/query representations in the KV cache.

After `sparse_prefill()`, `_OffsetAdjustedRoPE` is installed globally on the model's attention
layers. This wrapper remains active across decode steps of the current request and — if
`cleanup_rope()` is skipped (e.g. on abort, scheduler reset, or exception) — persists into
**subsequent requests**. This explains the "works for 3-4 messages then permanently breaks" pattern.

Importantly: `VLMModelAdapter.layers` returns `self._language_model.model.layers` — the actual
Qwen3.5 `DecoderLayer` objects. `_find_attention_layers()` in `specprefill.py` finds layers with
`hasattr(layer, "self_attn")`, which correctly identifies only the full-attention layers (every 4th
in Qwen3.5, with `full_attention_interval=4`). SSM/GatedDeltaNet layers are skipped. So the RoPE
patch correctly targets only full-attention layers — the problem is the content of the patch, not
its scope.

### Revised root cause for #548 (after deeper investigation)

The RoPE patching itself is actually **correct** for Qwen3.5. Here's why:

`initialize_rope` with `rope_type == "mrope"` in mlx-lm `rope_utils.py:301` returns a **plain
`nn.RoPE`** — the `mrope_section` config is parsed but the returned module is standard 1D RoPE.
`Qwen3NextAttention.rope` is therefore `nn.RoPE` taking a scalar `offset`. `manual_rope()` in
`_PositionMappedRoPE` uses the same `dims`/`base`/`scale` from that `nn.RoPE` — identical math.
`_OffsetAdjustedRoPE` passes `offset + adjustment` as a scalar to `nn.RoPE`. Both are correct.

**The real bug for #548 is a missing `_cleanup_specprefill` call in the abort path.**

`_cleanup_specprefill(request_id)` is called **only** at `scheduler.py:3298` in the normal
finish path. It is **never called** from `_do_abort_request()` (`scheduler.py:2553`).

When a SpecPrefill request is aborted mid-generation (client disconnect → `GeneratorExit` →
`abort_request()`), `_OffsetAdjustedRoPE` stays installed permanently on all full-attention
layers. Every subsequent request then decodes with offset-shifted positions, producing garbage —
tool calls fail, file edits produce empty or wrong results. One abort contaminates the whole
session, explaining the "works for 3-4 messages then breaks permanently" pattern.

### Fix 4 — #548: Add `_cleanup_specprefill` to abort path

**Target:** `omlx/scheduler.py` — two locations:

**Location 1:** End of `_do_abort_request()` (~line 2630), after `_cleanup_output_parser_session`:
```python
self._cleanup_specprefill(request_id)
```

**Location 2:** `_PrefillAbortedError` handler (~line 3625), after the reschedule call, before
the next `step()` iteration:
```python
for rid in [r.request_id for r in rescheduled]:
    self._cleanup_specprefill(rid)
```

`_cleanup_specprefill` already guards with `if self._specprefill_active_request_id == request_id`
— it is a no-op for non-SpecPrefill requests and safe to call redundantly.

**Verification:**
- Enable SpecPrefill on Qwen3.5, run a Claude Code session, trigger a client disconnect
  mid-generation (short timeout or Ctrl+C), then send a new message.
- Before fix: subsequent messages produce garbled output / empty tool calls.
- After fix: model responds correctly.
- Unit test: mock `_specprefill_active_request_id = "req-1"`, call `_do_abort_request("req-1")`,
  assert `cleanup_rope` was invoked and `_specprefill_active_request_id` is `None`.

### Fix 3 (Secondary — reproducible): Make `input_schema` optional in `AnthropicTool`

**Target:** `omlx/api/anthropic_models.py:114–120`

```python
class AnthropicTool(BaseModel):
    type: str | None = None                      # add: for built-in tool types
    name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None   # change: optional for built-in tools
    cache_control: dict[str, str] | None = None
```

---

## Critical Files

| File | Lines | Purpose |
|------|-------|---------|
| `omlx/models/vlm.py` | ~163, ~246–285 | `_uses_mrope` detection, decode path guard — already fixed in `512c21b` |
| `omlx/server.py` | 1262–1272 | `_with_sse_keepalive` disconnect poll — Fix 2 target |
| `omlx/server.py` | 1315–1325 | `_run_with_disconnect_guard` — Fix 2 target |
| `omlx/server.py` | 1357–1367 | `_with_json_keepalive` — Fix 2 target |
| `omlx/api/anthropic_models.py` | 114–120 | Fix 3 target |

---

## Verification

### Fix 1 (mRoPE decode corruption)
- Run a Qwen3.5 or Qwen3-VL model on `0.3.5.dev2` vs `b8a9c5a`: on dev2 output is garbage
  after a few tokens; on b8a9c5a output is coherent
- Automated: the integration tests added in `512c21b`
  (`tests/integration/test_vlm_mrope_integration.py`) verify correct mRoPE decode

### Fix 2 (disconnect false positive — non-deterministic)
- **Unit test**: mock `is_disconnected()` to return `[True, False, True, True, True]`;
  assert task is cancelled only after the 3rd consecutive `True` (index 2, 3, 4),
  not after index 0
- **Observability**: debug log "waiting for confirmation (N/3)" makes false positives
  visible without needing to reproduce the abort

### Fix 3 (422 — fully reproducible)
```bash
curl -X POST http://localhost:PORT/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"...","messages":[{"role":"user","content":"hi"}],
       "tools":[{"type":"web_search_20250305","name":"web_search","max_uses":8}]}'
# Expect: 200 (not 422)
```
