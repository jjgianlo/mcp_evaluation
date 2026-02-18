"""agent.py

Async agent loop for the MCP evaluation harness.

Drives the LLM ↔ MCP tool interaction cycle for a single task run:
1. Sends the task prompt to the LLM.
2. Executes any requested tool calls via the MCP client.
3. Feeds tool results back to the LLM.
4. Captures timing, token usage, tool invocations, and error events.
5. Returns a fully-populated TaskResult (without success scoring).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from fastmcp import Client
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from config import EVALUATION_SYSTEM_PROMPT
from models import (
    ErrorEvent,
    ErrorType,
    TaskResult,
    TimingBreakdown,
    ToolInvocation,
)

from mcp_setup import validate_tool_arguments
from utils import _classify_backend_error, _extract_xml_tag, _safe_json_dumps


async def agent_loop(
    *,
    mcp_client: Client,
    llm_client: OpenAI,
    tools: list[ChatCompletionFunctionToolParam],
    tool_schemas: dict[str, dict],
    prompt: str,
    task_id: int,
    run_id: int,
    max_iterations: int = 12,
) -> TaskResult:

    messages: list[Any] = [
        ChatCompletionSystemMessageParam(role="system", content=EVALUATION_SYSTEM_PROMPT),
        ChatCompletionUserMessageParam(role="user", content=prompt),
    ]

    start_total = time.perf_counter()
    llm_total = 0.0
    tool_total = 0.0

    total_prompt_tokens = 0
    total_completion_tokens = 0
    tool_invocations: list[ToolInvocation] = []
    error_events: list[ErrorEvent] = []

    def _elapsed() -> float:
        return time.perf_counter() - start_total

    def _finalize(full_text: str) -> TaskResult:
        total_latency = time.perf_counter() - start_total
        timing = TimingBreakdown(
            total_latency=total_latency,
            llm_reasoning_latency=llm_total,
            tool_execution_latency=tool_total,
        )
        return TaskResult(
            task_id=task_id,
            run_id=run_id,
            prompt_snippet=prompt[:60] + ("..." if len(prompt) > 60 else ""),
            full_response_text=full_text,
            actual_response=_extract_xml_tag(full_text, "response"),
            score=0,
            total_duration=total_latency,
            timing_breakdown=timing,
            tool_invocations=tool_invocations,
            error_events=error_events,
            token_usage={
                "prompt": total_prompt_tokens,
                "completion": total_completion_tokens,
                "total": total_prompt_tokens + total_completion_tokens,
            },
        )

    def _mark_recovered(tool_name: str, recovered_iteration: int) -> None:
        for ev in reversed(error_events):
            if ev.tool_name == tool_name and not ev.was_recovered:
                ev.was_recovered = True
                ev.recovered_iteration = recovered_iteration
                ev.recovered_elapsed_s = _elapsed()
            else:
                return

    for iteration in range(1, max_iterations + 1):
        try:
            t0 = time.perf_counter()
            resp = llm_client.chat.completions.create(
                model=os.getenv("LLM_MODEL"),
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            llm_total += time.perf_counter() - t0

            if resp.usage:
                total_prompt_tokens += resp.usage.prompt_tokens
                total_completion_tokens += resp.usage.completion_tokens

            msg = resp.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                return _finalize(msg.content or "")

            for tc in msg.tool_calls:
                tool_name = tc.function.name

                # ---- Parse tool arguments
                parse_failed = False
                try:
                    tool_args = json.loads(tc.function.arguments or "{}")
                    if not isinstance(tool_args, dict):
                        raise ValueError("Tool arguments must be a JSON object.")
                except Exception:
                    parse_failed = True
                    tool_args = {}

                # ---- Schema validation (record only; do not emit ErrorEvent)
                schema = tool_schemas.get(tool_name, {}) or {}
                schema_valid, schema_errors = validate_tool_arguments(schema, tool_args)

                # ---- Decide tool_result (avoid calling backend if parse failed)
                exec_time: float = 0.0
                tool_result: Any = None
                raised_exception = False

                if parse_failed:
                    # Synthetic tool response: faithful and side-effect free
                    tool_result = {
                        "success": False,
                        "error": {"type": "InvalidToolArguments", "message": "Tool arguments JSON could not be parsed."},
                    }
                    # Also record as an ErrorEvent, but mark it clearly as harness-side
                    error_events.append(
                        ErrorEvent(
                            iteration=iteration,
                            tool_name=tool_name,
                            error_type=ErrorType.INVALID_PARAMETER,
                            error_message="Tool arguments parse error (invalid JSON).",
                            backend_error_type="harness_invalid_tool_arguments",
                            was_recovered=False,
                            error_elapsed_s=_elapsed(),
                        )
                    )
                else:
                    # ---- Execute tool with guaranteed timing (even if it throws)
                    tool_start = time.perf_counter()
                    try:
                        result = await mcp_client.call_tool(tool_name, tool_args)
                        tool_result = result.data
                    except Exception as e:
                        raised_exception = True
                        tool_result = {"success": False, "error": {"type": type(e).__name__, "message": str(e)}}

                        classified = _classify_backend_error(type(e).__name__, str(e))
                        error_events.append(
                            ErrorEvent(
                                iteration=iteration,
                                tool_name=tool_name,
                                error_type=classified,
                                error_message=str(e),
                                backend_error_type=classified.value,
                                was_recovered=False,
                                error_elapsed_s=_elapsed(),
                            )
                        )
                    finally:
                        exec_time = time.perf_counter() - tool_start
                        tool_total += exec_time

                # ---- Semantic tool error capture (ONLY if not already recorded via exception)
                if (not raised_exception) and isinstance(tool_result, dict) and tool_result.get("success") is False:
                    err = tool_result.get("error") or {}
                    backend_type = (err.get("type") or "").strip() or None
                    backend_msg = err.get("message") or ""

                    classified = _classify_backend_error(backend_type, backend_msg)
                    error_events.append(
                        ErrorEvent(
                            iteration=iteration,
                            tool_name=tool_name,
                            error_type=classified,
                            error_message=backend_msg,
                            backend_error_type=classified.value,
                            was_recovered=False,
                            error_elapsed_s=_elapsed(),
                        )
                    )

                # ---- Record invocation
                tool_invocations.append(
                    ToolInvocation(
                        name=tool_name,
                        arguments=tool_args,
                        argument_schema_valid=(schema_valid and (not parse_failed)),
                        argument_schema_errors=schema_errors,
                        result_data=tool_result,
                        result_snippet=str(tool_result)[:240],
                        execution_time=float(exec_time),
                    )
                )

                # ---- Recovery: mark most recent unrecovered error on success:true
                if isinstance(tool_result, dict) and tool_result.get("success") is True:
                    _mark_recovered(tool_name, recovered_iteration=iteration)

                # ---- Feed tool output back to the model
                messages.append(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        tool_call_id=tc.id,
                        content=_safe_json_dumps(tool_result),
                    )
                )

        except Exception as e:
            error_events.append(
                ErrorEvent(
                    iteration=iteration,
                    tool_name=None,
                    error_type=ErrorType.UNKNOWN,
                    error_message=f"LLM call failed: {e}",
                    backend_error_type=ErrorType.UNKNOWN.value,
                    was_recovered=False,
                    error_elapsed_s=_elapsed(),
                )
            )
            break

    return _finalize("Max iterations reached")
