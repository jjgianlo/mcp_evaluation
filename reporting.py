"""reporting.py

Reporting and aggregation helpers for the MCP evaluation harness.

Provides three public functions:

- ``flatten_results_for_csv``      - converts all run results into a flat list of dicts for CSV export.
- ``compute_task_level_summary``   - aggregates per-task statistics across all runs.
- ``render_markdown_summary``      - produces a human-readable Markdown report for thesis write-ups.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from evaluator import _evaluate_assertions
from models import Assertion, TaskCriteria, TaskResult, ToolInvocation
from utils import _ci95_mean, _mean, _stdev


# =========================
# Internal helpers
# =========================


def _counts_by_tool(invocations: list[ToolInvocation]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for inv in invocations:
        counts[inv.name] = counts.get(inv.name, 0) + 1
    return counts


def _tool_recall_precision(
    c: TaskCriteria,
    invocations: list[ToolInvocation],
) -> tuple[float, float, int, int]:
    counts = _counts_by_tool(invocations)
    total_calls = len(invocations)

    required = [t for t in c.expected_tools if t.required]
    if not required:
        # No required tools: define recall=1; precision=1 if no calls, else 0 (or 1, depending on your convention)
        return 1.0, (1.0 if total_calls == 0 else 0.0), total_calls, 0

    satisfied = sum(1 for t in required if counts.get(t.name, 0) >= t.min_calls)
    recall = satisfied / len(required)

    required_calls = sum(counts.get(t.name, 0) for t in required)
    precision = (required_calls / total_calls) if total_calls > 0 else 0.0
    overtool_calls = max(0, total_calls - required_calls)

    return recall, precision, total_calls, overtool_calls


def _schema_valid_rate_for_required(c: TaskCriteria, invocations: list[ToolInvocation]) -> float:
    required_names = {t.name for t in c.expected_tools if t.required}
    req_invs = [i for i in invocations if i.name in required_names]
    if not req_invs:
        return 0.0 if required_names else 1.0
    return sum(1 for i in req_invs if i.argument_schema_valid) / len(req_invs)


def _assertion_pass_rate(
    invocations: list[ToolInvocation],
    assertions: list[Assertion],
) -> tuple[int, int, float]:
    # Reuse your existing evaluator for correctness, but derive a rate by counting failures.
    if not assertions:
        return 0, 0, 1.0
    ok, failures = _evaluate_assertions(invocations, assertions)
    total = len(assertions)
    failed = len(failures)
    passed = max(0, total - failed)
    rate = passed / total if total else 1.0
    return passed, total, rate


# =========================
# Public API
# =========================


def flatten_results_for_csv(
    all_runs: list[list[TaskResult]],
    criteria: list[TaskCriteria],
) -> list[dict[str, Any]]:
    """Flatten run-level results + criteria-derived metrics for external analysis."""
    crit_by_task = {c.task_id: c for c in criteria}

    rows: list[dict[str, Any]] = []
    for run in all_runs:
        for r in run:
            a = r.success_assessment
            c = crit_by_task.get(r.task_id)

            # Base fields (your current set)
            row: dict[str, Any] = {
                "run_id": r.run_id,
                "task_id": r.task_id,
                "overall_success": int(a.overall_success) if a else 0,
                "objective_success": int(a.objective_success) if a else 0,
                "recovery_pass": int(a.recovery_pass) if a else 0,
                "distinct_checks_pass": int(a.distinct_checks_pass) if a else 0,
                "confidence": round(a.confidence, 4) if a else 0.0,
                "verification_required": int(a.verification_required) if a else 1,
                "total_latency_s": round(r.timing_breakdown.total_latency, 4),
                "llm_latency_s": round(r.timing_breakdown.llm_reasoning_latency, 4),
                "tool_latency_s": round(r.timing_breakdown.tool_execution_latency, 4),
                "tool_latency_share": round(
                    (r.timing_breakdown.tool_execution_latency / r.timing_breakdown.total_latency)
                    if r.timing_breakdown.total_latency > 0 else 0.0,
                    4,
                ),
                "tokens_prompt": r.token_usage.get("prompt", 0),
                "tokens_completion": r.token_usage.get("completion", 0),
                "tokens_total": r.token_usage.get("total", 0),
                "tokens_per_second": round(
                    (r.token_usage.get("total", 0) / r.timing_breakdown.total_latency)
                    if r.timing_breakdown.total_latency > 0 else 0.0,
                    4,
                ),
                "required_tools_missing": ";".join(a.required_tools_missing) if a else "",
                "forbidden_tools_used": ";".join(a.forbidden_tools_used) if a else "",
                "schema_validation_pass": int(a.schema_validation_pass) if a else 0,
                "assertion_pass": int(a.assertion_pass) if a else 0,
                "expected_error_observed": int(a.expected_error_observed) if a else 0,
                "observed_backend_error_types": ";".join(a.observed_backend_error_types) if a else "",
            }

            # Criteria-joined fields + tool-level metrics
            if c is not None:
                recall, precision, total_calls, overtool_calls = _tool_recall_precision(c, r.tool_invocations)
                row.update(
                    {
                        "task_name": c.task_name,
                        "should_fail_gracefully": int(c.should_fail_gracefully),
                        "expected_error_types": ";".join(c.expected_error_types),
                        "expected_required_tools": ";".join(
                            f"{t.name}:{t.min_calls}" for t in c.expected_tools if t.required
                        ),
                        "expected_forbidden_tools": ";".join(c.forbidden_tools),

                        "tool_calls_total": total_calls,
                        "tool_overtool_calls": overtool_calls,
                        "tool_recall": round(recall, 4),
                        "tool_precision": round(precision, 4),
                        "schema_valid_rate_required_calls": round(
                            _schema_valid_rate_for_required(c, r.tool_invocations), 4
                        ),
                    }
                )

                # Assertion rates (criteria-driven, stable, publishable)
                arg_passed, arg_total, arg_rate = _assertion_pass_rate(r.tool_invocations, c.argument_assertions)
                out_passed, out_total, out_rate = _assertion_pass_rate(r.tool_invocations, c.output_assertions)
                row.update(
                    {
                        "arg_assertions_total": arg_total,
                        "arg_assertions_passed": arg_passed,
                        "arg_assertion_rate": round(arg_rate, 4),
                        "out_assertions_total": out_total,
                        "out_assertions_passed": out_passed,
                        "out_assertion_rate": round(out_rate, 4),
                    }
                )

            # Recovery metrics (available now; richer if you add recovered_iteration/elapsed fields)
            err_total = len(r.error_events)
            err_recovered = sum(1 for e in r.error_events if e.was_recovered)
            row.update(
                {
                    "error_events_total": err_total,
                    "error_events_recovered": err_recovered,
                    "error_recovery_rate": round((err_recovered / err_total) if err_total else 1.0, 4),
                }
            )

            # Optional: if you add recovered_iteration / elapsed fields
            recovered_lags = [
                (e.recovered_iteration - e.iteration)
                for e in r.error_events
                if e.was_recovered and e.recovered_iteration is not None
            ]
            if recovered_lags:
                row["recovery_iterations_mean"] = round(sum(recovered_lags) / len(recovered_lags), 4)
            else:
                row["recovery_iterations_mean"] = ""

            recovered_times = [
                (e.recovered_elapsed_s - e.error_elapsed_s)
                for e in r.error_events
                if (
                    e.was_recovered
                    and e.recovered_elapsed_s is not None
                    and e.error_elapsed_s is not None
                )
            ]
            if recovered_times:
                row["recovery_time_s_mean"] = round(sum(recovered_times) / len(recovered_times), 4)
            else:
                row["recovery_time_s_mean"] = ""

            rows.append(row)

    return rows


def compute_task_level_summary(all_runs: list[list[TaskResult]]) -> list[dict[str, Any]]:
    by_task: dict[int, list[TaskResult]] = {}
    for run in all_runs:
        for r in run:
            by_task.setdefault(r.task_id, []).append(r)

    summary: list[dict[str, Any]] = []
    for tid, rs in sorted(by_task.items(), key=lambda kv: kv[0]):
        lat = [r.timing_breakdown.total_latency for r in rs]
        tok = [r.token_usage.get("total", 0) for r in rs]
        obj = [1 if (r.success_assessment and r.success_assessment.objective_success) else 0 for r in rs]
        overall = [1 if (r.success_assessment and r.success_assessment.overall_success) else 0 for r in rs]
        conf = [r.success_assessment.confidence for r in rs if r.success_assessment]
        ci_lo, ci_hi = _ci95_mean(lat)

        summary.append(
            {
                "task_id": tid,
                "n": len(rs),
                "objective_success_rate": _mean(obj),
                "overall_success_rate": _mean(overall),
                "latency_mean_s": _mean(lat),
                "latency_stdev_s": _stdev(lat),
                "latency_ci95_low_s": ci_lo,
                "latency_ci95_high_s": ci_hi,
                "tokens_mean": _mean([float(x) for x in tok]),
                "tokens_max": max(tok) if tok else 0,
                "avg_confidence": _mean(conf),
                "verification_required_count": sum(
                    1
                    for r in rs
                    if r.success_assessment and r.success_assessment.verification_required
                ),
            }
        )
    return summary


def render_markdown_summary(
    *,
    criteria: list[TaskCriteria],
    task_summary: list[dict[str, Any]],
    all_runs: list[list[TaskResult]],
    evaluation_xml: Path,
) -> str:
    total = sum(len(r) for r in all_runs)
    overall_successes = sum(
        1
        for run in all_runs
        for r in run
        if r.success_assessment and r.success_assessment.overall_success
    )
    objective_successes = sum(
        1
        for run in all_runs
        for r in run
        if r.success_assessment and r.success_assessment.objective_success
    )
    tokens_total = sum(r.token_usage.get("total", 0) for run in all_runs for r in run)

    out = []
    out.append(f"# MCP Evaluation Report\n")
    out.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    out.append(f"Evaluation file: {evaluation_xml}\n")
    out.append("\n## Executive Summary\n")
    out.append(f"- Total task executions: {total}\n")
    out.append(f"- Objective success rate: {objective_successes}/{total} ({(objective_successes/total*100 if total else 0):.1f}%)\n")
    out.append(f"- Overall success rate: {overall_successes}/{total} ({(overall_successes/total*100 if total else 0):.1f}%)\n")
    out.append(f"- Total tokens: {tokens_total:,}\n")

    out.append("\n## Task-Level Summary\n")
    out.append("| Task | Name | Obj. Success | Overall Success | Mean Latency (s) | StdDev | 95% CI | Mean Tokens | Review Flags |\n")
    out.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")

    name_by_id = {t.task_id: t.task_name for t in criteria}
    for s in task_summary:
        tid = s["task_id"]
        ci = f"[{s['latency_ci95_low_s']:.2f}, {s['latency_ci95_high_s']:.2f}]"
        out.append(
            f"| {tid} | {name_by_id.get(tid, '')} | {s['objective_success_rate']*100:.0f}% | {s['overall_success_rate']*100:.0f}% | "
            f"{s['latency_mean_s']:.2f} | {s['latency_stdev_s']:.2f} | {ci} | {s['tokens_mean']:.0f} | {s['verification_required_count']} |\n"
        )

    out.append("\n## Methodological Notes\n")
    out.append(
        "- Objective success is computed from tool evidence (required/forbidden tools, schema validation, explicit assertions, distinctness, and recovery where applicable).\n"
        "- For negative tests, success requires observing an expected backend error type, not the absence of Python exceptions.\n"
    )
    return "".join(out)
