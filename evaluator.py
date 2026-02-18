"""evaluator.py

Objective success evaluation for the MCP evaluation harness.

Contains three families of checks that are composed into a single SuccessAssessment:

1. Assertion evaluation  - validates JSON-path conditions on tool arguments and results.
2. Distinct checks       - validates diversity across repeated invocations of the same tool.
3. Quantile monotonicity - validates that lower ≤ median ≤ upper across forecast arrays.

The top-level function ``evaluate_task_success`` applies all checks and produces a
SuccessAssessment with a confidence score and a human-readable reasoning string.
"""

from __future__ import annotations

import json
from typing import Any

from models import (
    Assertion,
    DistinctCheck,
    QuantileMonotonicCheck,
    SuccessAssessment,
    TaskCriteria,
    TaskResult,
    ToolInvocation,
)
from utils import _get_by_path


# =========================
# Assertion evaluation
# =========================


def _evaluate_assertions(
    invocations: list[ToolInvocation],
    assertions: list[Assertion],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    by_tool: dict[str, list[ToolInvocation]] = {}
    for inv in invocations:
        by_tool.setdefault(inv.name, []).append(inv)

    for a in assertions:
        invs = by_tool.get(a.tool, [])
        if not invs:
            failures.append(f"Missing tool for assertion: {a.tool}.{a.path}")
            continue
        # Select which invocation to validate when a tool is called multiple times.
        occ = (a.occurrence or "last").strip().lower()
        if occ == "first":
            inv = invs[0]
        elif occ == "last":
            inv = invs[-1]
        else:
            # 1-based numeric index
            try:
                k = int(occ)
                if k < 1 or k > len(invs):
                    failures.append(f"Assertion occurrence out of range: {a.tool} occurrence={a.occurrence} calls={len(invs)}")
                    continue
                inv = invs[k - 1]
            except Exception:
                failures.append(f"Invalid assertion occurrence: {a.tool} occurrence={a.occurrence}")
                continue
        src = inv.arguments if a.target == "arguments" else inv.result_data
        val = _get_by_path(src, a.path)

        if a.length is not None:
            if not isinstance(val, (list, str, dict)):
                failures.append(f"Assertion length failed: {a.tool}.{a.path} not sized")
            else:
                if len(val) != a.length:
                    failures.append(
                        f"Assertion length failed: {a.tool}.{a.path} len={len(val)} expected={a.length}"
                    )
        if a.equals is not None:
            # string compare, but allow numeric
            exp = a.equals
            if isinstance(val, (int, float, bool)):
                exp_cast: Any
                if isinstance(val, bool):
                    exp_cast = exp.lower() == "true"
                elif isinstance(val, int):
                    exp_cast = int(float(exp))
                else:
                    exp_cast = float(exp)
                if val != exp_cast:
                    failures.append(f"Assertion equals failed: {a.tool}.{a.path}={val} expected={exp_cast}")
            else:
                if str(val) != exp:
                    failures.append(f"Assertion equals failed: {a.tool}.{a.path}='{val}' expected='{exp}'")
        if a.contains is not None:
            if val is None or a.contains.lower() not in str(val).lower():
                failures.append(f"Assertion contains failed: {a.tool}.{a.path} missing '{a.contains}'")

    return len(failures) == 0, failures


# =========================
# Distinct checks
# =========================


def _evaluate_distinct_checks(
    invocations: list[ToolInvocation],
    checks: list[DistinctCheck],
) -> tuple[bool, list[str]]:
    """Validate diversity across repeated tool invocations.

    Use-case: uncertainty probing tasks where the agent should try multiple candidate inputs (e.g., filename variants).
    """
    failures: list[str] = []
    if not checks:
        return True, failures

    by_tool: dict[str, list[ToolInvocation]] = {}
    for inv in invocations:
        by_tool.setdefault(inv.name, []).append(inv)

    for c in checks:
        invs = by_tool.get(c.tool, [])
        if not invs:
            failures.append(f"Missing tool for distinct check: {c.tool}.{c.path}")
            continue

        values = []
        for inv in invs:
            src = inv.arguments if c.target == "arguments" else inv.result_data
            v = _get_by_path(src, c.path)
            if v is None:
                continue
            # Normalize for hashing (JSON is fine for primitives/lists/dicts)
            try:
                v_norm = json.dumps(v, sort_keys=True, default=str)
            except Exception:
                v_norm = str(v)
            values.append(v_norm)

        distinct_n = len(set(values))
        if distinct_n < c.min_distinct:
            failures.append(
                f"Distinct check failed: {c.tool}.{c.path} distinct={distinct_n} required>={c.min_distinct}"
            )

    return len(failures) == 0, failures


# =========================
# Quantile monotonicity
# =========================


def _check_quantile_monotonic(
    invocations: list[ToolInvocation],
    checks: list[QuantileMonotonicCheck],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    by_tool: dict[str, list[ToolInvocation]] = {}
    for inv in invocations:
        by_tool.setdefault(inv.name, []).append(inv)

    for c in checks:
        invs = by_tool.get(c.tool, [])
        if not invs:
            failures.append(f"Missing tool for quantile check: {c.tool}")
            continue
        inv = invs[-1]
        res = inv.result_data
        lo = _get_by_path(res, c.lower_path)
        med = _get_by_path(res, c.median_path)
        hi = _get_by_path(res, c.upper_path)
        if not (isinstance(lo, list) and isinstance(med, list) and isinstance(hi, list)):
            failures.append(f"Quantile check failed: {c.tool} outputs not list arrays")
            continue
        n = min(len(lo), len(med), len(hi))
        if n == 0:
            failures.append(f"Quantile check failed: {c.tool} empty arrays")
            continue
        for i in range(n):
            try:
                if float(lo[i]) > float(med[i]) or float(med[i]) > float(hi[i]):
                    failures.append(f"Quantile monotonicity failed: {c.tool}[{i}] lo={lo[i]} med={med[i]} hi={hi[i]}")
                    break
            except Exception:
                failures.append(f"Quantile monotonicity failed: {c.tool}[{i}] non-numeric")
                break

    return len(failures) == 0, failures


# =========================
# Top-level scorer
# =========================


def evaluate_task_success(
    *,
    criteria: TaskCriteria,
    result: TaskResult,
) -> SuccessAssessment:
    # tool coverage
    counts: dict[str, int] = {}
    for inv in result.tool_invocations:
        counts[inv.name] = counts.get(inv.name, 0) + 1

    required_missing: list[str] = []
    for req in criteria.expected_tools:
        if req.required and counts.get(req.name, 0) < req.min_calls:
            required_missing.append(req.name)

    forbidden_used = [t for t in criteria.forbidden_tools if counts.get(t, 0) > 0]

    # schema validation: all REQUIRED tool invocations must have schema valid
    schema_pass = True
    for req in criteria.expected_tools:
        if not req.required:
            continue
        invs = [i for i in result.tool_invocations if i.name == req.name]
        if not invs:
            continue
        if not all(i.argument_schema_valid for i in invs):
            schema_pass = False
            break

    # assertions
    arg_pass, arg_fail = _evaluate_assertions(result.tool_invocations, criteria.argument_assertions)
    out_pass, out_fail = _evaluate_assertions(result.tool_invocations, criteria.output_assertions)
    q_pass, q_fail = _check_quantile_monotonic(result.tool_invocations, criteria.quantile_monotonic_checks)
    assertion_pass = arg_pass and out_pass and q_pass
    assertion_failures = arg_fail + out_fail + q_fail

    # additional validity checks: diversity and recovery
    distinct_pass, distinct_failures = _evaluate_distinct_checks(result.tool_invocations, criteria.distinct_checks)

    recovery_pass = True
    if criteria.requires_recovery:
        # Require at least one error that was later recovered by a successful tool invocation.
        recovery_pass = any(e.was_recovered for e in result.error_events)

    # expected errors (negative tests)
    observed_backend_types = [e.backend_error_type for e in result.error_events if e.backend_error_type]
    expected_error_observed = False
    if criteria.should_fail_gracefully:
        expected_set = {x.strip() for x in criteria.expected_error_types}
        expected_error_observed = any((t in expected_set) for t in observed_backend_types)

    # objective success definition
    if criteria.should_fail_gracefully:
        objective_success = (
            not required_missing
            and not forbidden_used
            and schema_pass
            and distinct_pass
            and recovery_pass
            and expected_error_observed
        )
    else:
        objective_success = (
            not required_missing
            and not forbidden_used
            and schema_pass
            and distinct_pass
            and recovery_pass
            and assertion_pass
        )

    overall_success = objective_success

    # confidence (0..1): weight objective signals heavily
    objective_score = 1.0
    if required_missing:
        objective_score *= 0.0
    if forbidden_used:
        objective_score *= 0.2
    if not schema_pass:
        objective_score *= 0.3
    if not distinct_pass:
        objective_score *= 0.4
    if criteria.requires_recovery and (not recovery_pass):
        objective_score *= 0.3
    if (not criteria.should_fail_gracefully) and (not assertion_pass):
        objective_score *= 0.3
    if criteria.should_fail_gracefully and (not expected_error_observed):
        objective_score *= 0.2

    confidence = max(0.0, min(1.0, objective_score))
    verification_required = (confidence < 0.85) or (not assertion_pass and not criteria.should_fail_gracefully)

    # reasoning
    parts = []
    parts.append(f"Tools missing: {required_missing}" if required_missing else "All required tools called")
    parts.append(f"Forbidden tools used: {forbidden_used}" if forbidden_used else "No forbidden tools")
    parts.append("Schema valid" if schema_pass else "Schema violations observed")
    if criteria.distinct_checks:
        parts.append("Distinct checks passed" if distinct_pass else f"Distinct checks failed: {distinct_failures[:2]}")
    if criteria.requires_recovery:
        parts.append("Recovery observed" if recovery_pass else "Recovery NOT observed")
    if criteria.should_fail_gracefully:
        parts.append(
            "Expected error observed" if expected_error_observed else f"Expected error NOT observed (observed={observed_backend_types})"
        )
    else:
        parts.append("Assertions passed" if assertion_pass else f"Assertions failed: {assertion_failures[:3]}")

    reasoning = "; ".join(parts)

    return SuccessAssessment(
        overall_success=overall_success,
        objective_success=objective_success,
        required_tools_missing=required_missing,
        forbidden_tools_used=forbidden_used,
        schema_validation_pass=schema_pass,
        assertion_pass=assertion_pass,
        assertion_failures=assertion_failures,
        distinct_pass=distinct_pass,
        distinct_failures=distinct_failures,
        recovery_pass=recovery_pass,
        expected_error_observed=expected_error_observed,
        observed_backend_error_types=observed_backend_types,
        confidence=confidence,
        verification_required=verification_required,
        reasoning=reasoning,
        distinct_checks_pass=distinct_pass,
    )
