"""xml_parser.py

Parses the evaluation.xml into a list of TaskCriteria objects.

All task specifications (expected tools, forbidden tools, assertions, distinct
checks, recovery flags, and negative-test settings) are read exclusively from
the XML file, making the harness fully reproducible without touching source code.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

from models import (
    Assertion,
    DistinctCheck,
    QuantileMonotonicCheck,
    TaskCriteria,
    ToolRequirement,
)


def parse_evaluation(xml_path: Path) -> list[TaskCriteria]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    tasks: list[TaskCriteria] = []
    for task in root.findall(".//task"):
        tid = int((task.findtext("task_id") or "0").strip())
        name = (task.findtext("task_name") or f"Task {tid}").strip()
        prompt = (task.findtext("prompt") or "").strip()
        # Narrative/keyword checks intentionally omitted (objective-only evaluation).

        # expected tools
        expected_tools: list[ToolRequirement] = []
        exp_tools_elem = task.find("expected_tools")
        if exp_tools_elem is not None:
            for t in exp_tools_elem.findall("tool"):
                expected_tools.append(
                    ToolRequirement(
                        name=t.attrib.get("name", "").strip(),
                        required=(t.attrib.get("required", "true").lower() == "true"),
                        min_calls=int(t.attrib.get("min_calls", "1")),
                    )
                )

        forbidden_tools_text = (task.findtext("forbidden_tools") or "").strip()
        forbidden_tools = [x.strip() for x in forbidden_tools_text.split(",") if x.strip()]

        # assertions
        def _parse_assertions(parent_tag: str, target: str) -> list[Assertion]:
            out: list[Assertion] = []
            parent = task.find(parent_tag)
            if parent is None:
                return out
            for a in parent.findall("assert"):
                tool = a.attrib.get("tool", "").strip()
                path = a.attrib.get("path", "").strip()
                equals = a.attrib.get("equals")
                contains = a.attrib.get("contains")
                length = a.attrib.get("length")
                occurrence: Literal["first", "last"] = (a.attrib.get("occurrence") or "last").strip().lower()

                out.append(
                    Assertion(
                        target=target,
                        tool=tool,
                        path=path,
                        equals=equals,
                        contains=contains,
                        length=int(length) if length else None,
                        occurrence=occurrence,
                    )
                )

            return out

        arg_assertions = _parse_assertions("argument_assertions", target="arguments")
        out_assertions = _parse_assertions("output_assertions", target="result")

        qm_checks: list[QuantileMonotonicCheck] = []
        qm_parent = task.find("quantile_monotonic_checks")
        if qm_parent is not None:
            for q in qm_parent.findall("check"):
                qm_checks.append(
                    QuantileMonotonicCheck(
                        tool=q.attrib.get("tool", "").strip(),
                        lower_path=q.attrib.get("lower_path", "").strip(),
                        median_path=q.attrib.get("median_path", "").strip(),
                        upper_path=q.attrib.get("upper_path", "").strip(),
                    )
                )

        should_fail = (task.findtext("should_fail_gracefully") or "false").strip().lower() == "true"
        exp_err_types_text = (task.findtext("expected_error_types") or "").strip()
        exp_err_types = [x.strip() for x in exp_err_types_text.split(",") if x.strip()]

        # recovery tasks
        requires_recovery = (task.findtext("requires_recovery") or "false").strip().lower() == "true"

        # distinct checks (argument/result diversity; useful for uncertainty probing tasks)
        distinct_checks: list[DistinctCheck] = []
        dc_parent = task.find("distinct_checks")
        if dc_parent is not None:
            for d in dc_parent.findall("distinct"):
                distinct_checks.append(
                    DistinctCheck(
                        tool=(d.attrib.get("tool", "") or "").strip(),
                        target=(d.attrib.get("target", "arguments") or "arguments").strip().lower(),
                        path=(d.attrib.get("path", "") or "").strip(),
                        min_distinct=int(d.attrib.get("min_distinct", "1")),
                    )
                )

        tasks.append(
            TaskCriteria(
                task_id=tid,
                task_name=name,
                prompt=prompt,
                expected_tools=expected_tools,
                forbidden_tools=forbidden_tools,
                argument_assertions=arg_assertions,
                output_assertions=out_assertions,
                quantile_monotonic_checks=qm_checks,
                distinct_checks=distinct_checks,
                requires_recovery=requires_recovery,
                should_fail_gracefully=should_fail,
                expected_error_types=exp_err_types,
            )
        )

    if not tasks:
        raise ValueError(f"No <task> entries found in {xml_path}")
    return tasks
