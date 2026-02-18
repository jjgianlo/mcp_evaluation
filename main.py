"""main.py

Orchestration entrypoint for the MCP evaluation harness.

Wires together all subsystems:
  config → xml_parser → mcp_setup → agent → evaluator → reporting

"""

from __future__ import annotations

import asyncio
import csv
from datetime import datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from config import CONFIG_REQUIRED_VARS, logger
from agent import agent_loop
from evaluator import evaluate_task_success
from mcp_setup import get_mcp_tools, setup_mcp_client
from reporting import (
    compute_task_level_summary,
    flatten_results_for_csv,
    render_markdown_summary,
)
from utils import _safe_json_dumps, validate_environment
from xml_parser import parse_evaluation

load_dotenv(find_dotenv())


async def run_evaluation(
    evaluation_xml: Path,
    output_json: Path,
    num_runs: int = 10,
) -> None:
    import os

    validate_environment(CONFIG_REQUIRED_VARS)

    criteria = parse_evaluation(evaluation_xml)

    llm_client = OpenAI(base_url=os.getenv("LLM_BASE_URL"))
    mcp_client = await setup_mcp_client()
    async with mcp_client:
        tools, schemas = await get_mcp_tools(mcp_client)

        all_runs = []
        for run_id in range(1, num_runs + 1):
            logger.info("=== RUN %d/%d ===", run_id, num_runs)
            run_results = []
            for t in criteria:
                logger.info("Task %d: %s", t.task_id, t.task_name)
                res = await agent_loop(
                    mcp_client=mcp_client,
                    llm_client=llm_client,
                    tools=tools,
                    tool_schemas=schemas,
                    prompt=t.prompt,
                    task_id=t.task_id,
                    run_id=run_id,
                )
                assess = evaluate_task_success(criteria=t, result=res)
                res.success_assessment = assess
                res.score = int(assess.overall_success)
                run_results.append(res)
            all_runs.append(run_results)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "num_runs": num_runs,
        "llm_model": os.getenv("LLM_MODEL"),
        "evaluation_xml": str(evaluation_xml),
        "runs": [[r.to_dict() for r in run] for run in all_runs],
    }
    output_json.write_text(_safe_json_dumps(payload), encoding="utf-8")
    logger.info("Wrote audit JSON to %s", output_json)

    # Convenience artifacts for thesis write-up
    csv_rows = flatten_results_for_csv(all_runs, criteria)
    csv_path = output_json.with_suffix("").with_name(output_json.stem + "_flattened.csv")
    if csv_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        logger.info("Wrote flattened CSV to %s", csv_path)

    task_summary = compute_task_level_summary(all_runs)
    md_report = render_markdown_summary(
        criteria=criteria,
        task_summary=task_summary,
        all_runs=all_runs,
        evaluation_xml=evaluation_xml,
    )
    md_path = output_json.with_suffix("").with_name(output_json.stem + "_report.md")
    md_path.write_text(md_report, encoding="utf-8")
    logger.info("Wrote markdown report to %s", md_path)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run MCP evaluation harness (v3)")
    p.add_argument("--eval", type=str, default="evaluation.xml", help="Path to evaluation XML")
    p.add_argument("--out", type=str, default="results/raw_audit_data.json", help="Output audit JSON")
    p.add_argument("--runs", type=int, default=10, help="Number of runs per task")
    args = p.parse_args()

    asyncio.run(run_evaluation(Path(args.eval), Path(args.out), num_runs=int(args.runs)))


if __name__ == "__main__":
    main()
