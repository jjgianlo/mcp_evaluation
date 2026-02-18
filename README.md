# MCP Evaluation Harness

Thesis-grade evaluation harness for MCP (Model Context Protocol) servers, focusing on time-series forecasting tools.

## Repository Structure

```
mcp_evaluation/
├── results/
│   ├── raw_audit_data.json          # Full audit trail for all runs
│   ├── raw_audit_data_flattened.csv # One row per task run, ready for analysis
│   └── raw_audit_data_report.md     # Markdown summary table for thesis write-up
├── __init__.py       # Package declaration
├── config.py         # Constants, logging, system prompt
├── models.py         # Dataclasses and enums
├── utils.py          # Shared helpers
├── xml_parser.py     # evaluation_v2.xml → TaskCriteria
├── mcp_setup.py      # Azure AD auth + MCP client setup
├── agent.py          # Async LLM ↔ MCP agent loop
├── evaluator.py      # Assertion checks + success scoring
├── reporting.py      # CSV, task summary, Markdown report
├── main.py           # Orchestration entrypoint (CLI)
├── evaluation.xml    # Task specifications (place here)
├── requirements.txt
└── README.md
```

## Setup

```bash
uv sync
```

Create a `.env` file with the following variables:

```
TENANT_ID=...
CLIENT_ID=...
CLIENT_APP_SECRET=...
BACKEND_APP_ID=...
REMOTE_URL=...
LLM_BASE_URL=...
LLM_MODEL=...
API_KEY=...
```

## Usage

```bash
python -m main --eval evaluation.xml --out raw_audit_data.json [--runs 10]
```

## Outputs:
- `raw_audit_data.json` — full audit trail for all runs
- `raw_audit_data_flattened.csv` — one row per task run, ready for analysis
- `raw_audit_data_report.md` — Markdown summary table for thesis write-up

## Module Overview

| Module | Responsibility |
|---|---|
| `config.py` | `CONFIG_REQUIRED_VARS`, logging, `EVALUATION_SYSTEM_PROMPT` |
| `models.py` | All dataclasses: `TaskCriteria`, `TaskResult`, `SuccessAssessment`, `ErrorEvent`, … |
| `utils.py` | `_get_by_path`, `_classify_backend_error`, `_mean`, `_ci95_mean`, … |
| `xml_parser.py` | `parse_evaluation_v2(xml_path)` → `list[TaskCriteria]` |
| `mcp_setup.py` | `setup_mcp_client()`, `get_mcp_tools()`, `validate_tool_arguments()` |
| `agent.py` | `agent_loop(...)` → `TaskResult` |
| `evaluator.py` | `evaluate_task_success(...)` → `SuccessAssessment` |
| `reporting.py` | `flatten_results_for_csv`, `compute_task_level_summary`, `render_markdown_summary` |
| `main.py` | `run_evaluation(...)`, `main()` CLI entrypoint |
