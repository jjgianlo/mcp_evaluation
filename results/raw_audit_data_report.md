# MCP Evaluation Report
Generated: 2026-01-21 16:58:41
Evaluation file: evaluation.xml

## Executive Summary
- Total task executions: 50
- Overall success rate: 50/50 (100.0%)
- Total tokens: 2,291,680 (avg. total: 45,834; avg. completion: 1,426; avg. prompt: 44,407)
- Total latency: 3457.02s (avg. 69.14s per task; std. 27.23)
- LLM reasoning latency: 2817.56s (avg. 56.35s per task)
- Tool execution latency: 639.42s (avg. 12.79s per task)
- Additional MCP latency (tool vs backend execution time): (avg. 1.32s per task)
- Total tool calls: 213 (min: 2; max: 15; avg: 4.26)
- Average recovery time for task 5: 29.56

## Task-Level Summary
| Task | Name | Overall Success | Mean Latency (s) | StdDev | Mean Tokens | Mean Tools | Tool Share |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Zero-Shot Inference with Metadata Lookup | 100% | 66.69 | 19.00 | 41,004 | 2.2 | 4.46% |
| 2 | Fine-Tuning Job Submission | 100% | 65.59 | 17.64 | 42,580 | 2.3 | 34.32% |
| 3 | Model Comparison (Fine-tuned vs. Pre-trained) | 100% | 103.93 | 28.35 | 58,058 | 5.1 | 12.44% |
| 4 | Error Handling: Missing Dataset (Negative Test with Filename Uncertainty) | 100% | 41.67 | 9.56 | 45,632 | 9.5 | 16.96% |
| 5 | Semantic Error Recovery: Column Mismatch During Training Submission | 100% | 67.82 | 16.23 | 41,895 | 2.2 | 31.29% |


## Model Token Cost Comparison

| Model         | Input Price   | Output Price   | Total Cost   | Avg. Cost/Task   |
|:--------------|:--------------|:---------------|:-------------|:-----------------|
| GLM-4.7       | $0.60/MTok    | $2.20/MTok     | $1.49        | $0.03            |
| GPT-5.2       | $1.75/MTok    | $14.00/MTok    | $4.88        | $0.10            |
| Claude Opus 4 | $15.00/MTok   | $75.00/MTok    | $38.65       | $0.77            |

---
## Methodological Notes
- Objective success is computed from tool evidence (required/forbidden tools, schema validation, explicit assertions, distinctness, and recovery where applicable).
- For negative tests, success requires observing an expected backend error type, not the absence of Python exceptions.
