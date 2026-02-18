# MCP Evaluation Report (v3)
Generated: 2026-01-21 16:58:41
Evaluation file: evaluation.xml

## Executive Summary
- Total task executions: 50
- Objective success rate: 50/50 (100.0%)
- Overall success rate: 50/50 (100.0%)
- Total tokens: 2,291,680

## Task-Level Summary
| Task | Name | Obj. Success | Overall Success | Mean Latency (s) | StdDev | 95% CI | Mean Tokens | Review Flags |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Zero-Shot Inference with Metadata Lookup | 100% | 100% | 66.69 | 19.00 | [54.92, 78.47] | 41004 | 0 |
| 2 | Fine-Tuning Job Submission | 100% | 100% | 65.59 | 17.64 | [54.66, 76.53] | 42580 | 0 |
| 3 | Model Comparison (Fine-tuned vs. Pre-trained) | 100% | 100% | 103.93 | 28.35 | [86.36, 121.50] | 58058 | 0 |
| 4 | Error Handling: Missing Dataset (Negative Test with Filename Uncertainty) | 100% | 100% | 41.67 | 9.56 | [35.75, 47.59] | 45632 | 0 |
| 5 | Semantic Error Recovery: Column Mismatch During Training Submission | 100% | 100% | 67.82 | 16.23 | [57.76, 77.88] | 41895 | 0 |

## Methodological Notes
- Objective success is computed from tool evidence (required/forbidden tools, schema validation, explicit assertions, distinctness, and recovery where applicable).
- For negative tests, success requires observing an expected backend error type, not the absence of Python exceptions.
