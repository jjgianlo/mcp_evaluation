"""mcp_eval

MCP Server Evaluation Harness (Thesis-grade)

Submodules
----------
config      - constants, logging, system prompt
models      - dataclasses and enums
utils       - shared helpers (JSON-path, error classification, stats)
xml_parser  - evaluation_v2.xml → TaskCriteria
mcp_setup   - Azure AD auth, MCP client, tool schema loading
agent       - async LLM ↔ MCP agent loop
evaluator   - assertion checks and objective success scoring
reporting   - CSV flattening, task-level aggregation, Markdown report
main        - orchestration entrypoint (CLI)
"""
