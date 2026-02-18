"""config.py

Global configuration, constants, and logging setup for the MCP evaluation harness.
"""

import logging
import sys

# =========================
# Configuration & Constants
# =========================

CONFIG_REQUIRED_VARS = [
    "TENANT_ID",
    "CLIENT_ID",
    "CLIENT_APP_SECRET",
    "BACKEND_APP_ID",
    "REMOTE_URL",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "API_KEY",
    "OPENAI_API_KEY"
]

EVALUATION_SYSTEM_PROMPT = """You are an AI assistant with access to MCP tools for time-series forecasting.

When given a task, you MUST:
1. Use the available tools to complete the task (if told to AVOID certain tools, do not use them)
2. Provide summary of each step in your approach, wrapped in <summary>...</summary> tags
3. Provide feedback on the tools provided, wrapped in <feedback>...</feedback> tags
4. Provide your final response, wrapped in <response>...</response> tags

Response Requirements:
- Keep response concise and inside <response>...</response>
- If task cannot be completed: <response>NOT_FOUND</response>
"""

# =========================
# Logging
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.FileHandler("evaluation_audit.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
