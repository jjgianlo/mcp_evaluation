"""mcp_setup.py

MCP client initialisation and tool schema loading for the evaluation harness.

Responsibilities:
- Authenticate against Azure AD and build an authorised MCP transport
- Connect to the remote MCP server and enumerate available tools
- Convert MCP tool schemas into OpenAI ChatCompletion tool definitions
- Validate tool call arguments against the MCP-provided JSON Schema
"""

from __future__ import annotations

import os
from typing import Any

from azure.identity import ClientSecretCredential
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from mcp.types import Tool
from openai.types.chat import ChatCompletionFunctionToolParam
from openai.types.shared_params import FunctionDefinition

from config import logger


async def setup_mcp_client() -> Client:
    credential = ClientSecretCredential(
        tenant_id=os.getenv("TENANT_ID"),
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_APP_SECRET"),
    )
    token = credential.get_token(f"{os.getenv('BACKEND_APP_ID')}/.default").token
    transport = StreamableHttpTransport(
        url=f"{os.getenv('REMOTE_URL')}/chronos/mcp",
        headers={
            "Authorization": f"Bearer {token}",
            "Ocp-Apim-Subscription-Key": os.getenv("API_KEY"),
        },
    )
    return Client(transport)


async def get_mcp_tools(
    mcp_client: Client,
) -> tuple[list[ChatCompletionFunctionToolParam], dict[str, dict]]:
    """Return OpenAI tool definitions AND the raw MCP inputSchema per tool name."""
    tools: list[Tool] = (await mcp_client.session.list_tools()).tools
    openai_tools: list[ChatCompletionFunctionToolParam] = []
    schemas: dict[str, dict] = {}
    for t in tools:
        schemas[t.name] = t.inputSchema or {}
        openai_tools.append(
            ChatCompletionFunctionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=t.name,
                    description=t.description,
                    parameters=t.inputSchema,
                ),
            )
        )
    logger.info("Loaded %d MCP tools", len(openai_tools))
    return openai_tools, schemas


def validate_tool_arguments(schema: dict, args: dict) -> tuple[bool, list[str]]:
    if not schema:
        # Some servers omit schemas; treat as unknown (non-failing) but flaggable.
        return True, []
    try:
        Draft202012Validator(schema).validate(args)
        return True, []
    except ValidationError as e:
        # Limit verbosity; keep the most actionable message.
        msg = f"{e.message} (path: {'/'.join([str(p) for p in e.path])})"
        return False, [msg]
    except Exception as e:
        return False, [f"Schema validation error: {e}"]
