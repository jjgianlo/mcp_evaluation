"""models.py

Data models (dataclasses and enums) for the MCP evaluation harness.

Covers:
- ErrorType              - enum for backend error classification
- ToolRequirement        - a required/optional tool with a min-call count
- Assertion              - JSON-path assertion on tool arguments or results
- QuantileMonotonicCheck - monotonicity check for probabilistic forecasts
- DistinctCheck          - diversity check across repeated tool invocations
- TaskCriteria           - full specification of one evaluation task (parsed from XML)
- TimingBreakdown        - latency components for one task run
- ErrorEvent             - a single error/recovery event during an agent loop
- ToolInvocation         - one tool call with arguments, result, and schema validity
- SuccessAssessment      - objective scoring of one task run
- TaskResult             - complete record for one task run
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


# =========================
# Error Classification
# =========================


class ErrorType(Enum):
    MISSING_DATASET = "missing_dataset"
    INVALID_PARAMETER = "invalid_parameter"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    AUTH_OR_QUOTA = "auth_or_quota"
    UNKNOWN = "unknown"


# =========================
# Criteria Model (XML-driven)
# =========================


@dataclass
class ToolRequirement:
    name: str
    required: bool = True
    min_calls: int = 1


@dataclass
class Assertion:
    """Simple JSON-path assertion on either tool arguments or tool results."""

    target: str  # "arguments" | "result"
    tool: str
    path: str
    equals: Optional[str] = None
    contains: Optional[str] = None
    length: Optional[int] = None
    # Which invocation to check if a tool is called multiple times: first|last|<1-based index>
    occurrence: str = "last"


@dataclass
class QuantileMonotonicCheck:
    tool: str
    lower_path: str
    median_path: str
    upper_path: str


@dataclass
class DistinctCheck:
    tool: str
    target: str  # "arguments" | "result"
    path: str
    min_distinct: int = 1


@dataclass
class TaskCriteria:
    task_id: int
    task_name: str
    prompt: str

    # Tool requirements
    expected_tools: list[ToolRequirement] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)

    # Assertions
    argument_assertions: list[Assertion] = field(default_factory=list)
    output_assertions: list[Assertion] = field(default_factory=list)
    quantile_monotonic_checks: list[QuantileMonotonicCheck] = field(default_factory=list)

    # Additional structural validity checks
    distinct_checks: list[DistinctCheck] = field(default_factory=list)

    # Recovery-specific tasks (fault injection → recovery → success)
    requires_recovery: bool = False

    # Negative tests
    should_fail_gracefully: bool = False
    expected_error_types: list[str] = field(default_factory=list)


# =========================
# Audit / Result Models
# =========================


@dataclass
class TimingBreakdown:
    total_latency: float
    llm_reasoning_latency: float
    tool_execution_latency: float


@dataclass
class ErrorEvent:
    iteration: int
    tool_name: Optional[str]
    error_type: ErrorType
    error_message: str
    backend_error_type: Optional[str] = None
    was_recovered: bool = False

    # Optional recovery telemetry (keeps old JSON readable; adds new value when available)
    recovered_iteration: Optional[int] = None
    error_elapsed_s: Optional[float] = None
    recovered_elapsed_s: Optional[float] = None


@dataclass
class ToolInvocation:
    name: str
    arguments: dict
    argument_schema_valid: bool
    argument_schema_errors: list[str]
    result_data: Any
    result_snippet: str
    execution_time: float


@dataclass
class SuccessAssessment:
    overall_success: bool
    objective_success: bool
    required_tools_missing: list[str]
    forbidden_tools_used: list[str]
    schema_validation_pass: bool
    assertion_pass: bool
    assertion_failures: list[str]
    distinct_pass: bool
    distinct_failures: list[str]
    recovery_pass: bool
    expected_error_observed: bool
    observed_backend_error_types: list[str]
    confidence: float
    verification_required: bool
    reasoning: str
    distinct_checks_pass: bool


@dataclass
class TaskResult:
    task_id: int
    run_id: int
    prompt_snippet: str
    full_response_text: str
    actual_response: Optional[str]
    score: int
    total_duration: float
    timing_breakdown: TimingBreakdown
    tool_invocations: list[ToolInvocation] = field(default_factory=list)
    error_events: list[ErrorEvent] = field(default_factory=list)
    token_usage: dict = field(default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0})
    success_assessment: Optional[SuccessAssessment] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timing_breakdown"] = asdict(self.timing_breakdown)
        d["tool_invocations"] = [asdict(t) for t in self.tool_invocations]
        d["error_events"] = [
            {**asdict(e), "error_type": e.error_type.value} for e in self.error_events
        ]
        d["success_assessment"] = asdict(self.success_assessment) if self.success_assessment else None
        return d
