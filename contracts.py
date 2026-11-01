from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class SubTask(BaseModel):
    step_id: int
    instruction: str
    depends_on: List[int] = Field(default_factory=list)


class PrunedTool(BaseModel):
    tool_name: str
    server_name: str
    pruned_schema: Dict[str, Any]
    raw_tokens: int
    pruned_tokens: int


class ToolCallPayload(BaseModel):
    step_id: int
    tool_name: str
    arguments: Dict[str, Any]
    is_safe: bool
    safety_note: str = ""


class ExecutionResult(BaseModel):
    step_id: int
    tool_name: str
    success: bool
    attempts: int
    error_recovered: bool = False
    raw_output: str
    raw_output_tokens: int


class CondensedStepState(BaseModel):
    step_id: int
    distilled_facts: str
    condensed_tokens: int


class PipelineTrace(BaseModel):
    user_query: str
    dag_plan: List[SubTask] = Field(default_factory=list)
    pruned_tools_per_step: Dict[int, List[PrunedTool]] = Field(default_factory=dict)
    tool_calls: Dict[int, ToolCallPayload] = Field(default_factory=dict)
    executions: Dict[int, ExecutionResult] = Field(default_factory=dict)
    condensed_memory: Dict[int, CondensedStepState] = Field(default_factory=dict)
    final_answer: str = ""
    is_complete: bool = False