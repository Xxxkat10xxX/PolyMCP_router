from typing import Dict, List, Optional
from contracts import CondensedStepState, PrunedTool, SubTask, ToolCallPayload
from llm_client import call_llm_structured


class ArgSynthesizerAgent:
    """
    Agent 3 (Максим Ларютин): Синтез аргументов вызова с учетом памяти A2A
    и пре-флайт проверкой безопасности (Semantic Policy Guard).
    """

    SYSTEM_PROMPT = (
        "Ты — ArgSynthesizerAgent и Semantic Policy Guard в системе PolyMCP Router. "
        "Сформируй JSON-вызов (ToolCallPayload) для выбранного MCP-инструмента на основе инструкции "
        "подзадачи, сжатой схемы и фактов из зависимых шагов (depends_on). "
        "Если обнаружена деструктивная команда (DROP TABLE, TRUNCATE, DELETE FROM, rm -rf, mkfs) "
        "или попытка утечки секретов, выстави is_safe=False и укажи причину в safety_note."
    )

    _DESTRUCTIVE_PATTERNS = ("DROP TABLE", "TRUNCATE", "DELETE FROM", "RM -RF", "MKFS", "SHUTDOWN")

    def _post_validate_safety(self, instruction: str, payload: ToolCallPayload) -> ToolCallPayload:
        combined_text = f"{instruction} {payload.arguments}".upper()
        for pattern in self._DESTRUCTIVE_PATTERNS:
            if pattern in combined_text:
                payload.is_safe = False
                payload.safety_note = f"Semantic Policy Guard: заблокирована опасная операция ({pattern})"
                return payload
        return payload

    def run(
        self,
        subtask: SubTask,
        pruned_tools: List[PrunedTool],
        memory: Optional[Dict[int, CondensedStepState]] = None,
    ) -> ToolCallPayload:
        memory = memory or {}
        if not pruned_tools:
            return ToolCallPayload(
                step_id=subtask.step_id,
                tool_name="none",
                arguments={},
                is_safe=False,
                safety_note="Не найдено подходящих инструментов",
            )

        target_tool = pruned_tools[0]
        relevant_facts = {
            step_id: state.distilled_facts
            for step_id, state in memory.items()
            if step_id in subtask.depends_on
        }

        user_prompt = (
            f"Подзадача (step_id={subtask.step_id}): {subtask.instruction}\n"
            f"Целевой инструмент: {target_tool.tool_name} ({target_tool.server_name})\n"
            f"Сжатая JSON-схема: {target_tool.pruned_schema}\n"
            f"Контекст A2A от шагов {subtask.depends_on}: {relevant_facts}\n"
            "Сформируй аргументы вызова и проверь безопасность."
        )

        result = call_llm_structured(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=ToolCallPayload,
        )

        result.step_id = subtask.step_id
        valid_names = {t.tool_name for t in pruned_tools}
        if result.tool_name not in valid_names:
            result.tool_name = target_tool.tool_name

        # Адаптируем ключи аргументов под конкретный домен
        allowed_props = set(target_tool.pruned_schema.get("properties", {}).keys())
        cleaned_args = {k: v for k, v in result.arguments.items() if k in allowed_props or k == "user_id"}
        if "query" not in cleaned_args:
            cleaned_args["query"] = subtask.instruction[:80]
        if target_tool.server_name == "github_mcp" and "repo" in allowed_props:
            cleaned_args.setdefault("repo", "org/polymcp-service")
        elif target_tool.server_name == "shell_mcp" and "command" in allowed_props:
            cleaned_args.setdefault("command", "tail -n 50 /var/log/syslog")

        result.arguments = cleaned_args
        return self._post_validate_safety(subtask.instruction, result)


if __name__ == "__main__":
    from agent_1_planner import PlannerAgent
    from agent_2_pruner import SchemaPrunerAgent

    planner = PlannerAgent()
    pruner = SchemaPrunerAgent()
    synthesizer = ArgSynthesizerAgent()

    tasks = planner.run("Сверь вчерашние коммиты в GitHub с открытыми багами в PostgreSQL")
    for t in tasks:
        tools = pruner.run(t)
        call = synthesizer.run(t, tools, {})
        print(f"Шаг {call.step_id} -> {call.tool_name} | Safe: {call.is_safe} | Args: {call.arguments}")