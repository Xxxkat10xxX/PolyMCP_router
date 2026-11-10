from pydantic import BaseModel
from contracts import CondensedStepState, ExecutionResult, PipelineTrace, SubTask
from llm_client import call_llm_structured, count_tokens


class _DistilledWrapper(BaseModel):
    distilled_facts: str


class _CriticVerdict(BaseModel):
    is_complete: bool
    final_answer: str


class CondenserCriticAgent:
    """
    Agent 5 (Савелий Негоденко): Семантическая дистилляция межшаговой памяти A2A
    и финальная верификация ответа (Critic Node).
    """

    CONDENSE_PROMPT = (
        "Ты — CondenserCriticAgent в системе PolyMCP Router. "
        "Сожми сырой JSON-ответ MCP-сервера в краткую сводку ключевых фактов (ID, SHA, статусы) "
        "для передачи следующим узлам графа по протоколу A2A."
    )

    CRITIC_PROMPT = (
        "Ты — финальный Агент-Критик в системе PolyMCP Router. "
        "Проверь собранные факты по всем шагам DAG и сформируй итоговый ответ пользователю."
    )

    def condense_step(self, subtask: SubTask, exec_result: ExecutionResult) -> CondensedStepState:
        # Bypass для коротких ответов (например, при блокировке Policy Guard)
        if exec_result.raw_output_tokens <= 45:
            return CondensedStepState(
                step_id=subtask.step_id,
                distilled_facts=exec_result.raw_output,
                condensed_tokens=exec_result.raw_output_tokens,
            )

        user_prompt = (
            f"Подзадача шага {subtask.step_id}: {subtask.instruction}\n"
            f"Сырой ответ инструмента {exec_result.tool_name}: {exec_result.raw_output}\n"
            "Выдели только ключевые факты для следующих шагов."
        )
        res = call_llm_structured(
            system_prompt=self.CONDENSE_PROMPT,
            user_prompt=user_prompt,
            response_schema=_DistilledWrapper,
        )
        condensed_tokens = count_tokens(res.distilled_facts)
        return CondensedStepState(
            step_id=subtask.step_id,
            distilled_facts=res.distilled_facts,
            condensed_tokens=condensed_tokens,
        )

    def verify_and_answer(self, trace: PipelineTrace) -> PipelineTrace:
        memory_dump = {k: v.distilled_facts for k, v in trace.condensed_memory.items()}
        user_prompt = (
            f"Исходный запрос: {trace.user_query}\n"
            f"Сжатая память шагов A2A: {memory_dump}\n"
            "Оцени полноту выполнения и сформируй финальный ответ."
        )
        verdict = call_llm_structured(
            system_prompt=self.CRITIC_PROMPT,
            user_prompt=user_prompt,
            response_schema=_CriticVerdict,
        )
        trace.is_complete = verdict.is_complete
        trace.final_answer = verdict.final_answer
        return trace