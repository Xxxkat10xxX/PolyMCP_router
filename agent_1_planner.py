from typing import List, Dict
from pydantic import BaseModel, Field
from contracts import SubTask, PipelineTrace
from llm_client import call_gemini_structured


class _DAGPlanWrapper(BaseModel):
    """Обертка для Structured Output списка подзадач (совместима с llm_client.py)."""
    subtasks: List[SubTask] = Field(default_factory=list)


class PlannerAgent:
    """Agent 1 (Даниил Епифанов): Семантическая декомпозиция запроса в DAG и динамический Re-planning."""

    SYSTEM_PROMPT = (
        "Ты — PlannerAgent в системе PolyMCP Router. "
        "Разбей сложный запрос пользователя на атомарные подзадачи (SubTask) для сервисов "
        "PostgreSQL, GitHub и Shell. Укажи целочисленный step_id (начиная с 1) и список depends_on "
        "(step_id шагов, от которых зависит текущий шаг; пустой список для параллельных веток)."
    )

    def run(self, user_query: str) -> List[SubTask]:
        user_prompt = f"Декомпозируй запрос в граф подзадач (DAG):\n{user_query}"
        result = call_gemini_structured(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=_DAGPlanWrapper,
        )
        return result.subtasks

    def replan(
        self,
        trace: PipelineTrace,
        failed_step: int,
        reason: str,
        max_replans: int = 2,
    ) -> List[SubTask]:
        """Динамическое перепланирование при сбое шага или замечании критика."""
        # Если сбой вызван блокировкой безопасности Policy Guard — обход запрещен
        failed_call = trace.tool_calls.get(failed_step)
        if failed_call and not failed_call.is_safe:
            return trace.dag_plan

        replan_count = getattr(trace, "replan_count", 0)
        if replan_count >= max_replans:
            return trace.dag_plan

        # Сохраняем уже выполненные шаги и перестраиваем сбойный узел
        completed_facts: Dict[int, str] = {
            k: v.distilled_facts for k, v in trace.condensed_memory.items() if k != failed_step
        }
        replan_prompt = (
            f"Исходный запрос: {trace.user_query}\n"
            f"Сбой на шаге {failed_step}. Причина: {reason}\n"
            f"Уже собранные факты: {completed_facts}\n"
            "Перестрой план подзадач для достижения цели другим путем."
        )
        result = call_gemini_structured(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=replan_prompt,
            response_schema=_DAGPlanWrapper,
        )
        return result.subtasks


if __name__ == "__main__":
    planner = PlannerAgent()
    test_queries = [
        "Сверь вчерашние коммиты в GitHub с открытыми багами в PostgreSQL и проверь логи через Shell",
    ]
    for idx, q in enumerate(test_queries, start=1):
        dag = planner.run(q)
        print(f"\n--- Тест #{idx}: {q} ---")
        for task in dag:
            print(f"  Шаг {task.step_id} (зависит от {task.depends_on}): {task.instruction}")