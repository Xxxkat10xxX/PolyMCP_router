"""
agent_5_condenser.py — Узел №5 конвейера PolyMCP Router.
Агент семантической дистилляции контекста A2A и финальной верификации (CondenserCriticAgent).
Разработчик: Негоденко Савелий Кириллович (Б26-504).
"""

from typing import Dict
from pydantic import BaseModel, Field

from contracts import (
    SubTask,
    ExecutionResult,
    CondensedStepState,
    PipelineTrace,
)
from llm_client import call_gemini_structured, count_tokens


class _DistilledWrapper(BaseModel):
    """Схема Structured Output для семантического сжатия сырого ответа MCP-сервера."""
    distilled_facts: str = Field(
        ...,
        description=(
            "Максимально плотная выжимка только ключевых фактов, ID, чисел, статусов "
            "и ошибок, строго необходимых для выполнения инструкции подзадачи."
        ),
    )


class _CriticVerdict(BaseModel):
    """Схема Structured Output для финального вердикта Агента-Критика."""
    is_complete: bool = Field(
        ...,
        description=(
            "True, если собранных в памяти A2A фактов достаточно для полного ответа "
            "на исходный запрос пользователя. False, если требуется перепланирование (replan)."
        ),
    )
    final_answer: str = Field(
        ...,
        description=(
            "Итоговый структурированный технический ответ пользователю на основе всех "
            "дистиллированных фактов, либо описание недостающих данных при is_complete=False."
        ),
    )


class CondenserCriticAgent:
    """
    Узел №5: CondenserCriticAgent.
    Решает проблему раздувания межшаговой памяти A2A от гигантских сырых ответов MCP-серверов
    и проводит финальную ревизию полноты выполнения пользовательского запроса.
    """

    CONDENSE_SYSTEM_PROMPT = (
        "Ты — CondenserAgent (Узел №5 в архитектуре PolyMCP Router). "
        "Твоя задача — выполнять целевую семантическую дистилляцию гигантских сырых ответов "
        "от MCP-серверов (дампов таблиц PostgreSQL, JSON-ответов GitHub API, логов Shell). "
        "Правила дистилляции:\n"
        "1. Извлекай ТОЛЬКО конкретные идентификаторы (ID, SHA, UUID), числовые метрики, "
        "временные метки, статусы и коды ошибок, которые напрямую отвечают на инструкцию подзадачи.\n"
        "2. Полностью удаляй пустые поля, системный шум, повторяющиеся трейсы, "
        "декоративное форматирование и нерелевантные строки логов.\n"
        "3. Сожми объем текста минимум на 85-95% без потери семантики для следующих шагов DAG."
    )

    CRITIC_SYSTEM_PROMPT = (
        "Ты — CriticAgent (финальный верификатор конвейера PolyMCP Router). "
        "Ты получаешь исходный многосоставной запрос пользователя, план подзадач DAG "
        "и агрегированную сжатую память A2A (condensed_memory) после завершения всех веток. "
        "Твоя задача:\n"
        "1. Проверить, на все ли части исходного запроса найдены подтвержденные факты.\n"
        "2. Если все подзадачи выполнены успешно и данных достаточно — выставить is_complete=True "
        "и синтезировать четкий, технически точный итоговый ответ (final_answer).\n"
        "3. Если ключевой шаг упал или данных недостаточно для ответа — выставить is_complete=False "
        "и явно указать в final_answer, каких фактов не хватает для запуска перепланирования (replan)."
    )

    def __init__(self, bypass_token_threshold: int = 100) -> None:
        self.bypass_token_threshold = bypass_token_threshold

    def condense_step(
        self,
        execution: ExecutionResult,
        subtask: SubTask,
    ) -> CondensedStepState:
        """
        Дистиллирует сырой ответ MCP-сервера (ExecutionResult) под задачу SubTask.
        Использует механизм раннего выхода (Bypass), если ответ уже короче порога.
        """
        raw_text = execution.raw_output or ""
        raw_tokens = (
            execution.raw_output_tokens
            if execution.raw_output_tokens > 0
            else count_tokens(raw_text)
        )

        # Механизм раннего выхода (Bypass) для коротких ответов без вызова LLM
        if raw_tokens < self.bypass_token_threshold:
            condensed_tokens = count_tokens(raw_text)
            return CondensedStepState(
                step_id=execution.step_id,
                distilled_facts=raw_text.strip(),
                condensed_tokens=condensed_tokens,
            )

        user_prompt = (
            f"ШАГ DAG #{subtask.step_id}\n"
            f"ЦЕЛЕВАЯ ИНСТРУКЦИЯ ПОДЗАДАЧИ: {subtask.instruction}\n"
            f"ВЫЗВАННЫЙ ИНСТРУМЕНТ: {execution.tool_name} "
            f"(успех: {execution.success}, попыток: {execution.attempts}, "
            f"восстановление после сбоя: {execution.error_recovered})\n\n"
            f"СЫРОЙ ОТВЕТ СЕРВЕРА ({raw_tokens} токенов):\n"
            f"{raw_text}\n\n"
            "Сформируй предельно сжатую сводку ключевых фактов (distilled_facts) для памяти A2A."
        )

        distilled_response: _DistilledWrapper = call_gemini_structured(
            system_prompt=self.CONDENSE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=_DistilledWrapper,
        )

        distilled_text = distilled_response.distilled_facts.strip()
        condensed_tokens = count_tokens(distilled_text)

        return CondensedStepState(
            step_id=execution.step_id,
            distilled_facts=distilled_text,
            condensed_tokens=condensed_tokens,
        )

    def verify_and_answer(self, trace: PipelineTrace) -> PipelineTrace:
        """
        Агрегирует словарь condensed_memory после выполнения всех веток DAG,
        проверяет соответствие исходному user_query и формирует финальный вердикт.
        """
        subtask_map: Dict[int, SubTask] = {st.step_id: st for st in trace.dag_plan}

        # Страхующая дистилляция: если какой-то шаг есть в executions, но еще не в condensed_memory
        for step_id, execution in trace.executions.items():
            if step_id not in trace.condensed_memory:
                subtask = subtask_map.get(
                    step_id,
                    SubTask(step_id=step_id, instruction=f"Выполнение шага #{step_id}", depends_on=[]),
                )
                trace.condensed_memory[step_id] = self.condense_step(execution, subtask)

        if not trace.condensed_memory:
            trace.is_complete = False
            trace.final_answer = (
                "Конвейер не получил данных ни от одного MCP-инструмента. "
                "Требуется повторное планирование запроса."
            )
            return trace

        memory_lines = []
        for step_id in sorted(trace.condensed_memory.keys()):
            mem_state = trace.condensed_memory[step_id]
            subtask = subtask_map.get(step_id)
            instr = subtask.instruction if subtask else f"Шаг #{step_id}"
            exec_res = trace.executions.get(step_id)
            status_tag = "OK" if (exec_res is None or exec_res.success) else "FAILED"
            memory_lines.append(
                f"- Шаг #{step_id} [{status_tag}] ({instr}): {mem_state.distilled_facts} "
                f"[сжато до {mem_state.condensed_tokens} ток.]"
            )

        aggregated_memory_str = "\n".join(memory_lines)

        user_prompt = (
            f"ИСХОДНЫЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ:\n{trace.user_query}\n\n"
            f"СОБРАННЫЕ ФАКТЫ В СЖАТОЙ ПАМЯТИ A2A (condensed_memory):\n"
            f"{aggregated_memory_str}\n\n"
            "Проверь полноту выполнения всех пунктов запроса и сформируй итоговый ответ."
        )

        verdict: _CriticVerdict = call_gemini_structured(
            system_prompt=self.CRITIC_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=_CriticVerdict,
        )

        trace.is_complete = verdict.is_complete
        trace.final_answer = verdict.final_answer.strip()
        return trace

    def run(self, trace: PipelineTrace) -> PipelineTrace:
        """Единая точка входа для вызова из оркестратора pipeline.py."""
        return self.verify_and_answer(trace)


if __name__ == "__main__":
    print("=" * 78)
    print("ТЕСТ УЗЛА №5: CondenserCriticAgent (Семантическая дистилляция + Агент-Критик)")
    print("=" * 78)

    agent = CondenserCriticAgent(bypass_token_threshold=100)

    # 1. Генерируем реалистичный многословный сырой дамп (550+ слов / >1300 токенов),
    # имитирующий выгрузку PostgreSQL и системного журнала Shell MCP
    verbose_log_entries = [
        "[2026-11-05T08:00:01.104Z] INFO [pg-pool-worker-01] Connection established to postgresql://prod-db.internal:5432/incidents_db (sslmode=verify-full, application_name=polymcp_probe)",
        "[2026-11-05T08:00:01.119Z] DEBUG [query-planner] Parsing SQL statement: SELECT id, severity, title, status, assigned_team, commit_ref, created_at, diagnostic_dump FROM bugs WHERE status='open' ORDER BY severity DESC;",
        "[2026-11-05T08:00:01.145Z] TRACE [storage-engine] Scanning index idx_bugs_status_severity on heap pages 1024..2048, buffer hits=842, disk reads=0,shared_blks_dirtied=0.",
    ]

    # Наполняем лог избыточными техническими строками телеметрии (более 550 слов)
    for idx in range(1, 38):
        verbose_log_entries.append(
            f"[2026-11-05T08:00:{idx:02d}.412Z] DEBUG [telemetry-heartbeat-{idx:03d}] "
            f"Worker thread pool healthcheck passed: active_threads=16, idle_threads=48, "
            f"heap_allocated_mb=1024.5, gc_pause_ms=1.2, epoll_wait_Descriptors=128, "
            f"network_rx_bytes=4096512, network_tx_bytes=2048112, status=HEALTHY_NO_ACTION_REQUIRED."
        )

    # Вставляем целевые факты в середину и конец гигантского лога
    verbose_log_entries.extend([
        "[2026-11-05T08:00:39.002Z] RESULT_ROW_1: {\"bug_id\": 104, \"severity\": \"HIGH\", \"status\": \"open\", \"title\": \"Memory leak in OAuth token refresh\", \"linked_commit\": \"a1b2c1\", \"note\": \"Patched in staging, awaiting verification\"}",
        "[2026-11-05T08:00:39.005Z] RESULT_ROW_2: {\"bug_id\": 109, \"severity\": \"CRITICAL\", \"status\": \"open\", \"title\": \"Queue worker deadlock on Redis timeout\", \"linked_commit\": null, \"note\": \"No fix in yesterday commits (a1b2c1 does not touch queue_worker.py)\"}",
        "[2026-11-05T08:00:40.510Z] ERROR [shell-mcp-tail] /var/log/polymcp/worker.log: CRITICAL WorkerTimeoutError in task_runner(ticket=#109): Lock acquisition timed out after 30000ms on shard_03.",
    ])

    raw_heavy_output = "\n".join(verbose_log_entries)
    word_count = len(raw_heavy_output.split())
    raw_token_count = count_tokens(raw_heavy_output)

    subtask_step_3 = SubTask(
        step_id=3,
        instruction="Сверить открытые критические баги из PostgreSQL с коммитом a1b2c1 и проверить ошибки воркера в логах Shell MCP",
        depends_on=[1, 2],
    )

    heavy_execution = ExecutionResult(
        step_id=3,
        tool_name="shell_tail_log",
        success=True,
        attempts=2,
        error_recovered=True,
        raw_output=raw_heavy_output,
        raw_output_tokens=raw_token_count,
    )

    # 2. Запускаем дистилляцию тяжелого ответа
    condensed_state = agent.condense_step(heavy_execution, subtask_step_3)
    compression_ratio = (1.0 - (condensed_state.condensed_tokens / max(raw_token_count, 1))) * 100

    print(f"[Тест 1: Дистилляция гигантского лога]")
    print(f"  • Объем сырого ответа (слов):    {word_count} слов")
    print(f"  • Сырые токены (raw_tokens):     {raw_token_count} ток.")
    print(f"  • Сжатые токены (condensed):     {condensed_state.condensed_tokens} ток.")
    print(f"  • Коэффициент сжатия памяти:     {compression_ratio:.1f}% (целевой порог КП2 >= 50%)")
    print(f"  • Выжатые факты A2A:             {condensed_state.distilled_facts}\n")

    # 3. Тестируем Bypass (ранний выход для коротких ответов < 100 токенов)
    short_raw = "GitHub MCP: последний коммит вчера SHA a1b2c1 (автор: dev_42)."
    short_exec = ExecutionResult(
        step_id=1,
        tool_name="gh_get_commit_history",
        success=True,
        attempts=1,
        error_recovered=False,
        raw_output=short_raw,
        raw_output_tokens=count_tokens(short_raw),
    )
    subtask_step_1 = SubTask(step_id=1, instruction="Получить вчерашние коммиты из GitHub", depends_on=[])
    bypass_state = agent.condense_step(short_exec, subtask_step_1)
    print(f"[Тест 2: Bypass короткого ответа (< {agent.bypass_token_threshold} токенов)]")
    print(f"  • Токены до/после: {short_exec.raw_output_tokens} -> {bypass_state.condensed_tokens} ток. (LLM не вызывалась)\n")

    # 4. Тестируем финальную верификацию Агента-Критика (verify_and_answer)
    sample_trace = PipelineTrace(
        user_query="Сверь вчерашние коммиты в GitHub с открытыми багами в PostgreSQL и проверь логи через Shell",
        dag_plan=[
            subtask_step_1,
            SubTask(step_id=2, instruction="Выбрать открытые критические баги в PostgreSQL", depends_on=[]),
            subtask_step_3,
        ],
        executions={1: short_exec, 3: heavy_execution},
        condensed_memory={
            1: bypass_state,
            2: CondensedStepState(
                step_id=2,
                distilled_facts="В таблице issues открыты 2 бага: #104 (HIGH) и #109 (CRITICAL, нет фикса).",
                condensed_tokens=count_tokens("В таблице issues открыты 2 бага: #104 (HIGH) и #109 (CRITICAL, нет фикса)."),
            ),
            3: condensed_state,
        },
    )

    verified_trace = agent.verify_and_answer(sample_trace)
    print("[Тест 3: Финальный вердикт Агента-Критика (verify_and_answer)]")
    print(f"  • Статус завершенности (is_complete): {verified_trace.is_complete}")
    print(f"  • Итоговый ответ (final_answer):      {verified_trace.final_answer}")
    print("=" * 78)