"""
agent_4_executor.py — Финальный Узел №4 конвейера PolyMCP Router.
Автор: Никита Чумазин (Участник №4).

Назначение:
1. Исполнение ToolCallPayload с безусловной пре-флайт блокировкой опасных операций.
2. Механизм Self-Healing при транзиентных сбоях MCP-серверов.
3. Дистилляция многословных ответов серверов в сжатую A2A-память (CondensedStepState).
4. Синтез итогового отчета пользователя (synthesize_final_answer).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, Field

from contracts import (
    CondensedStepState,
    ExecutionResult,
    PipelineTrace,
    ToolCallPayload,
)
from llm_client import call_llm_structured, count_tokens


# ============================================================================
# Pydantic-схемы для структурированного вывода LLM при дистилляции и синтезе
# ============================================================================

class DistillationResponse(BaseModel):
    distilled_facts: str = Field(
        description=(
            "Краткая концентрированная выжимка фактов (до 40-50 слов). "
            "Обязательно сохраняй конкретные идентификаторы: хеши коммитов (a1b2c3d, e4f5a6b), "
            "номера багов (BUG-104, BUG-108), имена сервисов (auth-service, router-core) и статусы."
        )
    )


class FinalReportResponse(BaseModel):
    report: str = Field(
        description="Структурированный и понятный итоговый ответ для пользователя по результатам всех шагов."
    )


# ============================================================================
# Узел №4: ExecutorAgent
# ============================================================================

class ExecutorAgent:
    """
    Исполнитель MCP-вызовов с поддержкой пре-флайт безопасности,
    Self-Healing при сбоях и семантической дистилляции контекста.
    """

    def __init__(self, model_name: str = "gpt-6-luna"):
        self.model_name = model_name

    # ------------------------------------------------------------------------
    # 1. Исполнение вызова с обработкой безопасности и Self-Healing
    # ------------------------------------------------------------------------

    def execute(
        self,
        payload: ToolCallPayload,
        simulate_transient_error: bool = False,
    ) -> ExecutionResult:
        """
        Исполняет подготовленный ToolCallPayload.
        Если payload.is_safe == False, вызов немедленно блокируется без обращения к MCP.
        Если simulate_transient_error == True, симулируется транзиентный сбой с автовосстановлением.
        """
        # Пре-флайт блокировка деструктивных вызовов
        if not payload.is_safe:
            raw_output = f"[MCP EXECUTION BLOCKED] {payload.safety_note}"
            return ExecutionResult(
                step_id=payload.step_id,
                tool_name=payload.tool_name,
                success=False,
                attempts=0,
                error_recovered=False,
                raw_output=raw_output,
                raw_output_tokens=count_tokens(raw_output),
            )

        # Симуляция Self-Healing (транзиентный сбой на первой попытке)
        if simulate_transient_error:
            # Попытка 1 завершилась сбоем, агент автоматически повторил с коррекцией параметров
            attempts = 2
            error_recovered = True
            raw_data = self._dispatch_mock_mcp(payload.tool_name, payload.arguments)
            raw_output = (
                "[Self-Healing Warning]: Попытка 1 завершилась таймаутом шлюза (504 Gateway Timeout). "
                "Автоматическая перекалибровка сокета и параметров вызова выполнена успешно.\n"
                f"Результат попытки 2:\n{raw_data}"
            )
            return ExecutionResult(
                step_id=payload.step_id,
                tool_name=payload.tool_name,
                success=True,
                attempts=attempts,
                error_recovered=error_recovered,
                raw_output=raw_output,
                raw_output_tokens=count_tokens(raw_output),
            )

        # Штатное успешное выполнение
        raw_output = self._dispatch_mock_mcp(payload.tool_name, payload.arguments)
        return ExecutionResult(
            step_id=payload.step_id,
            tool_name=payload.tool_name,
            success=True,
            attempts=1,
            error_recovered=False,
            raw_output=raw_output,
            raw_output_tokens=count_tokens(raw_output),
        )

    # ------------------------------------------------------------------------
    # 2. Дистилляция сырого вывода MCP в сжатую память A2A
    # ------------------------------------------------------------------------

    def condense(self, execution_result: ExecutionResult) -> CondensedStepState:
        """
        Сжимает многословный сырой ответ MCP-сервера (150-350 токенов)
        в ультракомпактную выжимку фактов для последующих шагов DAG.
        """
        # Если шаг был заблокирован политикой безопасности
        if not execution_result.success:
            distilled = f"Шаг {execution_result.step_id} ({execution_result.tool_name}) заблокирован политикой безопасности."
            return CondensedStepState(
                step_id=execution_result.step_id,
                distilled_facts=distilled,
                condensed_tokens=count_tokens(distilled),
            )

        system_prompt = (
            "Ты — модуль дистилляции памяти PolyMCP Router. "
            "Твоя задача — извлечь из сырого технического вывода MCP-сервера только критические факты. "
            "Обязательно сохраняй конкретные идентификаторы (коммиты a1b2c3d, e4f5a6b, баги BUG-104, BUG-108, "
            "имена сервисов auth-service, router-core) и их статусы. Отсекай весь JSON-мусор, заголовки и служебные логи."
        )

        user_prompt = (
            f"Инструмент: {execution_result.tool_name}\n"
            f"Сырой вывод MCP:\n{execution_result.raw_output}\n\n"
            "Сформируй лаконичную выжимку фактов (до 40 слов):"
        )

        try:
            response: DistillationResponse = call_llm_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=DistillationResponse,
                model_name=self.model_name,
            )
            distilled_facts = response.distilled_facts.strip()
        except Exception:
            # Надежный детерминированный фоллбэк при недоступности внешнего API
            distilled_facts = self._deterministic_fallback_condense(execution_result)

        # Гарантируем наличие ключевых сущностей, если они присутствовали в сыром выводе
        distilled_facts = self._ensure_vital_entities(execution_result.raw_output, distilled_facts)

        return CondensedStepState(
            step_id=execution_result.step_id,
            distilled_facts=distilled_facts,
            condensed_tokens=count_tokens(distilled_facts),
        )

    # ------------------------------------------------------------------------
    # 3. Комплексный прогон шага (execute + condense)
    # ------------------------------------------------------------------------

    def run(
        self,
        payload: ToolCallPayload,
        simulate_transient_error: bool = False,
    ) -> Tuple[ExecutionResult, CondensedStepState]:
        """Универсальная точка входа для исполнения шага конвейера."""
        exec_result = self.execute(payload, simulate_transient_error=simulate_transient_error)
        condensed_state = self.condense(exec_result)
        return exec_result, condensed_state

    # ------------------------------------------------------------------------
    # 4. Синтез финального ответа для пользователя
    # ------------------------------------------------------------------------

    def synthesize_final_answer(self, trace: PipelineTrace) -> str:
        """
        Формирует итоговый структурированный отчет на базе всех шагов PipelineTrace.
        Устанавливает trace.final_answer и trace.is_complete = True.
        """
        memory_summary = "\n".join(
            f"- Шаг {step_id}: {state.distilled_facts}"
            for step_id, state in sorted(trace.condensed_memory.items())
        )

        system_prompt = (
            "Ты — ведущий ассистент PolyMCP Router. "
            "Сформируй понятный, профессиональный итоговый отчет для пользователя по его запросу, "
            "опираясь исключительно на подтвержденные факты из дистиллированной памяти шагов."
        )

        user_prompt = (
            f"Запрос пользователя: {trace.user_query}\n\n"
            f"Факты по шагам выполнения:\n{memory_summary}\n\n"
            "Предоставь четкий финальный ответ с выводами:"
        )

        try:
            response: FinalReportResponse = call_llm_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=FinalReportResponse,
                model_name=self.model_name,
            )
            final_text = response.report.strip()
        except Exception:
            # Детерминированный фоллбэк
            final_text = (
                f"Запрос «{trace.user_query}» успешно обработан конвейером PolyMCP Router.\n"
                f"Итоговые результаты по шагам:\n{memory_summary}"
            )

        trace.final_answer = final_text
        trace.is_complete = True
        return final_text

    # ------------------------------------------------------------------------
    # Вспомогательные эмуляторы MCP и детерминированные экстракторы
    # ------------------------------------------------------------------------

    def _dispatch_mock_mcp(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Генерирует реалистичный многословный технический ответ MCP-сервера на 150-300 токенов."""
        arg_query = str(arguments.get("query", ""))
        arg_repo = str(arguments.get("repo", "polymcp/router"))
        arg_cmd = str(arguments.get("command", ""))

        if "commit" in tool_name or "github" in tool_name or "gh_" in tool_name:
            data = {
                "status": 200,
                "protocol": "mcp/1.0",
                "server": "github_mcp_v2.4",
                "repository": arg_repo,
                "filter_query": arg_query,
                "response_meta": {
                    "rate_limit_remaining": 4982,
                    "x_request_id": "gh-trace-99120-a1b2",
                    "execution_time_ms": 142,
                },
                "commits": [
                    {
                        "sha": "a1b2c3d",
                        "full_sha": "a1b2c3d98ef0123456789abcdef0123456789abc",
                        "author": {"name": "daniil-planner", "email": "daniil@polymcp.internal"},
                        "date": "2026-11-04T19:30:12Z",
                        "message": "fix(auth-service): resolve token session deadlock (closes BUG-104)",
                        "affected_services": ["auth-service"],
                        "stats": {"additions": 48, "deletions": 14, "files": 2},
                    },
                    {
                        "sha": "e4f5a6b",
                        "full_sha": "e4f5a6b0123456789abcdef0123456789abcdef0",
                        "author": {"name": "roman-pruner", "email": "roman@polymcp.internal"},
                        "date": "2026-11-04T18:15:40Z",
                        "message": "perf(router-core): optimize context window compaction (ref BUG-108)",
                        "affected_services": ["router-core"],
                        "stats": {"additions": 112, "deletions": 35, "files": 3},
                    },
                ],
            }
            return json.dumps(data, ensure_ascii=False, indent=2)

        if "db_" in tool_name or "postgresql" in tool_name or "sql" in tool_name:
            data = {
                "status": "QUERY_SUCCESS",
                "server": "postgresql_mcp_primary.internal:5432",
                "database": "production_tracker",
                "executed_query": arg_query,
                "query_plan": {
                    "Node Type": "Index Scan",
                    "Relation Name": "issue_tickets",
                    "Startup Cost": 0.28,
                    "Total Cost": 14.82,
                    "Plan Rows": 2,
                    "Plan Width": 196,
                },
                "rows_returned": 2,
                "records": [
                    {
                        "bug_id": "BUG-104",
                        "service": "auth-service",
                        "severity": "CRITICAL",
                        "status": "RESOLVED_PENDING_VERIFY",
                        "linked_commit": "a1b2c3d",
                        "summary": "Deadlock under concurrent session refresh in auth-service",
                    },
                    {
                        "bug_id": "BUG-108",
                        "service": "router-core",
                        "severity": "HIGH",
                        "status": "IN_PROGRESS",
                        "linked_commit": "e4f5a6b",
                        "summary": "Race condition in streaming token truncation for router-core",
                    },
                ],
                "telemetry": {"shared_blks_hit": 24, "shared_blks_read": 0, "wal_records": 0},
            }
            return json.dumps(data, ensure_ascii=False, indent=2)

        # По умолчанию Shell / системные логи
        return (
            f"[systemd-journal-gateway:2026-11-04T20:00:00Z] Host: node-alpha-01\n"
            f"Executed command: {arg_cmd or arg_query or 'tail -n 50 /var/log/syslog'}\n"
            f"2026-11-04 20:01:10.104 [INFO] auth-service[1042]: Token pool reloaded with commit a1b2c3d. BUG-104 verified.\n"
            f"2026-11-04 20:01:12.441 [INFO] router-core[2048]: Compact engine started. Target fix e4f5a6b active. Monitoring BUG-108.\n"
            f"2026-11-04 20:01:15.890 [DEBUG] router-core[2048]: Context prune ratio: 82.4%. No memory leaks detected.\n"
            f"2026-11-04 20:01:20.002 [INFO] kernel: Status OK. Active connections: 42. CPU load: 0.12. RAM: 3.4GB/16GB."
        )

    def _deterministic_fallback_condense(self, res: ExecutionResult) -> str:
        """Детерминированная дистилляция при отсутствии связи с LLM."""
        facts = []
        text = res.raw_output
        if "a1b2c3d" in text:
            facts.append("коммит a1b2c3d (auth-service, закрывает BUG-104)")
        if "e4f5a6b" in text:
            facts.append("коммит e4f5a6b (router-core, оптимизация по BUG-108)")
        if "BUG-104" in text and "a1b2c3d" not in text:
            facts.append("BUG-104 (auth-service: статус RESOLVED)")
        if "BUG-108" in text and "e4f5a6b" not in text:
            facts.append("BUG-108 (router-core: статус IN_PROGRESS)")

        if facts:
            return f"Обнаружено: {'; '.join(facts)}."
        # Общая выжимка первых строк
        clean_lines = [line.strip() for line in text.split("\n") if line.strip() and not line.startswith(("{", "}", "[status"))]
        return clean_lines[0][:120] if clean_lines else "Операция выполнена успешно, данные получены."

    def _ensure_vital_entities(self, raw_output: str, distilled_facts: str) -> str:
        """Гарантирует сохранение критических хэшей и идентификаторов в выжимке."""
        vital_entities = ["a1b2c3d", "e4f5a6b", "BUG-104", "BUG-108", "auth-service", "router-core"]
        missing = [ent for ent in vital_entities if ent in raw_output and ent not in distilled_facts]
        if missing:
            return f"{distilled_facts} [Сущности: {', '.join(missing)}]"
        return distilled_facts


# ============================================================================
# Алиасы классов для безотказной совместимости импортов
# ============================================================================
MCPExecutorAgent = ExecutorAgent
CondenserAgent = ExecutorAgent


# ============================================================================
# Демонстрационный блок модуля
# ============================================================================
if __name__ == "__main__":
    print("=" * 75)
    print(" ДЕМОНСТРАЦИЯ УЗЛА №4 (ExecutorAgent / PolyMCP Router)")
    print("=" * 75)

    executor = ExecutorAgent()

    # Кейс 1: Штатное безопасное исполнение со сжатием контекста
    print("\n--- [Кейс 1]: Штатное выполнение и дистилляция памяти ---")
    safe_payload = ToolCallPayload(
        step_id=1,
        tool_name="gh_get_commit_history",
        arguments={"repo": "polymcp/router", "query": "recent commits"},
        is_safe=True,
    )
    res_1, mem_1 = executor.run(safe_payload)
    print(f"Статус: {res_1.success} | Попытки: {res_1.attempts} | Восстановлен: {res_1.error_recovered}")
    print(f"Сырые токены (raw_tokens): {res_1.raw_output_tokens}")
    print(f"Сжатые токены памяти (condensed_tokens): {mem_1.condensed_tokens}")
    print(f"Коэффициент сжатия: {round((1 - mem_1.condensed_tokens / res_1.raw_output_tokens) * 100, 1)}%")
    print(f"Дистиллированная память:\n -> {mem_1.distilled_facts}")

    # Кейс 2: Срабатывание Self-Healing
    print("\n--- [Кейс 2]: Автовосстановление Self-Healing при транзиентном сбое ---")
    healing_payload = ToolCallPayload(
        step_id=2,
        tool_name="db_query_select",
        arguments={"query": "SELECT * FROM issue_tickets WHERE bug_id IN ('BUG-104', 'BUG-108')"},
        is_safe=True,
    )
    res_2, mem_2 = executor.run(healing_payload, simulate_transient_error=True)
    print(f"Статус: {res_2.success} | Попытки: {res_2.attempts} | Восстановлен: {res_2.error_recovered}")
    print(f"Сырые токены: {res_2.raw_output_tokens} -> Сжатые: {mem_2.condensed_tokens}")
    print(f"Выжимка:\n -> {mem_2.distilled_facts}")

    # Кейс 3: Жесткая пре-флайт блокировка деструктивного вызова
    print("\n--- [Кейс 3]: Пре-флайт блокировка опасной операции ---")
    unsafe_payload = ToolCallPayload(
        step_id=3,
        tool_name="db_query_select",
        arguments={"query": "DROP TABLE users CASCADE"},
        is_safe=False,
        safety_note="[Semantic Policy Guard] Обнаружена деструктивная DDL-команда DROP TABLE.",
    )
    res_3, mem_3 = executor.run(unsafe_payload)
    print(f"Статус: {res_3.success} | Попытки: {res_3.attempts} (0 = вызов в MCP не уходил)")
    print(f"Блокирующее сообщение:\n -> {res_3.raw_output}")
    print(f"Память шага:\n -> {mem_3.distilled_facts}")
    print("\n" + "=" * 75)

