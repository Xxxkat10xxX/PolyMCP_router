"""
app_ui.py — Интерактивный Enterprise-дашборд трассировки PolyMCP Router на Streamlit.
Визуализирует прохождение запроса через 5 карточек LLM-агентов (Слайд 3 защиты КП2).
Разработчик: Негоденко Савелий Кириллович (Б26-504).
"""

import json
from typing import Optional
import streamlit as st

from contracts import (
    SubTask,
    PrunedTool,
    ToolCallPayload,
    ExecutionResult,
    CondensedStepState,
    PipelineTrace,
)
from llm_client import count_tokens


def build_demo_pipeline_trace() -> PipelineTrace:
    """
    Формирует эталонный объект PipelineTrace для мгновенной демонстрации на защите КП2.
    Отражает параллельные ветки DAG (шаги 1 и 2), обрезку схем (~92%), Policy Guard,
    Self-Healing (исправление user_id -> account_uuid на 2-й попытке) и дистилляцию памяти.
    """
    raw_pg_log = (
        "[PG-DUMP] SELECT * FROM bugs WHERE status='open' AND account_uuid='dev_42';\n"
        + "\n".join(
            [
                f"DEBUG: buffer_pool_page_{i} scanned (latency=0.04ms, locks=none, tx_state=clean)"
                for i in range(1, 28)
            ]
        )
        + "\nROW 1: id=104 | severity=HIGH | status=open | commit=a1b2c1"
        + "\nROW 2: id=109 | severity=CRITICAL | status=open | commit=NULL (unpatched)"
    )

    raw_shell_log = (
        "[SHELL-MCP] tail -n 100 /var/log/polymcp/worker.log | grep -E 'ERROR|CRITICAL'\n"
        + "\n".join(
            [
                f"2026-11-05 08:01:{i:02d} INFO [healthcheck] node-0{i % 3 + 1}.internal heartbeat OK (cpu=14%, mem=42%)"
                for i in range(1, 32)
            ]
        )
        + "\n2026-11-05 08:01:55 CRITICAL [worker-03] TimeoutError on ticket #109: lock wait exceeded 30s."
    )

    raw_gh_log = (
        '{"repository": "org/core-backend", "branch": "main", "commits": ['
        '{"sha": "a1b2c1", "author": "dev_42", "date": "2026-11-04T19:40:00Z", '
        '"message": "fix(auth): patch token leak (#104)", "files_changed": 4, '
        '"ci_status": "passed", "reviewers": ["lead_01", "sec_bot"]}]}'
    )

    return PipelineTrace(
        user_query=(
            "Сверь вчерашние коммиты в GitHub с открытыми критическими багами в PostgreSQL, "
            "и если по тикетам нет фиксов — проверь последние ошибки воркера через Shell MCP."
        ),
        dag_plan=[
            SubTask(
                step_id=1,
                instruction="Получить историю вчерашних коммитов из репозитория через GitHub MCP",
                depends_on=[],
            ),
            SubTask(
                step_id=2,
                instruction="Выбрать открытые критические баги из таблицы bugs в PostgreSQL MCP",
                depends_on=[],
            ),
            SubTask(
                step_id=3,
                instruction="Проверить системный лог воркера через Shell MCP по незакрытым критическим багам",
                depends_on=[1, 2],
            ),
        ],
        pruned_tools_per_step={
            1: [
                PrunedTool(
                    tool_name="gh_get_commit_history",
                    server_name="github_mcp",
                    pruned_schema={
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string", "description": "Target repository owner/repo"},
                            "since": {"type": "string", "description": "ISO-8601 start timestamp"},
                        },
                        "required": ["repo"],
                    },
                    raw_tokens=3240,
                    pruned_tokens=210,
                )
            ],
            2: [
                PrunedTool(
                    tool_name="db_query_select",
                    server_name="postgresql_mcp",
                    pruned_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Read-only SQL SELECT query"},
                            "account_uuid": {"type": "string", "description": "Developer account UUID"},
                        },
                        "required": ["query"],
                    },
                    raw_tokens=3480,
                    pruned_tokens=225,
                )
            ],
            3: [
                PrunedTool(
                    tool_name="shell_tail_log",
                    server_name="shell_mcp",
                    pruned_schema={
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Isolated log inspection command"},
                            "limit": {"type": "integer", "description": "Max lines to return"},
                        },
                        "required": ["command"],
                    },
                    raw_tokens=3190,
                    pruned_tokens=195,
                )
            ],
        },
        tool_calls={
            1: ToolCallPayload(
                step_id=1,
                tool_name="gh_get_commit_history",
                arguments={"repo": "org/core-backend", "since": "2026-11-04T00:00:00Z"},
                is_safe=True,
                safety_note="Read-only запрос к GitHub API; относительная дата 'вчера' переведена в ISO-8601.",
            ),
            2: ToolCallPayload(
                step_id=2,
                tool_name="db_query_select",
                arguments={
                    "query": "SELECT id, severity, status, commit_ref FROM bugs WHERE status='open';",
                    "account_uuid": "dev_42",
                },
                is_safe=True,
                safety_note="Проверено Semantic Policy Guard: деструктивных операторов (DROP/DELETE) не обнаружено.",
            ),
            3: ToolCallPayload(
                step_id=3,
                tool_name="shell_tail_log",
                arguments={
                    "command": "tail -n 100 /var/log/polymcp/worker.log | grep 'ticket=#109'",
                    "limit": 50,
                },
                is_safe=True,
                safety_note="Безопасная команда чтения логов в изолированной песочнице; инъекции и rm -rf отсутствуют.",
            ),
        },
        executions={
            1: ExecutionResult(
                step_id=1,
                tool_name="gh_get_commit_history",
                success=True,
                attempts=1,
                error_recovered=False,
                raw_output=raw_gh_log,
                raw_output_tokens=count_tokens(raw_gh_log),
            ),
            2: ExecutionResult(
                step_id=2,
                tool_name="db_query_select",
                success=True,
                attempts=2,
                error_recovered=True,
                raw_output=raw_pg_log,
                raw_output_tokens=count_tokens(raw_pg_log),
            ),
            3: ExecutionResult(
                step_id=3,
                tool_name="shell_tail_log",
                success=True,
                attempts=1,
                error_recovered=False,
                raw_output=raw_shell_log,
                raw_output_tokens=count_tokens(raw_shell_log),
            ),
        },
        condensed_memory={
            1: CondensedStepState(
                step_id=1,
                distilled_facts="Коммит вчера: SHA a1b2c1 (автор dev_42), закрывает только баг #104.",
                condensed_tokens=24,
            ),
            2: CondensedStepState(
                step_id=2,
                distilled_facts="В БД открыты 2 бага: #104 (HIGH, патч a1b2c1) и #109 (CRITICAL, без патча).",
                condensed_tokens=29,
            ),
            3: CondensedStepState(
                step_id=3,
                distilled_facts="В /var/log/polymcp/worker.log критический сбой TimeoutError (30s) по тикету #109.",
                condensed_tokens=26,
            ),
        },
        final_answer=(
            "Сверка завершена успешно:\n"
            "1. В GitHub найден вчерашний коммит SHA `a1b2c1`, который закрывает только тикет #104 (утечка памяти OAuth).\n"
            "2. В PostgreSQL обнаружен открытый критический баг **#109** (`Queue worker deadlock`), по которому нет фиксов.\n"
            "3. Проверка через Shell MCP в `/var/log/polymcp/worker.log` подтвердила активную ошибку `TimeoutError (lock wait > 30s)` на шарде `worker-03` по тикету #109."
        ),
        is_complete=True,
    )


def inject_enterprise_css() -> None:
    """Строгие стили Enterprise-интерфейса для наглядной презентации на защите."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1380px;
        }
        .agent-header {
            background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
            color: #f8fafc;
            padding: 0.75rem 1.1rem;
            border-radius: 8px;
            border-left: 5px solid #3b82f6;
            margin-bottom: 0.75rem;
            font-weight: 600;
            font-size: 1.05rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .agent-role-tag {
            background-color: #334155;
            color: #93c5fd;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-family: monospace;
        }
        .badge-safe {
            background-color: #065f46;
            color: #ecfdf5;
            padding: 0.35rem 0.75rem;
            border-radius: 6px;
            font-weight: 700;
            border: 1px solid #10b981;
            display: inline-block;
        }
        .badge-unsafe {
            background-color: #7f1d1d;
            color: #fef2f2;
            padding: 0.35rem 0.75rem;
            border-radius: 6px;
            font-weight: 700;
            border: 1px solid #ef4444;
            display: inline-block;
        }
        .badge-healed {
            background-color: #78350f;
            color: #fef3c7;
            padding: 0.28rem 0.65rem;
            border-radius: 6px;
            font-weight: 600;
            border: 1px solid #f59e0b;
            display: inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_trace_from_sidebar() -> PipelineTrace:
    """Боковая панель выбора источника трассировки (Демо / JSON / Живой прогон)."""
    st.sidebar.markdown("## ⚙️ PolyMCP Router")
    st.sidebar.caption("Интеллектуальный шлюз и мультиагентная среда оптимизации MCP")

    mode = st.sidebar.radio(
        "Источник трассировки PipelineTrace:",
        options=[
            "Эталонный трейс (Демо КП2)",
            "Загрузить JSON-файл трассы",
            "Живой вызов PolyMCPPipeline",
        ],
        index=0,
    )

    trace: Optional[PipelineTrace] = None

    if mode == "Загрузить JSON-файл трассы":
        uploaded = st.sidebar.file_uploader("Выберите файл PipelineTrace (.json)", type=["json"])
        if uploaded is not None:
            try:
                raw_data = json.load(uploaded)
                trace = PipelineTrace.model_validate(raw_data)
                st.sidebar.success("Трасса успешно валидирована через contracts.py")
            except Exception as exc:
                st.sidebar.error(f"Ошибка валидации контракта: {exc}")

    elif mode == "Живой вызов PolyMCPPipeline":
        default_q = (
            "Сверь вчерашние коммиты в GitHub с открытыми багами в PostgreSQL "
            "и проверь логи через Shell"
        )
        custom_query = st.sidebar.text_area("Пользовательский запрос:", value=default_q, height=95)
        if st.sidebar.button("🚀 Запустить конвейер 5 агентов", type="primary", use_container_width=True):
            try:
                from pipeline import PolyMCPPipeline  # type: ignore

                with st.spinner("Выполнение DAG через 5 агентов PolyMCP..."):
                    pipe = PolyMCPPipeline()
                    trace = pipe.run(custom_query)
                    st.session_state["active_trace"] = trace
            except Exception as exc:
                st.sidebar.warning(
                    f"Модуль pipeline.py не найден или вернул ошибку ({exc}). "
                    "Загружен эталонный демо-трейс."
                )
                trace = build_demo_pipeline_trace()

    if trace is None:
        trace = st.session_state.get("active_trace") or build_demo_pipeline_trace()

    # Возможность симулировать блокировку безопасности на Карточке 3 для показа комиссии
    simulate_attack = st.sidebar.checkbox("🔍 Демо Policy Guard: симулировать DROP TABLE", value=False)
    if simulate_attack and 2 in trace.tool_calls:
        trace = trace.model_copy(deep=True)
        trace.tool_calls[2] = ToolCallPayload(
            step_id=2,
            tool_name="db_query_select",
            arguments={"query": "DROP TABLE bugs CASCADE; --", "account_uuid": "dev_42"},
            is_safe=False,
            safety_note="CRITICAL BLOCK: Обнаружена деструктивная SQL-инъекция (DROP TABLE). Вызов MCP заблокирован до сети.",
        )

    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="📥 Экспортировать PipelineTrace (.json)",
        data=trace.model_dump_json(indent=2),
        file_name="polymcp_pipeline_trace.json",
        mime="application/json",
        use_container_width=True,
    )
    return trace


def render_dashboard(trace: PipelineTrace) -> None:
    """Рендерит верхнюю сводку метрик и 5 карточек агентов по контракту PipelineTrace."""
    inject_enterprise_css()

    # Расчет сквозных метрик компрессии контекста
    total_raw_schema = sum(
        t.raw_tokens for tools in trace.pruned_tools_per_step.values() for t in tools
    )
    total_pruned_schema = sum(
        t.pruned_tokens for tools in trace.pruned_tools_per_step.values() for t in tools
    )
    schema_reduction_pct = (
        (1.0 - total_pruned_schema / max(total_raw_schema, 1)) * 100
        if total_raw_schema > 0
        else 0.0
    )

    total_raw_output = sum(ex.raw_output_tokens for ex in trace.executions.values())
    total_condensed_mem = sum(m.condensed_tokens for m in trace.condensed_memory.values())
    memory_reduction_pct = (
        (1.0 - total_condensed_mem / max(total_raw_output, 1)) * 100
        if total_raw_output > 0
        else 0.0
    )

    baseline_total = total_raw_schema + total_raw_output
    polymcp_total = total_pruned_schema + total_condensed_mem
    total_savings_pct = (
        (1.0 - polymcp_total / max(baseline_total, 1)) * 100 if baseline_total > 0 else 0.0
    )

    healed_steps = sum(1 for ex in trace.executions.values() if ex.error_recovered)
    successful_steps = sum(1 for ex in trace.executions.values() if ex.success)
    total_exec_steps = max(len(trace.executions), 1)

    # Заголовок и KPI-панель
    st.markdown("## 🛡️ PolyMCP Router — Трассировка конвейера 5 LLM-агентов")
    st.info(f"**Входной запрос (`user_query`):** {trace.user_query}")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(
        "Сквозная экономия токенов",
        f"{total_savings_pct:.1f}%",
        f"{baseline_total} → {polymcp_total} ток. (Порог КП1 ≥ 50%)",
    )
    kpi2.metric(
        "Сжатие JSON-схем (Agent 2)",
        f"{schema_reduction_pct:.1f}%",
        f"{total_raw_schema} → {total_pruned_schema} ток.",
    )
    kpi3.metric(
        "Дистилляция памяти A2A (Agent 5)",
        f"{memory_reduction_pct:.1f}%",
        f"{total_raw_output} → {total_condensed_mem} ток.",
    )
    kpi4.metric(
        "Успешность и Self-Healing (Agent 4)",
        f"{successful_steps}/{total_exec_steps} шагов",
        f"Восстановлено после сбоев API: {healed_steps}",
    )

    st.markdown("---")

    # =========================================================================
    # КАРТОЧКА 1: PlannerAgent (Декомпозиция в DAG)
    # =========================================================================
    st.markdown(
        '<div class="agent-header">'
        '<span>Карточка 1 — PlannerAgent (Семантическая декомпозиция и построение DAG)</span>'
        '<span class="agent-role-tag">agent_1_planner.py | List[SubTask]</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        if not trace.dag_plan:
            st.warning("План подзадач `dag_plan` пуст.")
        else:
            cols = st.columns(len(trace.dag_plan))
            for idx, subtask in enumerate(trace.dag_plan):
                with cols[idx]:
                    dep_str = (
                        "⚡ Параллельный старт (зависимостей нет)"
                        if not subtask.depends_on
                        else f"🔗 Ожидает шаги: {subtask.depends_on}"
                    )
                    st.markdown(f"**Шаг `step_id = {subtask.step_id}`**")
                    st.caption(dep_str)
                    st.write(subtask.instruction)

    # =========================================================================
    # КАРТОЧКА 2: SchemaPrunerAgent (Hybrid Retrieval + LLM Schema Pruning)
    # =========================================================================
    st.markdown(
        '<div class="agent-header">'
        '<span>Карточка 2 — SchemaPrunerAgent (Отбор инструментов и динамическая обрезка JSON-схем)</span>'
        '<span class="agent-role-tag">agent_2_pruner.py | Dict[int, List[PrunedTool]]</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        c_left, c_right = st.columns([1, 2.2])
        with c_left:
            st.markdown("**Счетчик компрессии схем (MCP Tax):**")
            st.metric(
                label="Сырой каталог vs Pruned Schema",
                value=f"{total_pruned_schema} токенов",
                delta=f"-{schema_reduction_pct:.1f}% (было {total_raw_schema} ток.)",
            )
            st.progress(
                min(max(int(schema_reduction_pct), 0), 100),
                text=f"Удалено {schema_reduction_pct:.1f}% избыточных опциональных полей",
            )
        with c_right:
            for step_id, tools in sorted(trace.pruned_tools_per_step.items()):
                for tool in tools:
                    step_saving = (
                        (1.0 - tool.pruned_tokens / max(tool.raw_tokens, 1)) * 100
                        if tool.raw_tokens > 0
                        else 0.0
                    )
                    with st.expander(
                        f"Шаг #{step_id} | Сервер: `{tool.server_name}` → Инструмент: `{tool.tool_name}` "
                        f"({tool.raw_tokens} → {tool.pruned_tokens} ток., сжатие {step_saving:.1f}%)",
                        expanded=(step_id == 1),
                    ):
                        st.json(tool.pruned_schema)

    # =========================================================================
    # КАРТОЧКА 3: ArgSynthesizerAgent (Синтез аргументов + Semantic Policy Guard)
    # =========================================================================
    st.markdown(
        '<div class="agent-header">'
        '<span>Карточка 3 — ArgSynthesizerAgent (Синтез аргументов и пре-флайт Semantic Policy Guard)</span>'
        '<span class="agent-role-tag">agent_3_synthesizer.py | Dict[int, ToolCallPayload]</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        call_cols = st.columns(max(len(trace.tool_calls), 1))
        for idx, (step_id, payload) in enumerate(sorted(trace.tool_calls.items())):
            with call_cols[idx]:
                st.markdown(f"**Шаг #{step_id}: `{payload.tool_name}`**")
                if payload.is_safe:
                    st.markdown(
                        '<div class="badge-safe">✅ IS_SAFE = TRUE (Вызов разрешен)</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="badge-unsafe">🛑 IS_SAFE = FALSE (Заблокировано Policy Guard)</div>',
                        unsafe_allow_html=True,
                    )
                st.caption(f"**Примечание аудитора (`safety_note`):** {payload.safety_note}")
                st.json(payload.arguments)

    # =========================================================================
    # КАРТОЧКА 4: SelfHealingExecutorAgent (Вызов MCP + Цикл Reflexion Loop)
    # =========================================================================
    st.markdown(
        '<div class="agent-header">'
        '<span>Карточка 4 — SelfHealingExecutorAgent (Исполнение и цикл самовосстановления при сбоях API)</span>'
        '<span class="agent-role-tag">agent_4_executor.py | Dict[int, ExecutionResult]</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        exec_cols = st.columns(max(len(trace.executions), 1))
        for idx, (step_id, ex) in enumerate(sorted(trace.executions.items())):
            with exec_cols[idx]:
                status_icon = "✅ УСПЕШНО" if ex.success else "❌ СБОЙ"
                st.markdown(f"**Шаг #{step_id}: `{ex.tool_name}` — {status_icon}**")
                st.write(f"• Количество попыток (`attempts`): **`{ex.attempts} / 3`**")
                if ex.error_recovered:
                    st.markdown(
                        '<div class="badge-healed">🔧 Self-Healing: error_recovered = True '
                        "(Аргументы скорректированы по трейсу ошибки API)</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("• `error_recovered = False` (Выполнено с 1-й попытки без ошибок API)")

                with st.expander(f"Сырой ответ сервера ({ex.raw_output_tokens} токенов)"):
                    st.code(ex.raw_output, language="text")

    # =========================================================================
    # КАРТОЧКА 5: CondenserCriticAgent (Дистилляция памяти A2A и Финальный вердикт)
    # =========================================================================
    st.markdown(
        '<div class="agent-header">'
        '<span>Карточка 5 — CondenserCriticAgent (Дистилляция памяти A2A и финальная верификация)</span>'
        '<span class="agent-role-tag">agent_5_condenser.py | CondensedStepState & PipelineTrace</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        col_mem, col_verdict = st.columns([1.35, 1.15])

        with col_mem:
            st.markdown("#### 🗜️ Сжатая межшаговая память A2A (`condensed_memory`)")
            for step_id, mem in sorted(trace.condensed_memory.items()):
                raw_step_tok = (
                    trace.executions[step_id].raw_output_tokens
                    if step_id in trace.executions
                    else mem.condensed_tokens
                )
                step_comp = (
                    (1.0 - mem.condensed_tokens / max(raw_step_tok, 1)) * 100
                    if raw_step_tok > 0
                    else 0.0
                )
                st.markdown(
                    f"**Шаг #{step_id}** — сжатие ответа: **`{raw_step_tok}` → `{mem.condensed_tokens}` ток. "
                    f"(-{step_comp:.1f}%)**"
                )
                st.success(mem.distilled_facts)

        with col_verdict:
            st.markdown("#### ⚖️ Финальный вердикт Агента-Критика")
            if trace.is_complete:
                st.markdown(
                    '<div class="badge-safe">✅ IS_COMPLETE = TRUE (Все цели запроса подтверждены)</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="badge-unsafe">🔄 IS_COMPLETE = FALSE (Требуется Re-plan в Agent 1)</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("**Итоговый ответ (`final_answer`):**")
            st.markdown(trace.final_answer)


def main() -> None:
    st.set_page_config(
        page_title="PolyMCP Router — Trace Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    trace = load_trace_from_sidebar()
    render_dashboard(trace)


if __name__ == "__main__":
    main()