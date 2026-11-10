import os
import pandas as pd
import streamlit as st
from pipeline import PolyMCPPipeline

st.set_page_config(page_title="PolyMCP Router Dashboard", layout="wide")

st.title("PolyMCP Router — Мультиагентный шлюз и оптимизатор контекста MCP")
st.caption("Архитектура из 5 специализированных LLM-агентов (LangGraph / A2A) | Валидация по методологии MASEval, Legroom и AttnCompress")

default_query = "Сверь вчерашние коммиты в GitHub с открытыми багами в PostgreSQL и проверь системные логи через Shell"
query = st.text_input("Введите кросс-доменный запрос (GitHub + PostgreSQL + Shell):", value=default_query)

if st.button("Запустить конвейер из 5 агентов", type="primary"):
    pipe = PolyMCPPipeline(top_k=3)
    trace = pipe.run(query)

    raw_schema_tok = sum(t.raw_tokens for tools in trace.pruned_tools_per_step.values() for t in tools)
    pruned_schema_tok = sum(t.pruned_tokens for tools in trace.pruned_tools_per_step.values() for t in tools)
    schema_saving_pct = (1 - pruned_schema_tok / max(raw_schema_tok, 1)) * 100

    raw_mem_tok = sum(ex.raw_output_tokens for ex in trace.executions.values())
    cond_mem_tok = sum(m.condensed_tokens for m in trace.condensed_memory.values())
    mem_saving_pct = (1 - cond_mem_tok / max(raw_mem_tok, 1)) * 100

    pass_1_steps = sum(1 for ex in trace.executions.values() if ex.success and ex.attempts == 1)
    pass_1_pct = (pass_1_steps / max(len(trace.executions), 1)) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Экономия токенов схем (Legroom)", f"-{schema_saving_pct:.1f}%", f"{raw_schema_tok} → {pruned_schema_tok} tok (Порог ≥ 50%)")
    c2.metric("Сжатие памяти A2A (AttnCompress)", f"-{mem_saving_pct:.1f}%", f"{raw_mem_tok} → {cond_mem_tok} tok")
    c3.metric("Успешность Pass@1 (без ретраев)", f"{pass_1_pct:.1f}%", "Порог КП1 ≥ 85%")

    st.subheader("Трассировка конвейера (5 LLM-агентов)")
    cols = st.columns(5)

    with cols[0]:
        st.markdown("**1. PlannerAgent**\n*(Епифанов Д.)*")
        for st_task in trace.dag_plan:
            st.info(f"Шаг {st_task.step_id} (deps: {st_task.depends_on}):\n{st_task.instruction}")

    with cols[1]:
        st.markdown("**2. SchemaPrunerAgent**\n*(Огородников Р.)*")
        for step_id, tools in trace.pruned_tools_per_step.items():
            names = [t.tool_name for t in tools]
            r_t = sum(t.raw_tokens for t in tools)
            p_t = sum(t.pruned_tokens for t in tools)
            st.success(f"Шаг {step_id}: {names}\nТокены: {r_t} → {p_t}")

    with cols[2]:
        st.markdown("**3. ArgSynthesizerAgent**\n*(Ларютин М.)*")
        for step_id, call in trace.tool_calls.items():
            badge = "SAFE" if call.is_safe else "BLOCKED"
            st.warning(f"Шаг {step_id} [{badge}]: `{call.tool_name}`\nArgs: `{call.arguments}`")

    with cols[3]:
        st.markdown("**4. SelfHealingExecutor**\n*(Чумазин Н.)*")
        for step_id, ex in trace.executions.items():
            st.write(f"Шаг {step_id}: `{ex.tool_name}` | Success: `{ex.success}` | Попыток: `{ex.attempts}` | Healed: `{ex.error_recovered}`")

    with cols[4]:
        st.markdown("**5. CondenserCriticAgent**\n*(Негоденко С.)*")
        for step_id, mem in trace.condensed_memory.items():
            st.write(f"Шаг {step_id} ({mem.condensed_tokens} tok): {mem.distilled_facts}")

    st.success(f"**Итоговый ответ Агента-Критика:** {trace.final_answer}")

st.divider()
st.subheader("Результаты валидации на открытых бенчмарках (MASEval / SWE-bench / AgentBench / ProtocolBench)")
if os.path.exists("benchmark_summary_by_standard.csv"):
    df_sum = pd.read_csv("benchmark_summary_by_standard.csv")
    st.dataframe(df_sum, use_container_width=True)
if os.path.exists("comparison_chart.png"):
    st.image("comparison_chart.png", caption="Сравнение 4 Baselines и выполнение целевых порогов КП1")