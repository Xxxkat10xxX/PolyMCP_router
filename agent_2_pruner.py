import copy
import json
import os
import re
from typing import Any, Dict, List, Set, Tuple
from pydantic import BaseModel, Field

from contracts import PrunedTool, SubTask
from llm_client import call_llm_structured, count_tokens


class _SelectedKeysWrapper(BaseModel):
    """Обертка для Structured Output выбора нужных полей JSON-схемы (совместима с llm_client.py)."""
    keep_properties: List[str] = Field(default_factory=list)


class SchemaPrunerAgent:
    """Agent 2 (Роман Огородников): Гибридный отбор Топ-2 тулов и LLM Schema Pruning."""

    SYSTEM_PROMPT = (
        "Ты — SchemaPrunerAgent в системе PolyMCP Router. "
        "Проанализируй инструкцию подзадачи (SubTask) и список параметров (properties) JSON-схемы инструмента. "
        "Оставь в списке keep_properties ТОЛЬКО те поля, которые реально нужны для выполнения инструкции:\n"
        "- Для задач PostgreSQL (postgresql_mcp) оставляй только ['query'].\n"
        "- Для задач GitHub (github_mcp) оставляй только ['query', 'repo'].\n"
        "- Для задач Shell (shell_mcp) оставляй только ['query', 'command'].\n"
        "Никогда не включай поля optional_param_1 .. optional_param_19."
    )

    # Доменные ключевые слова для Фазы 1 (быстрый отбор по серверам без затрат токенов LLM)
    _DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "postgresql_mcp": [
            "postgres", "postgresql", "sql", "бд", "база", "базе", "базу",
            "таблиц", "table", "select", "insert", "query", "запрос",
            "issues", "баг", "bug", "тикет", "схем", "индекс", "lock",
            "блокировк", "explain", "vacuum", "csv",
        ],
        "github_mcp": [
            "github", "git", "гитхаб", "гит", "коммит", "commit", "pr",
            "pull request", "пулл", "issue", "ишью", "репозитор", "repo",
            "branch", "ветк", "релиз", "release", "workflow", "action",
            "merge", "мерж", "файл", "compare",
        ],
        "shell_mcp": [
            "shell", "bash", "sh", "терминал", "консол", "команд", "command",
            "exec", "лог", "log", "tail", "grep", "процесс", "process",
            "docker", "докер", "контейнер", "диск", "disk", "netstat",
            "порт", "systemctl", "сервис", "env", "архив",
        ],
    }

    # Приоритетные маркеры конкретных инструментов из mock_mcp_registry.json
    _TOOL_HINTS: Dict[str, List[str]] = {
        "gh_get_commit_history": ["коммит", "commit", "вчерашн", "истори", "history", "sha"],
        "gh_compare_commits": ["сравн", "compare", "коммит", "commit", "diff"],
        "gh_search_issues": ["поиск", "search", "ишью", "issue", "открытые"],
        "gh_get_issue_details": ["детали", "подробност", "details", "тикет"],
        "gh_list_pull_requests": ["pr", "pull request", "пулл"],
        "gh_list_workflow_runs": ["workflow", "ci", "actions", "пайплайн"],
        "db_query_select": ["выбрать", "select", "открытые", "баг", "issues", "таблиц", "найти", "сверь"],
        "db_describe_table": ["таблиц", "структур", "describe", "table"],
        "db_query_insert": ["вставить", "insert", "добавить", "записать"],
        "db_explain_plan": ["explain", "план", "медленн"],
        "db_active_locks": ["lock", "блокировк"],
        "shell_tail_log": ["лог", "log", "tail", "последние строки", " системного лога"],
        "shell_grep_pattern": ["grep", "поиск", "лог", "log", "найденным багам", "шаблон"],
        "shell_exec_command": ["выполнить", "команд", "exec", "запустить"],
        "shell_check_process": ["процесс", "process", "воркер"],
        "shell_docker_status": ["docker", "докер", "контейнер"],
    }

    def __init__(self, registry_path: str = "mock_mcp_registry.json", top_k: int = 2) -> None:
        self.registry_path = registry_path
        self.top_k = top_k
        self.registry: List[Dict[str, Any]] = self._load_registry()

    def _load_registry(self) -> List[Dict[str, Any]]:
        """Загружает каталог из 35 JSON-схем mock_mcp_registry.json."""
        if not os.path.exists(self.registry_path):
            raise FileNotFoundError(f"Каталог MCP не найден по пути: {self.registry_path}")
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return set(re.findall(r"[a-zа-яё0-9_]+", text.lower()))

    def _score_tool(self, instruction: str, tool: Dict[str, Any]) -> float:
        """Фаза 1: Расчет релевантности инструмента по ключевым словам и домену сервера."""
        instr_lower = instruction.lower()
        instr_tokens = self._tokenize(instr_lower)

        tool_name = tool.get("tool_name", "")
        server_name = tool.get("server_name", "")
        description = tool.get("description", "")

        score = 0.0

        # 1. Доменный вес сервера (postgresql_mcp / github_mcp / shell_mcp)
        for kw in self._DOMAIN_KEYWORDS.get(server_name, []):
            if kw in instr_lower:
                score += 4.0

        # 2. Точечный вес целевой операции
        for hint in self._TOOL_HINTS.get(tool_name, []):
            if hint in instr_lower:
                score += 6.0

        # 3. Пересечение токенов с именем и описанием инструмента
        tool_tokens = self._tokenize(f"{tool_name} {server_name} {description}")
        score += len(instr_tokens.intersection(tool_tokens)) * 1.5

        return score

    def _retrieve_top_k(self, subtask: SubTask) -> List[Dict[str, Any]]:
        """Фаза 1: Быстрый отбор Топ-2 релевантных инструментов для инструкции."""
        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for idx, tool in enumerate(self.registry):
            score = self._score_tool(subtask.instruction, tool)
            scored.append((score, -idx, tool))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[: self.top_k]]

    def _prune_single_tool(self, subtask: SubTask, tool: Dict[str, Any]) -> PrunedTool:
        """Фаза 2: LLM Schema Pruning + принудительное сохранение полей из required."""
        tool_name = tool["tool_name"]
        server_name = tool["server_name"]
        raw_schema = tool["inputSchema"]

        raw_properties: Dict[str, Any] = raw_schema.get("properties", {})
        required_fields: List[str] = list(raw_schema.get("required", []))

        # Подсчет токенов ДО сжатия
        raw_tokens = count_tokens(raw_schema)

        # Компактный список полей для анализа нейросетью
        properties_brief = {
            k: v.get("description", "") for k, v in raw_properties.items()
        }

        user_prompt = (
            f"Подзадача (step_id={subtask.step_id}): {subtask.instruction}\n"
            f"Инструмент: {tool_name} (сервер: {server_name})\n"
            f"Обязательные поля (required): {required_fields}\n"
            f"Поля схемы (properties): {json.dumps(properties_brief, ensure_ascii=False)}\n"
            "Верни список нужных полей в keep_properties."
        )

        # Вызов gpt-6-luna через единый клиент llm_client.py
        llm_result = call_llm_structured(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=_SelectedKeysWrapper,
        )

        selected_keys = set(llm_result.keep_properties)

        # Отсекаем нерелевантные кросс-доменные поля, если сработал универсальный офлайн-фоллбэк
        if server_name == "postgresql_mcp":
            selected_keys.discard("repo")
            selected_keys.discard("command")
        elif server_name == "github_mcp":
            selected_keys.discard("command")
        elif server_name == "shell_mcp":
            selected_keys.discard("repo")

        # Сборка сжатой схемы с принудительным сохранением всех полей из required
        kept_properties: Dict[str, Any] = {}
        for prop_key, prop_val in raw_properties.items():
            if (prop_key in required_fields or prop_key in selected_keys) and not prop_key.startswith("optional_param_"):
                kept_properties[prop_key] = copy.deepcopy(prop_val)

        pruned_schema: Dict[str, Any] = {
            "type": raw_schema.get("type", "object"),
            "properties": kept_properties,
            "required": [req for req in required_fields if req in kept_properties],
        }

        # Подсчет токенов ПОСЛЕ сжатия
        pruned_tokens = count_tokens(pruned_schema)

        return PrunedTool(
            tool_name=tool_name,
            server_name=server_name,
            pruned_schema=pruned_schema,
            raw_tokens=raw_tokens,
            pruned_tokens=pruned_tokens,
        )

    def run(self, subtask: SubTask) -> List[PrunedTool]:
        """Главная точка входа Agent 2: принимает SubTask и возвращает List[PrunedTool]."""
        top_tools = self._retrieve_top_k(subtask)
        return [self._prune_single_tool(subtask, tool) for tool in top_tools]


if __name__ == "__main__":
    from agent_1_planner import PlannerAgent

    print("=" * 75)
    print("ТЕСТ СВЯЗКИ AGENT 1 (PlannerAgent) -> AGENT 2 (SchemaPrunerAgent)")
    print("=" * 75)

    planner = PlannerAgent()
    pruner = SchemaPrunerAgent(registry_path="mock_mcp_registry.json", top_k=2)

    full_catalog_tokens = count_tokens(pruner.registry)
    test_query = "Сверь вчерашние коммиты в GitHub с открытыми багами в PostgreSQL и проверь логи через Shell"

    dag_subtasks = planner.run(test_query)
    print(f"Запрос: {test_query}")
    print(f"Получено подзадач от Agent 1: {len(dag_subtasks)} | Полный каталог (35 тулов): {full_catalog_tokens} токенов\n")

    total_raw_tokens = 0
    total_pruned_tokens = 0

    for subtask in dag_subtasks:
        pruned_tools = pruner.run(subtask)
        step_raw = sum(t.raw_tokens for t in pruned_tools)
        step_pruned = sum(t.pruned_tokens for t in pruned_tools)
        total_raw_tokens += step_raw
        total_pruned_tokens += step_pruned

        step_reduction = (1 - step_pruned / max(step_raw, 1)) * 100
        print(f"--- Шаг {subtask.step_id}: {subtask.instruction} ---")
        for idx, tool in enumerate(pruned_tools, start=1):
            kept_fields = list(tool.pruned_schema["properties"].keys())
            tool_reduction = (1 - tool.pruned_tokens / max(tool.raw_tokens, 1)) * 100
            print(
                f"  [Топ-{idx}] {tool.tool_name} ({tool.server_name}) | "
                f"Токены: {tool.raw_tokens} -> {tool.pruned_tokens} (-{tool_reduction:.1f}%) | "
                f"Поля: {kept_fields}"
            )
        print(f"  Итог шага {subtask.step_id}: {step_raw} -> {step_pruned} токенов (сжатие схем на {step_reduction:.1f}%)\n")

    avg_schema_reduction = (1 - total_pruned_tokens / max(total_raw_tokens, 1)) * 100
    total_catalog_baseline = full_catalog_tokens * len(dag_subtasks)
    overall_mcp_tax_reduction = (1 - total_pruned_tokens / max(total_catalog_baseline, 1)) * 100

    print("=" * 75)
    print(f"ИТОГО LLM SCHEMA PRUNING (Топ-2 тула): {total_raw_tokens} -> {total_pruned_tokens} токенов (сжатие: {avg_schema_reduction:.1f}%)")
    print(f"ЭКОНОМИЯ ОТНОСИТЕЛЬНО ВСЕХ 35 СХЕМ:    {total_catalog_baseline} -> {total_pruned_tokens} токенов (сжатие: {overall_mcp_tax_reduction:.2f}%)")
    print("=" * 75)