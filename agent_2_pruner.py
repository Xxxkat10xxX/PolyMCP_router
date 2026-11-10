import json
import os
import re
from typing import Any, Dict, List, Set, Tuple
from pydantic import BaseModel, Field

from contracts import PrunedTool, SubTask
from llm_client import call_llm_structured, count_tokens
from generate_registry import ensure_mock_registry


class _SelectedKeysWrapper(BaseModel):
    keep_properties: List[str] = Field(default_factory=list)


class SchemaPrunerAgent:
    """
    Agent 2 (Роман Огородников): Гибридный отбор Топ-3 инструментов, LLM Schema Pruning
    и кэширование сжатых схем по домену сервера (Блок 1 спецификации).
    """

    SYSTEM_PROMPT = (
        "Ты — SchemaPrunerAgent в системе PolyMCP Router. "
        "Проанализируй инструкцию подзадачи (SubTask) и список параметров (properties) JSON-схемы инструмента. "
        "Оставь в списке keep_properties только те параметры, которые нужны для вызова:\n"
        "- Для задач PostgreSQL: ['query', 'limit']\n"
        "- Для задач GitHub: ['query', 'repo', 'limit']\n"
        "- Для задач Shell: ['query', 'command', 'limit']\n"
        "Исключи избыточные технические поля (timeout_ms, include_metadata, output_format, sort_order)."
    )

    _DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "postgresql_mcp": [
            "postgres", "postgresql", "sql", "бд", "база", "базе", "базу",
            "таблиц", "table", "select", "insert", "query", "запрос",
            "issues", "баг", "bug", "тикет", "инцидент", "индекс", "lock",
            "блокировк", "explain", "реплик", "wal", "csv", "размер",
        ],
        "github_mcp": [
            "github", "git", "гитхаб", "гит", "коммит", "commit", "pr",
            "pull request", "пулл", "issue", "ишью", "репозитор", "repo",
            "branch", "ветк", "релиз", "release", "workflow", "action",
            "ci/cd", "ci", "файл", "compare", "diff", "sha",
        ],
        "shell_mcp": [
            "shell", "bash", "sh", "терминал", "консол", "команд", "command",
            "exec", "лог", "log", "tail", "grep", "процесс", "process",
            "docker", "докер", "контейнер", "диск", "disk", "воркер",
            "сервис", "память", "cpu", "var/log",
        ],
    }

    _TOOL_HINTS: Dict[str, List[str]] = {
        "gh_get_commit_history": ["коммит", "commit", "вчерашн", "истори", "history", "sha"],
        "gh_compare_commits": ["сравн", "compare", "diff", "между ветками"],
        "gh_search_issues": ["поиск", "search", "ишью", "issue", "открытые задачи", "баг-репорт"],
        "gh_get_issue_details": ["детали", "подробност", "details", "стектрейс"],
        "gh_list_pull_requests": ["pr", "pull request", "пулл", "ревью"],
        "gh_list_workflow_runs": ["workflow", "ci", "actions", "пайплайн", "сборк"],
        "gh_get_file_contents": ["файл", "код", "содержимое", "contents"],
        "db_query_select": ["выбрать", "select", "открытые", "баг", "issues", "инцидент", "найти", "сверь"],
        "db_describe_table": ["структур", "describe", "колонк", "схем"],
        "db_explain_plan": ["explain", "план", "медленн", "индекс"],
        "db_active_locks": ["lock", "блокировк", "транзакц", "pid"],
        "db_replication_lag": ["реплик", "replication", "wal", "отставан"],
        "db_table_size": ["размер", "size", "bloat", "объем таблиц"],
        "db_export_csv": ["экспорт", "csv", "выгруз"],
        "shell_tail_log": ["лог", "log", "tail", "последние строки", "системного лога"],
        "shell_grep_pattern": ["grep", "шаблон", "поиск ошибок в логах", "найденным багам"],
        "shell_docker_status": ["docker", "докер", "контейнер", "healthcheck"],
        "shell_check_process": ["процесс", "process", "воркер", "cpu"],
        "shell_disk_usage": ["диск", "disk", "место", "df", "монтирован"],
        "shell_exec_command": ["выполнить", "команд", "exec", "скрипт"],
    }

    def __init__(self, registry_path: str = "mock_mcp_registry.json", top_k: int = 3) -> None:
        self.registry_path = registry_path
        self.top_k = top_k
        if not os.path.exists(self.registry_path):
            ensure_mock_registry(self.registry_path)
        self.registry: List[Dict[str, Any]] = self._load_registry()
        # Кэш выбранных ключей по серверу (устраняет лишние сетевые вызовы при обрезке Топ-3 тулов)
        self._server_keys_cache: Dict[str, Set[str]] = {}

    def _load_registry(self) -> List[Dict[str, Any]]:
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return set(re.findall(r"[a-zа-яё0-9_]+", text.lower()))

    def _score_tool(self, instruction: str, tool: Dict[str, Any]) -> float:
        instr_lower = instruction.lower()
        instr_tokens = self._tokenize(instr_lower)

        tool_name = tool.get("tool_name", "")
        server_name = tool.get("server_name", "")
        description = tool.get("description", "")

        score = 0.0
        for kw in self._DOMAIN_KEYWORDS.get(server_name, []):
            if kw in instr_lower:
                score += 4.5

        for hint in self._TOOL_HINTS.get(tool_name, []):
            if hint in instr_lower:
                score += 7.0

        tool_tokens = self._tokenize(f"{tool_name} {server_name} {description}")
        score += len(instr_tokens.intersection(tool_tokens)) * 1.5
        return score

    def _retrieve_top_k(self, subtask: SubTask) -> List[Dict[str, Any]]:
        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for idx, tool in enumerate(self.registry):
            score = self._score_tool(subtask.instruction, tool)
            scored.append((score, -idx, tool))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[: self.top_k]]

    def _prune_single_tool(self, subtask: SubTask, tool: Dict[str, Any]) -> PrunedTool:
        tool_name = tool["tool_name"]
        server_name = tool["server_name"]
        raw_schema = tool["inputSchema"]
        raw_properties: Dict[str, Any] = raw_schema.get("properties", {})
        required_fields: List[str] = list(raw_schema.get("required", []))

        full_tool_definition = {
            "tool_name": tool_name,
            "server_name": server_name,
            "description": tool.get("description", ""),
            "inputSchema": raw_schema,
        }
        raw_tokens = count_tokens(full_tool_definition)

        # Используем кэш ключей по домену сервера, чтобы не делать 3 одинаковых HTTP-запроса на одном шаге
        if server_name in self._server_keys_cache:
            selected_keys = set(self._server_keys_cache[server_name])
        else:
            properties_brief = {k: v.get("description", "") for k, v in raw_properties.items()}
            user_prompt = (
                f"Подзадача (step_id={subtask.step_id}): {subtask.instruction}\n"
                f"Инструмент: {tool_name} (сервер: {server_name})\n"
                f"Обязательные поля (required): {required_fields}\n"
                f"Поля схемы (properties): {json.dumps(properties_brief, ensure_ascii=False)}\n"
                "Верни список нужных полей в keep_properties."
            )
            llm_result = call_llm_structured(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_schema=_SelectedKeysWrapper,
            )
            selected_keys = set(llm_result.keep_properties)
            self._server_keys_cache[server_name] = set(selected_keys)

        selected_keys.add("limit")
        if server_name == "postgresql_mcp":
            selected_keys.discard("repo")
            selected_keys.discard("command")
        elif server_name == "github_mcp":
            selected_keys.discard("command")
            selected_keys.add("repo")
        elif server_name == "shell_mcp":
            selected_keys.discard("repo")
            selected_keys.add("command")

        kept_properties: Dict[str, Any] = {}
        for prop_key, prop_val in raw_properties.items():
            if prop_key in required_fields or prop_key in selected_keys:
                short_desc = prop_val.get("description", "").split("(")[0].strip()
                kept_properties[prop_key] = {
                    "type": prop_val.get("type", "string"),
                    "description": short_desc[:55],
                }

        pruned_schema: Dict[str, Any] = {
            "type": raw_schema.get("type", "object"),
            "properties": kept_properties,
            "required": [req for req in required_fields if req in kept_properties],
        }

        pruned_tool_definition = {
            "tool_name": tool_name,
            "server_name": server_name,
            "inputSchema": pruned_schema,
        }
        pruned_tokens = count_tokens(pruned_tool_definition)

        return PrunedTool(
            tool_name=tool_name,
            server_name=server_name,
            pruned_schema=pruned_schema,
            raw_tokens=raw_tokens,
            pruned_tokens=pruned_tokens,
        )

    def run(self, subtask: SubTask) -> List[PrunedTool]:
        top_tools = self._retrieve_top_k(subtask)
        return [self._prune_single_tool(subtask, tool) for tool in top_tools]


if __name__ == "__main__":
    from agent_1_planner import PlannerAgent

    planner = PlannerAgent()
    pruner = SchemaPrunerAgent(registry_path="mock_mcp_registry.json", top_k=3)
    dag = planner.run("Сверь вчерашние коммиты в GitHub с открытыми багами в PostgreSQL и проверь логи через Shell")

    for task in dag:
        tools = pruner.run(task)
        raw_sum = sum(t.raw_tokens for t in tools)
        pruned_sum = sum(t.pruned_tokens for t in tools)
        saving_pct = (1 - pruned_sum / max(raw_sum, 1)) * 100
        print(f"Шаг {task.step_id}: {[t.tool_name for t in tools]} | Сжатие схем (Legroom): {raw_sum} -> {pruned_sum} ток (-{saving_pct:.1f}%)")