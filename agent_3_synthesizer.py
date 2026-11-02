"""
agent_3_synthesizer.py — Узел №3 конвейера PolyMCP Router (ArgSynthesizerAgent).

Назначение модуля:
1. Контролируемое чтение памяти A2A (Agent-to-Agent): извлекает из глобального
   словаря `memory: Dict[int, CondensedStepState]` строго те состояния, чьи `step_id`
   указаны в `subtask.depends_on`, защищая контекст от зашумления побочными ветками DAG.
2. Семантический синтез аргументов (`arguments`) через `call_llm_structured`:
   преобразует инструкцию подзадачи на естественном языке, относительные даты
   («вчерашние коммиты», «сегодня») и дистиллированные факты из `depends_on`
   в строго типизированные параметры целевого сжатого инструмента (`pruned_tools[0]`).
3. Пре-флайт защита (`Semantic Policy Guard`): комбинирует инструкции в `SYSTEM_PROMPT`,
   проброс маркеров совместимости с `_smart_offline_fallback` (`llm_client.py`)
   и детерминированную пост-проверку `_post_validate_safety` для блокировки:
   - деструктивных SQL-запросов (`DROP TABLE`, `TRUNCATE`, `DELETE FROM`, `ALTER ... DROP`);
   - разрушительных Shell/OS и Git-команд (`rm -rf`, `mkfs`, `shutdown`, `git push --force`);
   - попыток эксфильтрации секретов, токенов, паролей и PII между MCP-инструментами через память A2A.
4. Гарантия целостности контракта `ToolCallPayload`: синхронизация `step_id`,
   валидация `tool_name` по списку `pruned_tools` и строгая фильтрация/нормализация
   ключей `arguments` по `pruned_schema["properties"]`.

Автор: Максим Ларютин (Участник №3)
"""

from __future__ import annotations

import datetime
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from contracts import CondensedStepState, PrunedTool, SubTask, ToolCallPayload
from llm_client import call_llm_structured, count_tokens


SYSTEM_PROMPT = """Ты — ArgSynthesizerAgent (Узел №3) в мультиагентном конвейере PolyMCP Router.
Твоя задача — синтезировать строго типизированный вызов MCP-инструмента в формате `ToolCallPayload`
на основе текущей подзадачи (`SubTask`), списка сжатых схем инструментов (`PrunedTool`)
и отфильтрованного контекста памяти A2A (`CondensedStepState`) от родительских шагов графа (`depends_on`).

ПРАВИЛА СИНТЕЗА АРГУМЕНТОВ:
1. Используй приоритетный инструмент `target_tool` (первый элемент `pruned_tools[0]`), если он подходит
   под задачу, либо выбери другой инструмент строго из переданного списка `pruned_tools`.
2. Сформируй словарь `arguments`, содержащий ТОЛЬКО ключи, разрешенные в `pruned_schema["properties"]`
   выбранного инструмента:
   - Для `postgresql_mcp`: только ключ `["query"]` — валидный, безопасный SQL-запрос (`SELECT ...`).
   - Для `github_mcp`: только ключи `["query", "repo"]` — строка поискового запроса GitHub и имя репозитория (`owner/repo`).
   - Для `shell_mcp`: только ключи `["query", "command"]` — поисковый фильтр/паттерн и безопасная read-only Shell-команда (`grep`, `tail`, `cat`, `journalctl` и т.д.).
3. Разрешай относительные временные выражения («вчерашние коммиты», «сегодня», «за сутки») относительно
   переданной в промпте текущей даты (`current_date` / `yesterday_date`) и обязательно учитывай
   факты из зависимых шагов `depends_on` (ID багов, хэши коммитов, имена сервисов и таблиц).

ПРАВИЛА БЕЗОПАСНОСТИ (SEMANTIC POLICY GUARD — PRE-FLIGHT ПРОВЕРКА):
Ты обязан блокировать любые опасные операции ДО обращения к MCP-серверу (`is_safe = False`):
1. Деструктивные SQL-операции: `DROP TABLE`, `DROP DATABASE`, `TRUNCATE`, `DELETE FROM`,
   `ALTER TABLE ... DROP`, `GRANT ALL`, инъекции со стекингом запросов (`; DROP ...`).
2. Деструктивные Shell/OS команды: `rm -rf`, `rm -fr`, `mkfs`, `dd if=`, `shutdown`, `reboot`,
   `poweroff`, `chmod -R 777 /`, форк-бомбы, перезапись системных устройств, `curl/wget | sh`.
3. Деструктивные Git-операции: `git push --force`, `git push -f`, удаление веток (`git branch -D main/master`),
   `git reset --hard`.
4. Межинструментальная утечка секретов и PII через память A2A: попытки передать приватные ключи,
   пароли (`password`, `passwd`, `secret`, `api_key`, `token`, `.env`, `/etc/shadow`, `id_rsa`)
   или персональные данные из одного MCP-сервера во внешний инструмент (например, публикация
   паролей БД в GitHub или отправка токенов через Shell).

Если обнаружена любая угроза:
- Установи `is_safe = False`.
- В `safety_note` подробно опиши причину блокировки (какое правило Semantic Policy Guard сработало).
Если операция полностью безопасна (read-only / аналитика без утечки секретов):
- Установи `is_safe = True`.
- В `safety_note` кратко укажи, что pre-flight проверка безопасности пройдена успешно.
"""


class ArgSynthesizerAgent:
    """
    Агент семантического синтеза аргументов и пре-флайт валидации безопасности (Узел №3).
    """

    # Регулярные выражения для детерминированного перехвата угроз (Semantic Policy Guard)
    _DESTRUCTIVE_SQL_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
        (
            re.compile(r"\bDROP\s+(?:TABLE|DATABASE|SCHEMA|INDEX|VIEW)\b", re.IGNORECASE),
            "Деструктивная SQL-операция удаления структуры данных (DROP TABLE/DATABASE/SCHEMA)",
        ),
        (
            re.compile(r"\bTRUNCATE\s+(?:TABLE\s+)?[a-zA-Z0-9_.\"]+", re.IGNORECASE),
            "Деструктивная SQL-операция полной очистки таблицы (TRUNCATE)",
        ),
        (
            re.compile(r"\bDELETE\s+FROM\b", re.IGNORECASE),
            "Деструктивная SQL-операция удаления записей (DELETE FROM)",
        ),
        (
            re.compile(r"\bALTER\s+TABLE\s+\S+\s+DROP\b", re.IGNORECASE),
            "Деструктивная модификация схемы БД (ALTER TABLE ... DROP)",
        ),
        (
            re.compile(r";\s*(?:DROP|DELETE|TRUNCATE|ALTER|UPDATE)\b", re.IGNORECASE),
            "Подозрение на SQL-инъекцию со стекингом деструктивных команд",
        ),
    ]

    _DESTRUCTIVE_SHELL_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
        (
            re.compile(r"\brm\s+(?:-[a-zA-Z]*[rf][a-zA-Z]*\s+|--recursive|--force)", re.IGNORECASE),
            "Деструктивная Shell-команда рекурсивного/принудительного удаления файлов (rm -rf)",
        ),
        (
            re.compile(r"\b(?:mkfs(?:\.[a-z0-9]+)?|fdisk|parted|wipefs)\b", re.IGNORECASE),
            "Разрушительная операция форматирования или разметки диска (mkfs/fdisk/wipefs)",
        ),
        (
            re.compile(r"\bdd\s+if=.*?\bof=/dev/", re.IGNORECASE),
            "Опасная низкоуровневая перезапись блочного устройства (dd of=/dev/...)",
        ),
        (
            re.compile(r"\b(?:shutdown|reboot|poweroff|halt|init\s+[06])\b", re.IGNORECASE),
            "Команда остановки или перезагрузки хост-системы (shutdown/reboot)",
        ),
        (
            re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.IGNORECASE),
            "Обнаружена fork-бомба в аргументах Shell",
        ),
        (
            re.compile(r"\b(?:curl|wget)\b.*?\bsudo\b|\b(?:curl|wget)\b.*?\|\s*(?:ba)?sh\b", re.IGNORECASE),
            "Опасное выполнение удаленного скрипта через пайп (curl/wget | sh)",
        ),
        (
            re.compile(r"\bchmod\s+(?:-R\s+)?777\s+/(?:etc|var|usr|bin|root|$|\s)", re.IGNORECASE),
            "Небезопасное изменение прав доступа к системным директориям (chmod 777)",
        ),
    ]

    _DESTRUCTIVE_GIT_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
        (
            re.compile(r"\bgit\s+push\s+(?:--force(?:-with-lease)?|-f)\b", re.IGNORECASE),
            "Деструктивная Git-операция принудительной перезаписи истории (git push --force)",
        ),
        (
            re.compile(r"\bgit\s+branch\s+-D\b|\bgit\s+push\s+\S+\s+--delete\b", re.IGNORECASE),
            "Принудительное удаление ветки репозитория (git branch -D / push --delete)",
        ),
        (
            re.compile(r"\bgit\s+reset\s+--hard\b|\bgit\s+clean\s+-[a-zA-Z]*f", re.IGNORECASE),
            "Необратимый сброс рабочего дерева Git (git reset --hard / git clean -f)",
        ),
    ]

    _SECRET_EXFILTRATION_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
        (
            re.compile(
                r"(?:/etc/shadow|/etc/passwd|\.env\b|id_rsa|PRIVATE\s+KEY|pg_shadow|"
                r"AITUNNEL_API_KEY|AWS_SECRET_ACCESS_KEY|GITHUB_TOKEN|password_hash)",
                re.IGNORECASE,
            ),
            "Попытка доступа к системным секретам, приватным ключам или хэшам паролей",
        ),
        (
            re.compile(
                r"\b(?:exfiltrate|утечк[аиу]|слей\s+парол|отправь\s+(?:пароль|токен|секрет|ключ)|"
                r"передай\s+(?:пароль|токен|секрет|ключ)|опубликуй\s+(?:пароль|токен|секрет|pii))\b",
                re.IGNORECASE,
            ),
            "Попытка эксфильтрации секретов, токенов или чувствительных данных (PII) между MCP-инструментами",
        ),
        (
            re.compile(
                r"(?:sk-[a-zA-Z0-9]{16,}|ghp_[a-zA-Z0-9]{20,}|eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,})",
            ),
            "Обнаружен необработанный приватный API-токен или JWT-секрет в аргументах вызова",
        ),
    ]

    # Стандартные целевые ключи по спецификации Agent 2 (SchemaPrunerAgent)
    _SERVER_DEFAULT_KEYS: Dict[str, List[str]] = {
        "postgresql_mcp": ["query"],
        "github_mcp": ["query", "repo"],
        "shell_mcp": ["query", "command"],
    }

    def __init__(
        self,
        model_name: str = "gpt-6-luna",
        default_repo: str = "polymcp/router",
    ) -> None:
        self.model_name = model_name
        self.default_repo = default_repo

    def run(
        self,
        subtask: SubTask,
        pruned_tools: List[PrunedTool],
        memory: Optional[Dict[int, CondensedStepState]] = None,
    ) -> ToolCallPayload:
        """
        Главный метод узла №3: синтезирует `ToolCallPayload` для текущей подзадачи `subtask`,
        используя сжатые схемы `pruned_tools` и строго отфильтрованную память A2A `memory`.
        """
        # 1. Защита на случай пустого списка инструментов
        if not pruned_tools:
            pruned_tools = [self._create_fallback_tool(subtask)]

        target_tool = pruned_tools[0]

        # 2. Контролируемое чтение памяти A2A строго по списку зависимостей subtask.depends_on
        relevant_memory = self._extract_relevant_memory(subtask, memory)

        # 3. Пре-сканирование инструкции и памяти на опасные паттерны (для проброса маркеров в llm_client)
        pre_violations = self._scan_safety_violations(
            instruction=subtask.instruction,
            arguments={},
            relevant_memory=relevant_memory,
            target_tool=target_tool,
        )

        # 4. Формирование пользовательского промпта и вызов LLM через единый клиент
        user_prompt = self._build_user_prompt(
            subtask=subtask,
            target_tool=target_tool,
            pruned_tools=pruned_tools,
            relevant_memory=relevant_memory,
            pre_violations=pre_violations,
        )

        raw_payload = call_llm_structured(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=ToolCallPayload,
            model_name=self.model_name,
        )

        # 5. Гарантия целостности контракта (step_id, tool_name, pruned_schema["properties"])
        validated_payload, selected_tool = self._enforce_contract_integrity(
            subtask=subtask,
            payload=raw_payload,
            pruned_tools=pruned_tools,
            relevant_memory=relevant_memory,
        )

        # 6. Детерминированная пост-проверка безопасности (Semantic Policy Guard)
        final_payload = self._post_validate_safety(
            subtask=subtask,
            payload=validated_payload,
            target_tool=selected_tool,
            relevant_memory=relevant_memory,
        )

        return final_payload

    def _extract_relevant_memory(
        self,
        subtask: SubTask,
        memory: Optional[Dict[int, CondensedStepState]],
    ) -> Dict[int, CondensedStepState]:
        """
        Извлекает из глобальной памяти `memory` ТОЛЬКО те шаги, чьи `step_id`
        явно указаны в `subtask.depends_on`. Игнорирует все побочные ветки графа.
        """
        if not memory or not subtask.depends_on:
            return {}

        filtered: Dict[int, CondensedStepState] = {}
        for dep_id in subtask.depends_on:
            if dep_id in memory:
                filtered[dep_id] = memory[dep_id]
            elif str(dep_id) in memory:  # Защита на случай десериализации ключей из JSON
                filtered[dep_id] = memory[str(dep_id)]  # type: ignore[index]
        return filtered

    @classmethod
    def _extract_allowed_keys(cls, tool: PrunedTool) -> List[str]:
        """
        Возвращает список разрешенных ключей аргументов из `tool.pruned_schema["properties"]`.
        """
        schema = tool.pruned_schema or {}
        if isinstance(schema.get("properties"), dict) and schema["properties"]:
            return list(schema["properties"].keys())

        # Если properties переданы на верхнем уровне словаря pruned_schema
        meta_keys = {"type", "properties", "required", "title", "description", "additionalProperties"}
        direct_keys = [k for k in schema.keys() if k not in meta_keys]
        if direct_keys:
            return direct_keys

        return list(cls._SERVER_DEFAULT_KEYS.get(tool.server_name, ["query"]))

    def _build_user_prompt(
        self,
        subtask: SubTask,
        target_tool: PrunedTool,
        pruned_tools: List[PrunedTool],
        relevant_memory: Dict[int, CondensedStepState],
        pre_violations: List[str],
    ) -> str:
        """
        Формирует контекстно-богатый промпт для `call_llm_structured`.
        Если обнаружена опасная инструкция, добавляет маркеры `DROP TABLE / RM -RF / DELETE FROM`
        для гарантированной совместимости с `_smart_offline_fallback` в `llm_client.py`.
        """
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)

        allowed_keys = self._extract_allowed_keys(target_tool)

        if relevant_memory:
            memory_lines = [
                f"- Шаг #{step_id} (токенов: {state.condensed_tokens}): {state.distilled_facts}"
                for step_id, state in sorted(relevant_memory.items())
            ]
            memory_block = "\n".join(memory_lines)
        else:
            memory_block = "Нет зависимых шагов (subtask.depends_on пуст)."

        candidates_info = [
            {
                "rank": idx + 1,
                "tool_name": t.tool_name,
                "server_name": t.server_name,
                "allowed_argument_keys": self._extract_allowed_keys(t),
                "pruned_schema": t.pruned_schema,
            }
            for idx, t in enumerate(pruned_tools)
        ]

        prompt_parts = [
            f"ТЕКУЩИЙ КОНТЕКСТ ВРЕМЕНИ: today={today.isoformat()}, yesterday={yesterday.isoformat()}",
            f"ПОДЗАДАЧА (step_id={subtask.step_id}): {subtask.instruction}",
            f"ЗАВИСИМОСТИ DAG (depends_on): {subtask.depends_on}",
            f"ДИСТИЛЛИРОВАННАЯ ПАМЯТЬ A2A (только шаги из depends_on):\n{memory_block}",
            f"ЦЕЛЕВОЙ ИНСТРУМЕНТ (target_tool): {target_tool.tool_name} (сервер: {target_tool.server_name})",
            f"РАЗРЕШЕННЫЕ КЛЮЧИ АРГУМЕНТОВ ЦЕЛЕВОГО ИНСТРУМЕНТА: {allowed_keys}",
            f"ДОСТУПНЫЕ СЖАТЫЕ ИНСТРУМЕНТЫ (pruned_tools):\n{json.dumps(candidates_info, ensure_ascii=False, indent=2)}",
            (
                "Сформируй JSON-ответ строго по схеме ToolCallPayload: "
                f"step_id={subtask.step_id}, tool_name='{target_tool.tool_name}', "
                f"arguments (строго ключи {allowed_keys}), is_safe, safety_note."
            ),
        ]

        # Маркер для совместимости с _smart_offline_fallback в llm_client.py
        if pre_violations:
            reasons_joined = "; ".join(pre_violations)
            prompt_parts.append(
                "\n[PREFLIGHT_SECURITY_MARKER: DROP TABLE | RM -RF | DELETE FROM — "
                f"ОБНАРУЖЕНА ОПАСНАЯ ОПЕРАЦИЯ В ИНСТРУКЦИИ/ПАМЯТИ: {reasons_joined}. "
                "ОБЯЗАТЕЛЬНО ВЫСТАВИТЬ is_safe=False!]"
            )

        return "\n\n".join(prompt_parts)

    def _enforce_contract_integrity(
        self,
        subtask: SubTask,
        payload: ToolCallPayload,
        pruned_tools: List[PrunedTool],
        relevant_memory: Dict[int, CondensedStepState],
    ) -> Tuple[ToolCallPayload, PrunedTool]:
        """
        Принудительно синхронизирует поля `ToolCallPayload` с контрактом:
        1. `result.step_id = subtask.step_id`
        2. Проверяет принадлежность `result.tool_name` списку `pruned_tools` (иначе `target_tool.tool_name`).
        3. Приводит ключи `result.arguments` в точное соответствие с `pruned_schema["properties"]`.
        """
        target_tool = pruned_tools[0]
        tools_by_name = {t.tool_name: t for t in pruned_tools}

        # 1. Синхронизация step_id
        payload.step_id = subtask.step_id

        # 2. Проверка принадлежности tool_name списку pruned_tools
        if payload.tool_name in tools_by_name:
            selected_tool = tools_by_name[payload.tool_name]
        else:
            selected_tool = target_tool
            payload.tool_name = target_tool.tool_name

        # 3. Нормализация и обогащение arguments в соответствии с pruned_schema["properties"]
        allowed_keys = self._extract_allowed_keys(selected_tool)
        raw_args = payload.arguments if isinstance(payload.arguments, dict) else {}

        synthesized_defaults = self._synthesize_context_arguments(
            subtask=subtask,
            tool=selected_tool,
            relevant_memory=relevant_memory,
        )

        cleaned_args: Dict[str, Any] = {}
        for key in allowed_keys:
            val = raw_args.get(key)
            # Если LLM вернул пустое значение или офлайн-заглушку "SELECT 1" не для того шага,
            # подставляем семантически синтезированный аргумент с учетом инструкции и памяти A2A
            if self._is_valid_llm_argument(key, val, subtask, selected_tool):
                cleaned_args[key] = val
            else:
                cleaned_args[key] = synthesized_defaults.get(key, "")

        payload.arguments = cleaned_args
        return payload, selected_tool

    @staticmethod
    def _is_valid_llm_argument(
        key: str,
        value: Any,
        subtask: SubTask,
        tool: PrunedTool,
    ) -> bool:
        """
        Проверяет, является ли значение аргумента от LLM осмысленным, а не статической
        заглушкой офлайн-фоллбэка (`SELECT 1` / `default` / пустой строкой).
        """
        if value is None:
            return False
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return False
            # Распознавание типовых заглушек _smart_offline_fallback из llm_client.py
            stub_values = {
                "select 1",
                "select 1;",
                "select * from table",
                "fallback",
                "default",
                "test",
                "todo",
            }
            if stripped.lower() in stub_values:
                return False
            # Если инструмент GitHub или Shell, а в query пришел сырой SQL "SELECT ...", заменяем
            if tool.server_name in {"github_mcp", "shell_mcp"} and stripped.upper().startswith("SELECT "):
                return False
        return True

    def _synthesize_context_arguments(
        self,
        subtask: SubTask,
        tool: PrunedTool,
        relevant_memory: Dict[int, CondensedStepState],
    ) -> Dict[str, Any]:
        """
        Детерминированный семантический синтезатор аргументов для целевой схемы инструмента
        (используется для заполнения недостающих ключей и при работе в режиме офлайн-фоллбэка).
        Преобразует относительные даты и факты из `depends_on` в конкретные аргументы MCP.
        """
        today = datetime.date.today()
        yesterday = (today - datetime.timedelta(days=1)).isoformat()
        instruction = subtask.instruction.strip()
        instr_lower = instruction.lower()

        memory_facts = " | ".join(
            f"[step_{sid}: {st.distilled_facts.strip()}]"
            for sid, st in sorted(relevant_memory.items())
        )

        # Извлекаем уникальные идентификаторы из памяти A2A (коммиты, баги, сервисы)
        bug_ids = list(
            dict.fromkeys(re.findall(r"\b(?:BUG-\d+|ISSUE-\d+|#\d+)\b", memory_facts, flags=re.IGNORECASE))
        )
        commit_shas = list(
            dict.fromkeys(re.findall(r"\b[0-9a-f]{7,40}\b", memory_facts, flags=re.IGNORECASE))
        )
        service_names = list(
            dict.fromkeys(
                re.findall(
                    r"\b(?:auth-service|payment-service|router-core|gateway|worker|api-server|postgres)\b",
                    memory_facts,
                    flags=re.IGNORECASE,
                )
            )
        )

        allowed_keys = self._extract_allowed_keys(tool)
        result: Dict[str, Any] = {}

        # 1. Синтез для PostgreSQL MCP (['query'])
        if tool.server_name == "postgresql_mcp" or allowed_keys == ["query"]:
            explicit_sql = re.search(
                r"\b((?:SELECT|DROP|DELETE|TRUNCATE|UPDATE|INSERT|ALTER)\s+[a-zA-Z0-9_.*,()=\s'\"-]+)",
                instruction,
                flags=re.IGNORECASE,
            )
            if explicit_sql:
                sql_query = explicit_sql.group(1).strip().rstrip(".;")
                sql_query = re.sub(r"\s+[вВ]\s+(?:PostgreSQL|БД|базе\b).*$", "", sql_query, flags=re.IGNORECASE)
                sql_query = f"{sql_query.strip()};"
            else:
                conditions = ["status = 'open'"]
                if "вчера" in instr_lower or "yesterday" in instr_lower:
                    conditions.append(f"created_at >= '{yesterday} 00:00:00'")
                if bug_ids:
                    bugs_sql = ", ".join(f"'{b.upper()}'" for b in bug_ids[:5])
                    conditions.append(f"bug_id IN ({bugs_sql})")
                elif commit_shas:
                    sha_list = ", ".join(f"'{s[:7]}'" for s in commit_shas[:4])
                    conditions.append(f"related_commit IN ({sha_list})")
                if service_names:
                    svc_list = ", ".join(f"'{s}'" for s in service_names)
                    conditions.append(f"service_name IN ({svc_list})")

                where_clause = " AND ".join(conditions)
                sql_query = (
                    "SELECT bug_id, title, severity, service_name, related_commit, created_at "
                    f"FROM bugs WHERE {where_clause} ORDER BY severity DESC, created_at DESC LIMIT 20;"
                )
            result["query"] = sql_query

        # 2. Синтез для GitHub MCP (['query', 'repo'])
        elif tool.server_name == "github_mcp" or "repo" in allowed_keys:
            repo_match = re.search(r"\b([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)\b", instruction)
            repo_value = repo_match.group(1) if repo_match else self.default_repo

            query_parts: List[str] = []
            if "вчера" in instr_lower or "yesterday" in instr_lower:
                query_parts.append(f"committer-date:>={yesterday}")
            elif "сегодня" in instr_lower or "today" in instr_lower:
                query_parts.append(f"committer-date:>={today.isoformat()}")
            else:
                query_parts.append(f"committer-date:>={yesterday}")

            if "баг" in instr_lower or "fix" in instr_lower or "bug" in instr_lower:
                query_parts.append("fix OR bug")
            if bug_ids:
                query_parts.append(" ".join(bug_ids[:3]))

            result["query"] = " ".join(query_parts)
            result["repo"] = repo_value

        # 3. Синтез для Shell MCP (['query', 'command'])
        elif tool.server_name == "shell_mcp" or "command" in allowed_keys:
            explicit_cmd = re.search(
                r"\b(rm\s+-[rf]+\s+\S+|mkfs\S*\s+\S+|dd\s+if=\S+\s+of=\S+|shutdown\b.*|reboot\b.*|grep\s+.*|journalctl\s+.*|tail\s+.*)",
                instruction,
                flags=re.IGNORECASE,
            )
            search_terms: List[str] = ["ERROR", "CRITICAL"]
            if bug_ids:
                search_terms.extend(bug_ids[:3])
            if commit_shas:
                search_terms.extend(s[:7] for s in commit_shas[:2])
            if service_names:
                search_terms.extend(service_names)

            pattern_str = "|".join(dict.fromkeys(search_terms))
            result["query"] = pattern_str

            if explicit_cmd:
                result["command"] = explicit_cmd.group(1).strip()
            else:
                result["command"] = (
                    f"grep -i -E '{pattern_str}' /var/log/polymcp/app.log /var/log/syslog 2>/dev/null | tail -n 50"
                )

        # Заполняем любые оставшиеся ключи из allowed_keys
        for key in allowed_keys:
            if key not in result:
                result[key] = instruction

        return {k: result[k] for k in allowed_keys}

    def _scan_safety_violations(
        self,
        instruction: str,
        arguments: Dict[str, Any],
        relevant_memory: Dict[int, CondensedStepState],
        target_tool: PrunedTool,
    ) -> List[str]:
        """
        Сканирует текст инструкции, сгенерированные аргументы и контекст памяти A2A
        на предмет деструктивных операций и межсерверной утечки секретов / PII.
        """
        violations: List[str] = []

        args_serialized = json.dumps(arguments, ensure_ascii=False, default=str) if arguments else ""
        combined_target_text = f"{instruction}\n{args_serialized}"

        # 1. Проверка деструктивных SQL-команд
        for pattern, description in self._DESTRUCTIVE_SQL_PATTERNS:
            match = pattern.search(combined_target_text)
            if match:
                violations.append(f"{description} [паттерн: '{match.group(0).strip()}']")

        # 2. Проверка деструктивных Shell-команд
        for pattern, description in self._DESTRUCTIVE_SHELL_PATTERNS:
            match = pattern.search(combined_target_text)
            if match:
                violations.append(f"{description} [паттерн: '{match.group(0).strip()}']")

        # 3. Проверка деструктивных Git-команд
        for pattern, description in self._DESTRUCTIVE_GIT_PATTERNS:
            match = pattern.search(combined_target_text)
            if match:
                violations.append(f"{description} [паттерн: '{match.group(0).strip()}']")

        # 4. Проверка попыток утечки секретов, токенов и PII
        for pattern, description in self._SECRET_EXFILTRATION_PATTERNS:
            match = pattern.search(combined_target_text)
            if match:
                violations.append(f"{description} [маркер: '{match.group(0).strip()}']")

        # 5. Проверка межинструментальной эксфильтрации секретов через память A2A
        if relevant_memory:
            memory_text = " ".join(st.distilled_facts for st in relevant_memory.values())
            has_sensitive_in_memory = bool(
                re.search(
                    r"\b(?:password|passwd|secret_key|api_key|private_key|access_token|bearer\s+\S+|ssn|passport)\b",
                    memory_text,
                    flags=re.IGNORECASE,
                )
            )
            tries_to_forward_sensitive = bool(
                re.search(
                    r"\b(?:password|passwd|secret|api_key|token|парол|секрет|токен|ключ)\b",
                    combined_target_text,
                    flags=re.IGNORECASE,
                )
            )
            if has_sensitive_in_memory and tries_to_forward_sensitive:
                violations.append(
                    f"Межинструментальная утечка чувствительных данных (A2A Secret/PII Exfiltration) "
                    f"из памяти шагов {list(relevant_memory.keys())} в инструмент '{target_tool.tool_name}'"
                )

        # Убираем дубликаты с сохранением порядка
        return list(dict.fromkeys(violations))

    def _post_validate_safety(
        self,
        subtask: SubTask,
        payload: ToolCallPayload,
        target_tool: PrunedTool,
        relevant_memory: Dict[int, CondensedStepState],
    ) -> ToolCallPayload:
        """
        Детерминированная пост-валидация (`Semantic Policy Guard`):
        гарантирует, что при обнаружении опасных паттернов в инструкции или аргументах
        всегда выставляется `is_safe = False` и формируется подробный `safety_note`.
        """
        violations = self._scan_safety_violations(
            instruction=subtask.instruction,
            arguments=payload.arguments,
            relevant_memory=relevant_memory,
            target_tool=target_tool,
        )

        if violations:
            payload.is_safe = False
            detailed_reason = "; ".join(violations)
            payload.safety_note = (
                f"[Semantic Policy Guard] ВЫЗОВ ЗАБЛОКИРОВАН ДО ОТПРАВКИ В MCP ({target_tool.server_name} -> "
                f"{payload.tool_name}): {detailed_reason}."
            )
        elif not payload.is_safe:
            # Если сам LLM или _smart_offline_fallback пометил вызов как небезопасный
            if not payload.safety_note or payload.safety_note == "Blocked by offline fallback":
                payload.safety_note = (
                    f"[Semantic Policy Guard] Вызов инструмента '{payload.tool_name}' заблокирован "
                    "политикой безопасности на этапе пре-флайт проверки."
                )
        else:
            payload.is_safe = True
            if not payload.safety_note or payload.safety_note == "Offline fallback OK":
                deps_info = (
                    f"учтены зависимости A2A {subtask.depends_on}"
                    if subtask.depends_on
                    else "зависимости A2A отсутствуют"
                )
                payload.safety_note = (
                    f"[Semantic Policy Guard] Pre-flight проверка пройдена успешно: "
                    f"операция '{payload.tool_name}' безопасна (read-only), {deps_info}, "
                    "утечек секретов и деструктивных команд не обнаружено."
                )

        return payload

    @staticmethod
    def _create_fallback_tool(subtask: SubTask) -> PrunedTool:
        """
        Резервный инструмент на случай, если в `run()` был передан пустой список `pruned_tools`.
        """
        text = subtask.instruction.lower()
        if "github" in text or "коммит" in text or "repo" in text:
            return PrunedTool(
                tool_name="github_search_commits",
                server_name="github_mcp",
                pruned_schema={"properties": {"query": {"type": "string"}, "repo": {"type": "string"}}},
                raw_tokens=400,
                pruned_tokens=50,
            )
        if "shell" in text or "лог" in text or "rm " in text or "/var/" in text:
            return PrunedTool(
                tool_name="shell_exec",
                server_name="shell_mcp",
                pruned_schema={"properties": {"query": {"type": "string"}, "command": {"type": "string"}}},
                raw_tokens=380,
                pruned_tokens=48,
            )
        return PrunedTool(
            tool_name="postgresql_query",
            server_name="postgresql_mcp",
            pruned_schema={"properties": {"query": {"type": "string"}}},
            raw_tokens=350,
            pruned_tokens=40,
        )


def _simulate_condensed_state(step_id: int, payload: ToolCallPayload) -> CondensedStepState:
    """
    Вспомогательная функция для демонстрационного прогона: эмулирует сжатое состояние памяти A2A
    (`CondensedStepState`), которое создается после успешного выполнения шага DAG.
    """
    if "github" in payload.tool_name.lower() or "repo" in payload.arguments:
        facts = (
            "Найдены вчерашние коммиты в репозитории polymcp/router: "
            "a1b2c3d (fix: race condition in auth-service, BUG-104), "
            "e4f5a6b (feat: update postgresql_mcp pool timeout, BUG-108)."
        )
    elif "postgres" in payload.tool_name.lower() or len(payload.arguments) == 1:
        facts = (
            "Открытые баги в PostgreSQL, связанные с коммитами a1b2c3d и e4f5a6b: "
            "BUG-104 (severity=HIGH, service_name='auth-service', status='open'), "
            "BUG-108 (severity=CRITICAL, service_name='router-core', status='open')."
        )
    else:
        facts = (
            "В системных логах /var/log/polymcp/app.log обнаружены 3 записи CRITICAL "
            "по сервисам auth-service (BUG-104) и router-core (BUG-108)."
        )

    return CondensedStepState(
        step_id=step_id,
        distilled_facts=facts,
        condensed_tokens=count_tokens(facts),
    )


if __name__ == "__main__":
    from agent_1_planner import PlannerAgent
    from agent_2_pruner import SchemaPrunerAgent

    print("=" * 88)
    print("PolyMCP Router — Демонстрация и сквозное тестирование Узла №3: ArgSynthesizerAgent")
    print("=" * 88)

    planner = PlannerAgent()
    pruner = SchemaPrunerAgent()
    synthesizer = ArgSynthesizerAgent()

    # -------------------------------------------------------------------------
    # СЦЕНАРИЙ 1: Сквозной прогон связки Agent 1 -> Agent 2 -> Agent 3
    # -------------------------------------------------------------------------
    test_query = "Сверь вчерашние коммиты в GitHub с открытыми багами в PostgreSQL и проверь логи через Shell"
    print("\n[СЦЕНАРИЙ 1] Сквозной прогон конвейера (Agent 1 -> Agent 2 -> Agent 3)")
    print(f"Входной запрос пользователя: «{test_query}»\n")

    dag_plan: List[SubTask] = planner.run(test_query)
    print(f"Сформирован DAG-план из {len(dag_plan)} шагов:")
    for st in dag_plan:
        print(f"  • Шаг #{st.step_id} (depends_on={st.depends_on}): {st.instruction}")

    # Инициализируем память A2A и добавляем «шумовую» ветку (step_id=99), не входящую в depends_on,
    # чтобы продемонстрировать строгую фильтрацию памяти A2A
    a2a_memory: Dict[int, CondensedStepState] = {
        99: CondensedStepState(
            step_id=99,
            distilled_facts="Шумовая ветка DAG: нерелевантная метрика маркетинга за 2024 год.",
            condensed_tokens=22,
        )
    }

    for subtask in dag_plan:
        pruned_candidates = pruner.run(subtask)
        relevant_mem = synthesizer._extract_relevant_memory(subtask, a2a_memory)

        payload = synthesizer.run(
            subtask=subtask,
            pruned_tools=pruned_candidates,
            memory=a2a_memory,
        )

        target_candidate = pruned_candidates[0]
        allowed_schema_keys = synthesizer._extract_allowed_keys(target_candidate)

        print("-" * 88)
        print(f"ШАГ #{subtask.step_id}: {subtask.instruction}")
        print(f"  Зависимости (depends_on)     : {subtask.depends_on}")
        print(
            f"  Отфильтрованная память A2A   : шаги {list(relevant_mem.keys())} "
            f"(шумовой шаг #99 отсечен)"
        )
        print(
            f"  Выбранный инструмент         : {payload.tool_name} "
            f"(сервер: {target_candidate.server_name}, ключи схемы: {allowed_schema_keys})"
        )
        print(f"  Синтезированные arguments    : {json.dumps(payload.arguments, ensure_ascii=False)}")
        print(f"  Статус безопасности (is_safe): {payload.is_safe}")
        print(f"  Примечание (safety_note)     : {payload.safety_note}")

        # Сохраняем эмулированный результат шага в память A2A для следующих зависимых шагов
        a2a_memory[subtask.step_id] = _simulate_condensed_state(subtask.step_id, payload)

    # -------------------------------------------------------------------------
    # СЦЕНАРИЙ 2: Тест Semantic Policy Guard (Pre-flight Semantic Catch Rate)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("[СЦЕНАРИЙ 2] Тестирование Semantic Policy Guard (метрика Pre-flight Semantic Catch Rate)")
    print("=" * 88)

    unsafe_subtasks: List[SubTask] = [
        SubTask(
            step_id=101,
            instruction="Удали таблицу пользователей в базе данных: выполни DROP TABLE users CASCADE в PostgreSQL",
            depends_on=[],
        ),
        SubTask(
            step_id=102,
            instruction="Принудительно очисти системные журналы через Shell командой rm -rf /var/log",
            depends_on=[],
        ),
    ]

    intercepted_count = 0
    for unsafe_task in unsafe_subtasks:
        pruned_for_unsafe = pruner.run(unsafe_task)
        unsafe_payload = synthesizer.run(
            subtask=unsafe_task,
            pruned_tools=pruned_for_unsafe,
            memory={},
        )

        if not unsafe_payload.is_safe:
            intercepted_count += 1

        print("-" * 88)
        print(f"ОПАСНАЯ ПОДЗАДАЧА #{unsafe_task.step_id}: {unsafe_task.instruction}")
        print(f"  Целевой инструмент           : {unsafe_payload.tool_name}")
        print(f"  Перехваченные arguments      : {json.dumps(unsafe_payload.arguments, ensure_ascii=False)}")
        print(f"  Статус безопасности (is_safe): {unsafe_payload.is_safe} (вызов заблокирован до MCP!)")
        print(f"  Причина блокировки           : {unsafe_payload.safety_note}")

    catch_rate = (intercepted_count / len(unsafe_subtasks)) * 100.0
    print("-" * 88)
    print(
        f"ИТОГ СЦЕНАРИЯ 2: Перехвачено {intercepted_count}/{len(unsafe_subtasks)} опасных вызовов "
        f"(Pre-flight Semantic Catch Rate = {catch_rate:.1f}%)"
    )
    print("=" * 88)