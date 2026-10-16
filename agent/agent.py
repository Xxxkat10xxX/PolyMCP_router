import json
import requests
from typing import Dict, Any
from jsonschema import validate, ValidationError

class ExecutionAndSandboxAgent:
    def __init__(self, catalog_path: str = "tools_catalog_raw.json"):
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        # Карта схем для валидации: { tool_name: inputSchema }
        self.schemas = {t["name"]: t.get("inputSchema", {}) for t in catalog.get("tools", [])}
        # Карта серверов: { tool_name: endpoint_url }
        server_urls = {
            "postgres": "http://localhost:8001/mcp",
            "github": "http://localhost:8002/mcp",
            "shell": "http://localhost:8003/mcp"
        }
        self.tool_routes = {t["name"]: server_urls.get(t.get("server", "shell")) for t in catalog.get("tools", [])}

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        1. Проверка существования инструмента
        2. Синтаксическая валидация аргументов схемы
        3. Отправка RPC-запроса в контейнер
        4. Перехват исключений и возврат структурированного результата
        """
        # Шаг 1: Проверка инструмента
        if tool_name not in self.schemas:
            return {
                "status": "error",
                "error_type": "tool_not_found",
                "message": f"Инструмент '{tool_name}' отсутствует в зарегистрированном каталоге."
            }

        # Шаг 2: Валидация аргументов (Schema Validation)
        expected_schema = self.schemas[tool_name]
        try:
            validate(instance=arguments, schema=expected_schema)
        except ValidationError as val_err:
            return {
                "status": "error",
                "error_type": "schema_validation_error",
                "message": f"Невалидные аргументы: {val_err.message}",
                "expected_schema": expected_schema,
                "received_arguments": arguments
            }

        # Шаг 3: Отправка JSON-RPC вызова в контейнер
        endpoint = self.tool_routes.get(tool_name)
        rpc_payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": 100
        }

        # Шаг 4: Безопасное исполнение и перехват исключений среды
        try:
            response = requests.post(endpoint, json=rpc_payload, timeout=5)
            response.raise_for_status()
            rpc_result = response.json()

            if "error" in rpc_result:
                return {
                    "status": "error",
                    "error_type": "mcp_rpc_error",
                    "message": rpc_result["error"].get("message", "Ошибка RPC сервера")
                }

            return {
                "status": "success",
                "data": rpc_result.get("result", {}),
                "tool_used": tool_name
            }

        except requests.exceptions.ConnectionError:
            # Если контейнер не запущен или упал
            return {
                "status": "error",
                "error_type": "runtime_connection_error",
                "message": f"Контейнер MCP-сервера недоступен по адресу {endpoint}."
            }
        except requests.exceptions.Timeout:
            return {
                "status": "error",
                "error_type": "runtime_timeout_error",
                "message": f"Превышено время ожидания ответа от контейнера ({tool_name})."
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": "internal_error",
                "message": str(e)
            }


# Пример ручной проверки агента:
if __name__ == "__main__":
    agent = ExecutionAndSandboxAgent()

    # Тест 1: Передаем валидные аргументы
    print("--- Тест валидного вызова ---")
    res1 = agent.execute("query_db", {"query": "SELECT * FROM system_incidents;"})
    print(json.dumps(res1, indent=2, ensure_ascii=False))

    # Тест 2: Передаем невалидные аргументы (вместо строки передан int)
    print("\n--- Тест ошибки валидации схемы ---")
    res2 = agent.execute("query_db", {"query": 12345})
    print(json.dumps(res2, indent=2, ensure_ascii=False))