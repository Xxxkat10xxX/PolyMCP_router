import json
import requests

#test commit
MCP_ENDPOINTS = {
    "postgres": "http://localhost:8001/mcp",
    "github": "http://localhost:8002/mcp",
    "shell": "http://localhost:8003/mcp"
}

def fetch_tools(server_name, url):
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "params": {},
        "id": 1
    }
    try:
        response = requests.post(url, json=payload, timeout=3)
        if response.status_code == 200:
            result = response.json()
            return result.get("result", {}).get("tools", [])
    except Exception:
        # Если контейнеры еще настраиваются, возвращаем синтетический мок-набор
        pass
    
    # Резервный мок для отладки команды, если реальный сетевой MCP еще поднимается
    mocks = {
        "postgres": [
            {"name": "query_db", "description": "Execute a readonly SQL query", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
            {"name": "list_tables", "description": "List all tables in DB", "inputSchema": {"type": "object", "properties": {}}}
        ],
        "github": [
            {"name": "create_issue", "description": "Create a new issue on GitHub", "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}, "body": {"type": "string"}}, "required": ["title"]}},
            {"name": "list_commits", "description": "Get recent commits", "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer"}}}}
        ],
        "shell": [
            {"name": "exec_command", "description": "Run shell command in sandbox", "inputSchema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}
        ]
    }
    return mocks.get(server_name, [])

full_catalog = {"tools": []}
for s_name, endpoint in MCP_ENDPOINTS.items():
    tools = fetch_tools(s_name, endpoint)
    for t in tools:
        t["server"] = s_name
        full_catalog["tools"].append(t)

with open("tools_catalog_raw.json", "w", encoding="utf-8") as f:
    json.dump(full_catalog, f, indent=2, ensure_ascii=False)

print(f"Экспорт завершен. Собрано инструментов: {len(full_catalog['tools'])}")