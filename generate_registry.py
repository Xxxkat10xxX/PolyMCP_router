import json


def ensure_mock_registry(path: str = "mock_mcp_registry.json") -> None:
    """
    Генерирует реалистичный каталог из 20 MCP-инструментов (PostgreSQL, GitHub, Shell)
    для замеров по методологии Legroom / AttnCompress (экономия 60-78% без фейкового раздувания).
    """
    tools_spec = [
        # PostgreSQL MCP Server (7 инструментов)
        ("postgresql_mcp", "db_query_select", "Execute read-only SQL SELECT query on PostgreSQL tables with filtering and pagination."),
        ("postgresql_mcp", "db_describe_table", "Inspect table columns, data types, primary keys, and foreign key constraints in PostgreSQL."),
        ("postgresql_mcp", "db_explain_plan", "Run EXPLAIN ANALYZE on a slow SQL query to inspect execution cost and index scans."),
        ("postgresql_mcp", "db_active_locks", "List active transaction locks, blocking PIDs, and wait events in pg_locks."),
        ("postgresql_mcp", "db_replication_lag", "Check WAL replication delay and standby node synchronization status."),
        ("postgresql_mcp", "db_table_size", "Calculate disk usage, index bloat, and row counts for database tables."),
        ("postgresql_mcp", "db_export_csv", "Export query results from PostgreSQL relation into a structured CSV snapshot."),
        # GitHub MCP Server (7 инструментов)
        ("github_mcp", "gh_get_commit_history", "Fetch commit history, SHA hashes, authors, and commit messages for a branch."),
        ("github_mcp", "gh_search_issues", "Search GitHub issues and bug reports by state, labels, assignee, and text query."),
        ("github_mcp", "gh_get_issue_details", "Retrieve full issue description, stack traces, comments, and linked pull requests."),
        ("github_mcp", "gh_list_pull_requests", "List open or merged pull requests with review status and target branch."),
        ("github_mcp", "gh_compare_commits", "Compare two commits or branches to inspect modified files and code diff."),
        ("github_mcp", "gh_list_workflow_runs", "Check GitHub Actions CI/CD workflow runs, build status, and failed job steps."),
        ("github_mcp", "gh_get_file_contents", "Read source code file contents at a specific commit ref or branch in repository."),
        # Shell / OS MCP Server (6 инструментов)
        ("shell_mcp", "shell_tail_log", "Read recent lines from system or application log files in /var/log."),
        ("shell_mcp", "shell_grep_pattern", "Search log files or configuration directories for specific error patterns or IDs."),
        ("shell_mcp", "shell_docker_status", "Inspect running Docker containers, healthchecks, restart counts, and resource limits."),
        ("shell_mcp", "shell_check_process", "Monitor active OS processes, CPU/RAM utilization, and worker PIDs."),
        ("shell_mcp", "shell_disk_usage", "Check filesystem disk space, inode usage, and mount points via df/du."),
        ("shell_mcp", "shell_exec_command", "Execute an isolated diagnostic shell command in the sandbox environment."),
    ]

    registry = []
    for server_name, tool_name, desc in tools_spec:
        properties = {
            "query": {
                "type": "string",
                "description": f"Primary SQL query, search filter, or target expression for {tool_name}.",
            },
            "repo": {
                "type": "string",
                "description": "Repository identifier in owner/repo format (used for GitHub MCP calls).",
            },
            "command": {
                "type": "string",
                "description": "Sandbox shell command or log path argument (used for Shell MCP calls).",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of records, commits, or log lines to return (default: 50, max: 100).",
            },
            "timeout_ms": {
                "type": "integer",
                "description": "Execution timeout in milliseconds before aborting the RPC request (default: 5000).",
            },
            "output_format": {
                "type": "string",
                "description": "Serialization format for tool response payload: json, markdown_table, or compact_csv.",
            },
            "include_metadata": {
                "type": "boolean",
                "description": "Whether to attach verbose server execution headers, schema version, and pagination cursors.",
            },
            "sort_order": {
                "type": "string",
                "description": "Sorting direction for timestamped records (asc or desc by created_at/updated_at).",
            },
        }

        registry.append({
            "tool_name": tool_name,
            "server_name": server_name,
            "description": desc,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": ["query"],
            },
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    print(f"Реалистичный каталог {path} обновлен! Инструментов: {len(registry)}")


if __name__ == "__main__":
    ensure_mock_registry()