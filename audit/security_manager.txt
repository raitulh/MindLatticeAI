"""Basic security layer."""

from __future__ import annotations

import os


class ApiKeyVault:
    """Reads API keys from the environment without storing secrets in traces."""

    def get(self, name: str) -> str | None:
        return os.getenv(name)

    def require(self, name: str) -> str:
        value = self.get(name)
        if not value:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return value


class SandboxPolicy:
    """Basic allowlist/denylist checks for tool execution."""

    def __init__(self, *, allowed_tools: set[str] | None = None, denied_tools: set[str] | None = None) -> None:
        self.allowed_tools = allowed_tools
        self.denied_tools = denied_tools or set()

    def validate_tool(self, tool_name: str) -> None:
        if tool_name in self.denied_tools:
            raise PermissionError(f"Tool is denied by sandbox policy: {tool_name}")
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            raise PermissionError(f"Tool is not allowed by sandbox policy: {tool_name}")

