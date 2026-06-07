"""Plugin hooks for runtime customization."""

from __future__ import annotations

import inspect
from typing import Any, Protocol


class Plugin(Protocol):
    """Optional hook protocol.

    Plugins may implement any of:
    before_plan, after_plan, before_execute, after_execute,
    before_reflect, after_reflect, before_learn, after_learn.
    """


class PluginManager:
    def __init__(self, plugins: list[Plugin] | None = None) -> None:
        self.plugins = plugins or []

    def register(self, plugin: Plugin) -> None:
        self.plugins.append(plugin)

    async def emit(self, hook_name: str, **payload: Any) -> None:
        for plugin in self.plugins:
            hook = getattr(plugin, hook_name, None)
            if hook is None:
                continue
            result = hook(**payload)
            if inspect.isawaitable(result):
                await result

