"""Safe built-in tools for the starter runtime."""

from __future__ import annotations

import ast
import operator
from datetime import datetime, timezone
from typing import Any

from mindlatticeai.tools.registry import Tool, ToolRegistry

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Expression contains unsupported syntax")


def calculator(arguments: dict[str, Any]) -> dict[str, Any]:
    expression = str(arguments.get("expression") or arguments.get("query") or "")
    parsed = ast.parse(expression, mode="eval")
    return {"expression": expression, "value": _safe_eval(parsed.body)}


def utc_time(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"utc": datetime.now(timezone.utc).isoformat()}


def echo_search(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query", ""))
    return {
        "query": query,
        "results": [
            {
                "title": "Local semantic placeholder",
                "snippet": "Attach a retrieval plugin to replace this deterministic starter result.",
            }
        ],
    }


def register_builtin_tools(registry: ToolRegistry) -> None:
    registry.register(
        Tool(
            name="calculator",
            description="Safely evaluates simple arithmetic expressions.",
            handler=calculator,
        )
    )
    registry.register(
        Tool(
            name="utc_time",
            description="Returns the current UTC timestamp.",
            handler=utc_time,
        )
    )
    registry.register(
        Tool(
            name="local_search",
            description="Deterministic retrieval placeholder for examples and tests.",
            handler=echo_search,
        )
    )

