"""Policy engine for dynamic runtime constraints."""

from __future__ import annotations

from mindlatticeai.schemas import AIRequest, Intent, PolicyDecision, Strategy


class PolicyEngine:
    """Applies policy rules before a strategy becomes executable."""

    def __init__(self, *, disabled_tools: set[str] | None = None) -> None:
        self.disabled_tools = disabled_tools or set()

    def evaluate(self, request: AIRequest, intent: Intent, strategy: Strategy) -> PolicyDecision:
        reasons: list[str] = []
        allowed = True
        disabled = sorted(set(intent.required_tools).intersection(self.disabled_tools))

        if request.budget.max_cost_usd < 0:
            allowed = False
            reasons.append("Negative cost budget is invalid.")
        if disabled:
            allowed = False
            reasons.append(f"Disabled tools requested: {', '.join(disabled)}")
        if intent.risk_level == "elevated":
            reasons.append("Elevated risk: security-sensitive terms detected.")
        if not reasons:
            reasons.append("Policy checks passed.")

        preferences = {
            "prefer_low_cost": request.budget.max_cost_usd <= 0.05,
            "prefer_low_latency": request.budget.latency_budget_ms <= 2500,
            "quality_floor": request.budget.quality_floor,
        }
        return PolicyDecision(
            allowed=allowed,
            reasons=reasons,
            budgets=request.budget,
            disabled_tools=disabled,
            routing_preferences=preferences,
            metadata={"strategy_id": strategy.id},
        )

