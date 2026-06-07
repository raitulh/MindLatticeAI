from mindlatticeai import Agent
from mindlatticeai.plugins import PluginManager


class AuditPlugin:
    def __init__(self) -> None:
        self.events = []

    def after_plan(self, **payload) -> None:
        self.events.append(("after_plan", payload["strategy"].id))

    def after_learn(self, **payload) -> None:
        self.events.append(("after_learn", payload["signal"].quality_score))


def main() -> None:
    plugin = AuditPlugin()
    manager = PluginManager([plugin])
    agent = Agent(plugins=manager)
    response = agent.run("Calculate 5 + 7")
    print(response.text)
    print(plugin.events)


if __name__ == "__main__":
    main()

