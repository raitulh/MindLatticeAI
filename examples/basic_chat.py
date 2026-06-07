from mindlatticeai import Agent


def main() -> None:
    agent = Agent()
    response = agent.run("Explain MindLatticeAI in two sentences.")
    print(response.text)
    print(response.metrics.model_dump())


if __name__ == "__main__":
    main()

