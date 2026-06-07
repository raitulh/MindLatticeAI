from mindlatticeai import Agent


def main() -> None:
    agent = Agent()
    response = agent.run("Calculate 21 * 2")
    print(response.text)
    print(response.artifacts["graph_dot"])


if __name__ == "__main__":
    main()

