# MindLatticeAI

MindLatticeAI is a research-oriented adaptive AI orchestration framework. It treats every AI request as an experiment: parse intent, generate a strategy, apply runtime policy, compile a cognitive execution graph, route models and tools, reflect, evaluate, learn, update memory, and return an explainable response.

This repository is intentionally not a LangChain clone, a CrewAI-style fixed crew, a LiteLLM router, or a retrieval-only index. The core abstraction is an **Execution History Learning Loop** over **Cognitive Execution Graphs** and **evolvable Prompt Genomes**.

```text
User Input
  |
  v
Intent Parser -> Planner -> Policy Engine -> DAG Builder
                                          |
                                          v
                       Tool Layer -> Model Router -> Provider Layer
                                          |
                                          v
                       Reflection -> Evaluation -> Learning
                                          |
                                          v
                          Memory Update -> Replay -> Final Response
```

## What Exists Now

The starter implementation is working and local-first:

- DAG-based execution for each request
- Retry and checkpoint support at graph-node level
- Deterministic offline provider for tests and development
- Provider contracts for OpenAI, Gemini, Claude, Ollama, and DeepSeek
- Tool registry with safe calculator, UTC time, and local retrieval placeholder
- Multi-layer memory facade: short-term, long-term, semantic, episodic, knowledge graph
- Prompt genome registry with mutation and scoring
- Reflection and evaluation loop
- Learning engine for strategy, prompt, model, tool, and memory importance feedback
- Telemetry traces, graph DOT export, replay snapshots
- Plugin hooks around planning, execution, reflection, and learning
- Benchmark suite primitives

## Quickstart

```bash
python -m pip install -e .
```

```python
from mindlatticeai import Agent

agent = Agent()
response = agent.run("Calculate 21 * 2")

print(response.text)
print(response.metrics)
print(response.artifacts["graph_dot"])
```

Async:

```python
from mindlatticeai import Agent, AIRequest

agent = Agent()
response = await agent.arun(AIRequest(input="Analyze why DAG execution helps AI reliability."))
```

## Public API

```python
from mindlatticeai import Agent, AIRequest, RuntimeConfig, TaskType
from mindlatticeai.tools import Tool, ToolRegistry
from mindlatticeai.providers import LocalHeuristicProvider
from mindlatticeai.plugins import PluginManager
```

The primary facade is `Agent`. Advanced users can replace any subsystem:

```python
agent = Agent(
    config=RuntimeConfig(),
    providers=[LocalHeuristicProvider()],
    tools=my_tool_registry,
    plugins=my_plugin_manager,
)
```

## Architecture

MindLatticeAI separates the runtime into policy-aware layers:

```text
mindlatticeai/
  core/          Agent facade and runtime config
  planner/       Intent parser and adaptive strategy generator
  policy/        Dynamic runtime rules and budget constraints
  graph/         DAG builder, async executor, retries, checkpoints
  router/        Cost/latency/quality/availability-aware model router
  tools/         Tool registry and built-in safe tools
  memory/        Multi-layer memory system
  reflection/    Self-check and evaluator
  learner/       Execution-history learning engine
  prompts/       Prompt genome scoring, mutation, selection
  telemetry/     Traces, metrics, graph visualization
  cache/         Exact response and lexical semantic caches
  replay/        Execution replay capture
  plugins/       Custom hook system
  providers/     Local and external model provider contracts
  benchmarks/    Evaluation suite primitives
  security/      API key vault and sandbox policy
```

## Cognitive Execution Graphs

Every request is represented as a DAG. The default graph has:

```text
tool.*        optional tool nodes
  |
  v
model.main -> reflect.self_check -> evaluate.score
                                      |        |
                                      v        v
                                memory.write  learn.update
                                      |        |
                                      v        v
                                  final.response
```

Node states:

```text
pending -> running -> succeeded
pending -> running -> retrying -> succeeded
pending -> running -> failed
pending -> checkpointed/succeeded
```

The graph executor validates dependencies, executes ready nodes concurrently, retries failed nodes, records attempts, and checkpoints successful node outputs.

## Prompt Genome

Prompts are structured genomes rather than static strings:

- role
- instruction
- context
- constraints
- examples
- output format
- safety rules
- self-evaluation rules
- generation
- score

Low-scoring executions can mutate a genome. Future research can replace the simple mutation policy with evolutionary search, bandits, or reinforcement learning.

## Learning Loop

The learner records:

- latency
- cost
- quality score
- success/failure
- user feedback
- hallucination probability estimate
- tool efficiency score

It updates:

- strategy scores
- prompt genome scores
- model routing priors
- tool priority scores
- memory importance bias

## Novelty Versus Existing Frameworks

MindLatticeAI is designed around adaptive execution rather than static chains:

| System | Common center of gravity | MindLatticeAI difference |
| --- | --- | --- |
| LangChain | Chains, components, integrations | Experiment loop with learning and strategy mutation |
| CrewAI | Role-based multi-agent workflows | Policy-driven cognitive graph runtime |
| AutoGen | Conversational multi-agent coordination | Replayable DAG execution with adaptive scoring |
| LiteLLM | Provider abstraction and routing | Routing is learned from execution history and policies |
| LlamaIndex | Retrieval and indexing | Memory is multi-layer and tied to execution learning |

## Limitations

This is a production-grade skeleton, not a finished research platform. The current provider layer is local by default, semantic memory uses lexical vectors, evaluation is heuristic, and durable stores are not yet implemented. Those choices keep the first repo runnable and testable without external services.

## Future Research Directions

- Learned intent parsing and policy synthesis
- Evolutionary prompt genome populations
- Offline reinforcement learning over strategy traces
- Durable graph checkpoint stores
- Vector database and knowledge graph backends
- Human feedback and preference modeling
- Formal safety policies for high-risk tools
- Distributed graph execution
- Dashboard for trace, cost, and graph replay analysis

