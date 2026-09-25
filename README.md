<div align="center">

<p align="center">
  <img src="docs/assets/banner.svg" alt="MindLatticeAI Banner" width="100%" />
</p>

# 🧠 MindLatticeAI

### *Adaptive AI Orchestration Framework with Execution-History Learning & Cognitive DAGs*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Tests-18%20Passing-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![Architecture](https://img.shields.io/badge/Architecture-Cognitive%20DAG-8A2BE2?style=for-the-badge)](docs/architecture.md)
[![Status](https://img.shields.io/badge/Status-Research%20Alpha-00F0FF?style=for-the-badge)](https://github.com/raitulh/MindLatticeAI)
[![Stars](https://img.shields.io/github/stars/raitulh/MindLatticeAI?style=for-the-badge&color=gold)](https://github.com/raitulh/MindLatticeAI/stargazers)

<br/>

[**Explore Architecture**](docs/architecture.md) •
[**Quickstart**](#-quickstart) •
[**Core Innovations**](#-core-innovations) •
[**Live Providers**](#-multi-provider-support) •
[**Framework Comparison**](#-why-mindlatticeai) •
[**Roadmap**](docs/roadmap.md)

</div>

---

## 🌟 Overview

**MindLatticeAI** is a research-oriented, adaptive AI orchestration framework designed around a fundamental premise:

> **Every AI request is an experiment.** 
> An autonomous agent must formulate a strategy, compile a resilient execution graph, evaluate its own intermediate reasoning, mutate its prompt genomes, learn from execution telemetry, and store experiential knowledge across multi-layer memory meshes.

This repository is intentionally **not** a LangChain clone, a CrewAI-style static conversational crew, or a simple LiteLLM router. Its core abstraction is an **Execution-History Learning Loop** operating over **Cognitive Directed Acyclic Graphs (DAGs)** with **checkpointed node resilience** and **evolvable prompt genomes**.

---

## ⚡ Terminal Execution Preview

```ansi
[1;36m[MindLatticeAI Runtime][0m Initializing Request: [1;33m"Calculate 21 * 2 and reflect on strategy"[0m
[1;34m├── [1. INTENT][0m      Task: [32mTOOL_AUGMENTED_REASONING[0m | Tools Required: [33m['calculator'][0m | Risk: [32mNORMAL[0m
[1;34m├── [2. PLANNER][0m     Compiled Strategy: [35mstrategy_da4539[0m | Genome: [35mgenome.default.cognitive (gen 0)[0m
[1;34m├── [3. POLICY][0m      Budget: Max $0.25 | Latency: 15,000ms | [1;32mChecks Passed (ALLOWED)[0m
[1;34m├── [4. DAG ENGINE][0m  Executing Cognitive Graph (7 Nodes):
[1;34m│   ├── [RUN][0m        tool.calculator       ───► [1;32mSUCCEEDED[0m (42.0) [1ms]
[1;34m│   ├── [RUN][0m        model.main            ───► [1;32mSUCCEEDED[0m (routed: local:heuristic-v1) [2ms]
[1;34m│   ├── [RUN][0m        reflect.self_check    ───► [1;32mSUCCEEDED[0m (critique: grounded) [1ms]
[1;34m│   ├── [RUN][0m        evaluate.score        ───► [1;32mSUCCEEDED[0m (quality: 0.932, hallucination: 0.05)
[1;34m│   ├── [PARALLEL][0m   memory.write & learn.update ───► [1;32mSUCCEEDED[0m [checkpointed]
[1;34m│   └── [FINALIZE][0m   final.response        ───► [1;32mREADY[0m
[1;34m└── [5. LEARNER][0m     Feedback Prior Updated: Prior 0.85 ──► [1;32m0.932[0m | Genome Fitness: [1;32m+0.082[0m
```

---

## 🔬 Why MindLatticeAI?

Traditional agent frameworks rely on static, brittle chains or fixed multi-agent chat loops. MindLatticeAI pioneers an **adaptive, closed-loop execution paradigm**:

| Feature / Dimension | LangChain / LlamaIndex | CrewAI / AutoGen | LiteLLM | 🧠 **MindLatticeAI** |
| :--- | :--- | :--- | :--- | :--- |
| **Execution Topology** | Static Chains / Agents | Conversational Chat Loops | Flat Provider Routing | **Dynamic Cognitive DAGs** with node checkpoints & retries |
| **Prompt Engineering** | Static template strings | Role descriptions | Passthrough | **Evolvable Prompt Genomes** with scoring & mutation |
| **Adaptation Mechanism** | None (Deterministic) | Prompt engineering | Heuristic failover | **Continuous Learning Loop** updating model & strategy priors |
| **Memory Architecture** | Buffer / Vector Index | Chat history buffer | None | **5-Layer Memory Mesh** (Short, Long, Semantic, Episodic, Graph) |
| **Self-Correction** | Ad-hoc retry | Conversational critique | None | **Dedicated Reflection & Hallucination Evaluator** |
| **Local-First Capability** | Needs external APIs | Needs external APIs | Needs external APIs | **100% Zero-Dependency Offline-First** with live API contracts |

---

## 🏗️ Architecture & Cognitive Loop

Every user query triggers an adaptive cognitive cycle:

```mermaid
flowchart TD
    classDef startNode fill:#0ea5e9,stroke:#0284c7,stroke-width:2px,color:#fff;
    classDef processNode fill:#1e293b,stroke:#3b82f6,stroke-width:1.5px,color:#f8fafc;
    classDef engineNode fill:#4c1d95,stroke:#8b5cf6,stroke-width:2px,color:#fff;
    classDef feedbackNode fill:#065f46,stroke:#10b981,stroke-width:2px,color:#fff;

    User([User Request / AIRequest]):::startNode --> Intent[Intent Parser]:::processNode
    Intent --> Planner[Adaptive Planner]:::processNode
    Planner --> Policy[Policy Engine & Budget Constraints]:::processNode
    Policy --> DAG[Cognitive DAG Builder]:::engineNode

    subgraph ExecutionGraph [Cognitive Graph Execution Engine]
        direction TB
        DAG --> TNode[Tool Nodes: Safe Calculator / Retrieval]:::processNode
        TNode --> MNode[Model Router: Adaptive Prior Selection]:::processNode
        MNode --> RNode[Reflection Engine: Self-Check & Critique]:::processNode
        RNode --> ENode[Evaluator: Scoring & Hallucination Filter]:::processNode
    end

    ENode --> Mem[Memory Update: 5-Layer Mesh]:::feedbackNode
    ENode --> Learn[Learner Engine: Strategy & Prior Update]:::feedbackNode
    Mem --> Resp([Final Replayable Response]):::startNode
    Learn --> Resp
```

---

## 🧬 Core Innovations

### 1. 🕸️ Cognitive Execution Graphs (DAG Engine)
- Requests are compiled into directed acyclic graphs containing atomic, inspectable nodes (`TOOL_CALL`, `MODEL_CALL`, `REFLECTION`, `EVALUATION`, `MEMORY_WRITE`, `LEARNING`).
- Independent nodes execute concurrently in non-blocking event loops.
- **Node-level checkpoints** enable pause, resume, and automated retries on transient network faults.
- Native export to Graphviz DOT format for instant visualization of execution pathways.

### 2. 🧬 Evolvable Prompt Genomes
Prompts are not static strings—they are structured genomes containing:
- **Identity & Instructions:** Role definitions, instruction sets, and dynamic constraints.
- **Self-Evaluation Criteria:** Groundedness checks, uncertainty estimations, and safety bounds.
- **Generational Evolution:** When an execution receives low evaluation scores, the registry mutates the genome, generating improved offspring with updated instructions and tracked parentage.

### 3. 🔄 Closed-Loop Feedback & Learning Engine
- Tracks execution metrics across every interaction: latency, financial cost, quality score, and hallucination probability.
- Automatically adjusts:
  - **Strategy preference scores**
  - **Prompt genome fitness priors**
  - **Model routing weightings**
  - **Memory retrieval importance biases**

### 4. 🧠 5-Layer Memory Mesh
- **Short-Term Memory:** Working session context with automated sliding-window compression.
- **Long-Term Memory:** Summarized durable memory across past interactions.
- **Semantic Memory:** Content-based similarity retrieval using vectorized lexical signatures.
- **Episodic Memory:** Chronological historical traces of previous graph trajectories.
- **Knowledge Graph:** Triple-based relational store (`(Entity, Relation, Entity)`) with deduplication.

---

## 🚀 Quickstart

### Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/raitulh/MindLatticeAI.git
cd MindLatticeAI
python -m pip install -e .
```

### 1. Basic Synchronous Execution

MindLatticeAI runs **100% offline out-of-the-box** using deterministic local heuristic providers:

```python
from mindlatticeai import Agent

# Initialize default agent
agent = Agent()

# Execute request
response = agent.run("Calculate 21 * 2 and explain why DAG execution improves AI reliability.")

print("Response:\n", response.text)
print("\nMetrics:", response.metrics)
print("\nReplay ID:", response.replay_id)
```

### 2. Asynchronous Execution with Custom Budget

```python
import asyncio
from mindlatticeai import Agent
from mindlatticeai.schemas import AIRequest, RuntimeBudget

async def main():
    agent = Agent()
    request = AIRequest(
        input="Provide a structured overview of cognitive computing.",
        budget=RuntimeBudget(
            max_cost_usd=0.10,
            latency_budget_ms=3000,
            quality_floor=0.80
        )
    )
    
    response = await agent.arun(request)
    print(response.text)

asyncio.run(main())
```

---

## 🌐 Multi-Provider Support

MindLatticeAI includes built-in, zero-dependency HTTP completion support for all major LLM backends. Simply export your API key:

```python
import os
from mindlatticeai import Agent
from mindlatticeai.providers import (
    GeminiProvider,
    OpenAIProvider,
    ClaudeProvider,
    DeepSeekProvider,
    OllamaProvider,
)

# Set keys
os.environ["GEMINI_API_KEY"] = "your-gemini-key"
# os.environ["OPENAI_API_KEY"] = "your-openai-key"
# os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"
# os.environ["DEEPSEEK_API_KEY"] = "your-deepseek-key"

agent = Agent(
    providers=[
        GeminiProvider(model="gemini-1.5-pro"),
        OpenAIProvider(model="gpt-4.1-mini"),
        ClaudeProvider(model="claude-3-5-sonnet"),
        DeepSeekProvider(model="deepseek-chat"),
        OllamaProvider(model="llama3.1"),
    ]
)

response = agent.run("Explain how adaptive prior routing works.")
print(response.text)
```

---

## 🛠️ Extending MindLatticeAI

<details>
<summary><b>1. Custom Tool Registration</b></summary>

```python
from mindlatticeai import Agent
from mindlatticeai.tools import Tool, ToolRegistry

def reverse_string(text: str) -> str:
    return text[::-1]

tools = ToolRegistry()
tools.register(
    Tool(
        name="string_reverser",
        description="Reverses any input string",
        handler=reverse_string,
    )
)

agent = Agent(tools=tools)
```
</details>

<details>
<summary><b>2. Lifecycle Plugins & Hooks</b></summary>

```python
from mindlatticeai import Agent
from mindlatticeai.plugins import PluginManager

plugins = PluginManager()

@plugins.hook("after_plan")
def log_plan(strategy):
    print(f"Strategy formulated: {strategy.id} with {len(strategy.steps)} steps")

@plugins.hook("after_learn")
def log_learning(score):
    print(f"Learning feedback registered: Quality Score = {score:.4f}")

agent = Agent(plugins=plugins)
```
</details>

<details>
<summary><b>3. Automated Benchmarking</b></summary>

```python
import asyncio
from mindlatticeai import Agent
from mindlatticeai.benchmarks import BenchmarkCase, BenchmarkSuite

async def run_eval():
    agent = Agent()
    suite = BenchmarkSuite([
        BenchmarkCase(name="math", input="Calculate 15 * 3", expected_keywords=["45.0"]),
        BenchmarkCase(name="concept", input="What is a DAG?", expected_keywords=["directed"]),
    ])
    
    report = await suite.arun(agent)
    print(f"Success Rate: {report.success_rate * 100:.1f}%")
    print(f"Mean Latency: {report.mean_latency_ms:.2f}ms")

asyncio.run(run_eval())
```
</details>

---

## 📂 Project Architecture

```text
mindlatticeai/
├── src/mindlatticeai/
│   ├── core/          # Agent facade & RuntimeConfig
│   ├── graph/         # DAG builder, concurrent execution engine, checkpoints
│   ├── planner/       # Intent parsing & adaptive strategy synthesis
│   ├── policy/        # Dynamic runtime policies & budget governors
│   ├── prompts/       # Prompt Genome registry, scoring, & mutation
│   ├── router/        # Multi-factor model router (latency, cost, prior)
│   ├── learner/       # Continuous learning loop & prior updater
│   ├── memory/        # 5-layer memory mesh (short, long, semantic, episodic, graph)
│   ├── providers/     # Local heuristic & live API providers (OpenAI, Gemini, etc.)
│   ├── reflection/    # Self-reflection critique & hallucination evaluator
│   ├── tools/         # Built-in safe tools & extensible registry
│   ├── cache/         # Exact & semantic similarity caches
│   ├── replay/        # Execution replay & snapshot serialization
│   ├── telemetry/     # Tracing, latency metrics, & DOT visualization
│   ├── benchmarks/    # Evaluation suite primitives
│   └── security/      # Sandboxing & security policies
├── tests/             # 18 Unit & Integration test suites
├── examples/          # Executable end-to-end usage examples
└── docs/              # In-depth architectural specifications & roadmap
```

---

## 🧪 Testing & Quality Assurance

MindLatticeAI comes with comprehensive test suites covering all architectural layers:

```bash
# Run complete test suite
pytest -v

# Run with test coverage
pytest --cov=mindlatticeai
```

---

## 🗺️ Roadmap & Research Directions

- [x] DAG Execution Engine with atomic checkpoints & retries
- [x] Evolvable Prompt Genomes with generational mutations
- [x] 5-Layer Multi-Layer Memory Mesh
- [x] Zero-dependency HTTP providers for OpenAI, Gemini, Claude, Ollama, DeepSeek
- [ ] Contextual Bandit reinforcement learning for model routing
- [ ] Vector database backend adapter (Qdrant, Chroma, Milvus)
- [ ] Distributed DAG workers via Redis / Celery
- [ ] Web dashboard for real-time trace, cost, and graph replay analysis

See the full [Roadmap Document](docs/roadmap.md) for details.

---

## 🤝 Contributing

Contributions from the AI research and engineering community are warmly welcome! Whether it's novel graph mutation algorithms, new providers, memory backends, or benchmark datasets:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/CognitiveFeature`)
3. Commit your Changes (`git commit -m 'Add CognitiveFeature'`)
4. Push to the Branch (`git push origin feature/CognitiveFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more details.

---

## ✍️ Citation

If you use **MindLatticeAI** in your academic research or production systems, please cite:

```bibtex
@software{mindlatticeai2026,
  author = {Raitul Hasan Priyo and Contributors},
  title = {MindLatticeAI: Adaptive AI Orchestration Framework with Execution-History Learning and Cognitive Graphs},
  url = {https://github.com/raitulh/MindLatticeAI},
  year = {2026}
}
```

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/raitulh">Raitul Hasan Priyo</a> and the MindLatticeAI community.</sub>
</div>
