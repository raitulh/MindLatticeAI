# MindLatticeAI Architecture

MindLatticeAI is a cognitive operating layer for AI applications. It does not treat orchestration as a static chain. It treats every request as an experiment whose result updates future planning, routing, prompting, memory, and policy decisions.

## Core Mission

```text
Request
  |
  v
Intent Parser
  |
  v
Planner: generate a candidate strategy
  |
  v
Policy Engine: enforce budgets, safety, routing preferences
  |
  v
Execution Graph Builder: compile strategy into a DAG
  |
  v
Graph Engine: execute tools, models, reflection, evaluation, learning
  |
  v
Memory + Replay + Telemetry
  |
  v
Response
```

## Dependency Flow

```text
schemas
  |
  +--> planner ----+
  +--> policy -----+--> core.Agent --> graph.Executor
  +--> prompts ----+       |             |
  +--> providers --+       |             +--> tools
  +--> memory -----+       |
  +--> learner ----+       +--> telemetry
  +--> reflection -+       +--> replay
  +--> cache ------+
  +--> plugins ----+
  +--> security ---+
```

The package keeps shared data models in `schemas.py` so subsystems can evolve independently while still passing validated execution artifacts.

## Class-Level Design

| Module | Main classes | Responsibility |
| --- | --- | --- |
| `core.agent` | `Agent` | Orchestrates the full experiment loop |
| `core.config` | `RuntimeConfig` | Runtime flags and storage paths |
| `planner.intent` | `IntentParser` | Infers task type, tools, risk, constraints |
| `planner.strategy` | `AdaptivePlanner` | Emits strategy steps and prompt genome choice |
| `policy.engine` | `PolicyEngine` | Applies budgets, tool controls, routing preferences |
| `graph.builder` | `ExecutionGraphBuilder` | Compiles strategy steps into DAG nodes and edges |
| `graph.engine` | `ExecutionGraphEngine`, `CheckpointStore` | Executes DAG nodes, retries, checkpoints |
| `router.model_router` | `ModelRouter` | Scores models by quality, latency, cost, availability, history |
| `tools.registry` | `Tool`, `ToolRegistry` | Registers and executes tools asynchronously |
| `memory.store` | `MemorySystem` | Short, long, semantic, episodic, and graph memory |
| `reflection.engine` | `ReflectionEngine` | Revises draft outputs and estimates hallucination risk |
| `reflection.evaluator` | `Evaluator` | Produces execution quality score |
| `learner.engine` | `LearningEngine` | Updates adaptive statistics from execution signals |
| `prompts.genome` | `PromptGenomeRegistry` | Selects, mutates, and scores prompt genomes |
| `telemetry.tracer` | `ExecutionTracer` | Captures traces and graph DOT visualization |
| `cache.stores` | `ResponseCache`, `SemanticCache` | Exact and lexical semantic caches |
| `replay.engine` | `ReplayEngine` | Stores replay snapshots for debugging |
| `plugins.hooks` | `PluginManager` | Emits customization hooks |
| `providers.*` | `ModelProvider`, provider classes | Local and external model providers |
| `security.manager` | `ApiKeyVault`, `SandboxPolicy` | Secrets and tool allow/deny checks |
| `benchmarks.suite` | `BenchmarkSuite` | Evaluates the agent across benchmark cases |

## Data Models

Pydantic models are used for:

- `AIRequest` and `AIResponse`
- `Intent`, `Strategy`, `PlanStep`, `PolicyDecision`
- `ExecutionGraph`, `GraphNode`, `GraphEdge`
- `ModelProfile`, `ModelResponse`
- `PromptGenome`
- `ToolResult`
- `ReflectionResult`, `EvaluationResult`
- `ExecutionMetrics`
- `MemoryRecord`
- `LearningSignal`
- `ExecutionTrace`, `TraceEvent`

These models make execution artifacts replayable, serializable, inspectable, and suitable for research datasets.

## Execution Graph Requirement

Each request is compiled into a DAG:

```text
optional tool nodes
  |
  v
model.main
  |
  v
reflect.self_check
  |
  v
evaluate.score
  |            |
  v            v
memory.write  learn.update
  |            |
  +------------+
       |
       v
final.response
```

Node types:

- `tool_call`
- `model_call`
- `reflection`
- `evaluation`
- `memory_write`
- `learning`
- `finalize`

The schema also defines `intent`, `plan`, and `policy` node types for deployments that want to compile the entire pre-execution phase into the graph.

Execution states:

- `pending`
- `running`
- `retrying`
- `checkpointed`
- `succeeded`
- `failed`
- `skipped`

Failure recovery:

- Nodes can retry up to their policy-approved retry limit.
- Successful nodes are checkpointed.
- A graph with failed required dependencies stops with a `GraphExecutionError`.
- Future durable stores can resume by graph id, node id, strategy id, and request fingerprint.

## Learning System

The learning loop records:

```text
latency_ms
cost_usd
quality_score
success
user_feedback
hallucination_probability
tool_efficiency_score
model_id
strategy_id
prompt_genome_id
```

It updates:

```text
strategy_stats
prompt_stats
model_stats
tool_stats
memory_importance_bias
```

The current implementation uses exponential moving averages. The interface can be upgraded to contextual bandits, Bayesian optimization, or offline reinforcement learning over replay traces.

## Prompt Genome

A prompt genome contains:

```text
role
instruction
context
constraints
examples
output_format
safety_rules
self_evaluation_rules
generation
score
metadata
```

Selection is score-based. Mutation creates a new generation with inherited behavior plus a change reason. Production systems can maintain a prompt population per task family.

## Memory System

Memory layers:

- Short-term: session-scoped working memory
- Long-term: persistent high-value outcomes
- Semantic: vector-like recall over content
- Episodic: execution event memory
- Knowledge graph: subject-predicate-object relationships

Memory operations:

- Decay: exponential importance decay
- Compression: archive low-priority short-term records
- Deduplication: keep highest-importance duplicate
- Conflict resolution: select most important and newest record
- Importance scoring: combine quality and hallucination risk

## Model Routing

The router scores candidates using:

- task compatibility
- cost constraint
- latency budget
- quality score
- historical model score
- availability
- policy preferences

This makes routing an adaptive decision rather than a static provider mapping.

## Plugin Hooks

Plugins may implement:

```text
before_plan()
after_plan()
before_execute()
after_execute()
before_reflect()
after_reflect()
before_learn()
after_learn()
```

The plugin manager invokes sync or async hooks.

## Observability

The telemetry system captures:

- execution traces
- graph visualization in DOT format
- cost tracking
- latency
- node failures
- strategy performance signals

The replay system stores request, response, graph, and trace snapshots for debugging and research datasets.

