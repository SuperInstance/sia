# sia

> Self-Improving Architecture: spectral orchestration for adaptive AI systems

## What This Does

SIA (Self-Improving Architecture) is a Python framework that orchestrates AI agents using spectral methods. It provides a central orchestrator that decomposes tasks, manages context windows, and applies spectral analysis to coordinate multi-agent workflows. The "spectral" variant uses eigenvalue-based optimization to allocate agent resources and detect task bottlenecks.

## The Key Idea

Traditional multi-agent systems use heuristics for task assignment. SIA treats the agent-task relationship as a spectral graph problem: agents are nodes, task dependencies are edges, and the eigenstructure of this graph reveals optimal task decomposition and agent specialization. The orchestrator continuously adapts using spectral feedback — if an agent's performance eigenvalue drops, the system redistributes work.

## Install

```bash
pip install sia
```

## Quick Start

```python
from sia.orchestrator import Orchestrator, OrchestratorConfig
from sia.context_manager import ContextManager
from sia.tasks import Task, TaskType

# Create orchestrator
config = OrchestratorConfig(
    max_agents=5,
    spectral_window=100,
    adaptation_rate=0.01,
)
orch = Orchestrator(config)

# Create a task
task = Task(
    name="analyze-data",
    task_type=TaskType.Analysis,
    description="Analyze the dataset",
    priority=1.0,
)

# Submit and run
result = orch.submit(task)
print(f"Result: {result}")

# Check spectral state
spectrum = orch.spectral_state()
print(f"Top eigenvalues: {spectrum.top_k(3)}")

# Use spectral orchestrator for adaptive routing
from sia.spectral_orchestrator import SpectralOrchestrator

spec_orch = SpectralOrchestrator(config)
spec_orch.adapt()  # Perform spectral adaptation
```

## API Reference

### `orchestrator`

| Type | Description |
|------|-------------|
| `OrchestratorConfig` | Configuration: `max_agents`, `spectral_window`, `adaptation_rate`. |
| `Orchestrator(config)` | Main orchestrator. Manages agent pool and task queue. |
| `submit(task)` | Submit a task, returns result after execution. |
| `spectral_state()` | Returns current spectral decomposition of agent-task graph. |
| `shutdown()` | Graceful shutdown of all agents. |

### `spectral_orchestrator`

| Type | Description |
|------|-------------|
| `SpectralOrchestrator(config)` | Extended orchestrator with spectral adaptation. |
| `adapt()` | Perform one spectral adaptation step — reassign agents based on eigenstructure. |
| `bottleneck_detect()` | Find task bottlenecks via spectral gap analysis. |

### `context_manager`

| Type | Description |
|------|-------------|
| `ContextManager(window_size)` | Manages sliding context window for agents. |
| `add(message)` | Add a message to the context. |
| `get_context()` | Get current context within window. |
| `compress()` | Spectrally compress context to retain key information. |

### `tasks`

| Type | Description |
|------|-------------|
| `Task` | Task with `name`, `task_type`, `description`, `priority`. |
| `TaskType` | Enum: `Analysis`, `Generation`, `Planning`, `Execution`. |
| `TaskResult` | Result with `output`, `metrics`, `agent_id`. |

## How It Works

1. **Task Submission**: Tasks enter a priority queue.
2. **Agent Assignment**: The orchestrator builds a bipartite graph (agents × tasks) and uses spectral clustering to assign agents to tasks.
3. **Execution**: Agents execute their assigned tasks, streaming results back.
4. **Spectral Feedback**: Performance metrics update the graph weights. The top eigenvalues track overall system health; spectral gaps indicate bottlenecks.
5. **Adaptation**: Periodically, the `SpectralOrchestrator` recomputes the eigenstructure and rebalances agent assignments.

## Testing

Tests covering:
- Orchestrator lifecycle (start, submit, shutdown)
- Spectral adaptation convergence
- Context manager windowing and compression
- Task priority ordering
- Spectral integration correctness

## License

MIT
