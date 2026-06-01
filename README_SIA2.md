# SIA² — Spectral Improvement Architecture

> *"Self-improvement is not hope. It's a Banach fixed point."*

Fork of [hexo-ai/sia](https://github.com/hexo-ai/sia) with mathematical foundations from the [SuperInstance ecosystem](https://github.com/SuperInstance).

## What Changed

Original SIA: Meta → Target → Feedback → repeat. Linear. No guarantees.

**SIA²** replaces the linear loop with a **spectral improvement loop**:

```
┌─────────────────────────────────────────────┐
│           SPECTRAL IMPROVEMENT LOOP          │
│                                             │
│  1. DECOMPOSE performance into eigenmodes   │
│  2. IDENTIFY weakest eigenmode              │
│  3. IMPROVE specifically that mode          │
│  4. VERIFY conservation laws hold           │
│  5. CHECK Banach convergence                │
│  6. PREDICT next generation via PDE         │
│  7. REPEAT or STOP (convergence guaranteed) │
└─────────────────────────────────────────────┘
```

## The 6 Mathematical Pillars

### 1. Banach Fixed Point Theorem → Guaranteed Convergence
The improvement operator T maps agent_n to agent_{n+1}. If ||T(x)-T(y)|| ≤ q·||x-y|| for q < 1, the sequence **MUST** converge. Not empirical. Provable. We compute q every generation.

### 2. Spectral Decomposition → Targeted Improvement
Agent performance is decomposed into eigenmodes via the capability correlation matrix. We find the weakest mode and improve specifically it. No wasted effort.

### 3. Information Geometry → Optimal Improvement Direction
The performance landscape is a Riemannian manifold. The Fisher information metric gives the true distance. The natural gradient (F⁻¹∇L) gives the steepest ascent direction. Regular gradient ignores curvature.

### 4. Conservation Laws → No Capability Lost
Noether's theorem: every symmetry produces a conserved quantity. We verify:
- **Capability conservation**: total score never decreases
- **Entropy bound**: improvement respects Landauer's principle  
- **Continuity**: no discontinuous jumps

### 5. PDE Dynamics → Performance Prediction
Agent improvement follows the heat equation on the performance manifold:
∂u/∂t = D·Δu + R(u)
This predicts next generation's performance before running it.

### 6. Renormalization Group → Universality Classification
Zoom out far enough, only a few types of improvement exist:
- **Gaussian**: at fixed point (converged)
- **Wilson-Fisher**: near phase transition (breakthrough imminent)
- **Asymptotic freedom**: gets better at getting better

## Usage

```bash
# Standard SIA (backward compatible)
sia --task gpqa --max_gen 5 --run_id 1

# SIA² mode — spectral analysis enabled
sia --task gpqa --max_gen 5 --run_id 1 --spectral
```

Output includes `spectral_trajectory.json` with full improvement trajectory.

## New Files

| File | Purpose |
|------|---------|
| `sia/spectral_orchestrator.py` | Core spectral analysis engine (34K LOC) |
| `sia/spectral_integration.py` | Hooks into original SIA loop |
| `runs/*/spectral_trajectory.json` | Full improvement trajectory |

## Multi-Language Implementations

| Language | Repo | Target |
|----------|------|--------|
| Python | This repo | Research, prototyping |
| Rust | [lau-sia2-engine](https://github.com/SuperInstance/lau-sia2-engine) | High-performance core |
| C99 | [lau-sia2-engine-c](https://github.com/SuperInstance/lau-sia2-engine-c) | Edge, embedded, Jetson |
| Go | [lau-sia2-engine-go](https://github.com/SuperInstance/lau-sia2-engine-go) | Cloud, K8s, microservices |
| WASM | [lau-sia2-engine-wasm](https://github.com/SuperInstance/lau-sia2-engine-wasm) | Browser, edge workers |

## The 14 Theorems Behind SIA²

| Theorem | SIA² Application |
|---------|-----------------|
| Kalman = Hodge | Optimal state estimation = harmonic forms |
| RL = Thermodynamics | Reward = negentropy, policy = equilibrium |
| Deadlock = H¹ | Performance bottlenecks = cohomology |
| Gradient = Fokker-Planck | Improvement = stochastic diffusion |
| Noether | Symmetries → conserved capabilities |
| CALM | Conservation-aware learning always monotone |
| Obs ⊣ Ctrl | Observation and control are adjoint |
| tr(id) | Identity of agent = trace of identity |
| Varadhan | Long-time behavior = large deviations |
| sunset = colimit | Agent lifecycle = categorical colimit |
| reward-hacking = H¹ | Gaming rewards = cohomology detection |
| policy = eigenfunction | Optimal policy = Laplacian eigenfunction |
| CALM = Noether | Conservation = symmetry (deep) |
| Landauer | Every bit of improvement costs kT ln 2 |

## Citation

Original SIA:
```bibtex
@article{hebbar2026sia,
  title   = {SIA: Self Improving AI with Harness \& Weight Updates},
  author  = {Hebbar, Prannay and Manawat, Yogendra and Verboomen, Samuel and Ivanova, Alesia and Palanimalai, Selvam and Bhatia, Kunal and Baskaran, Vignesh},
  journal = {arXiv preprint arXiv:2605.27276},
  year    = {2026}
}
```

SIA² Extensions:
```bibtex
@misc{sia2,
  title   = {SIA²: Spectral Improvement Architecture},
  author  = {SuperInstance},
  year    = {2026},
  url     = {https://github.com/SuperInstance/sia}
}
```

## License

MIT (inherits from original SIA)
