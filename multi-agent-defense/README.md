# Multi-Agent Cooperative Defense: Comparative Study of Coordination Strategies
### Deep RL · DQN · PyTorch · Multi-Agent Systems · Pygame · SciPy

---

## Overview

This project compares three fundamentally different coordination strategies for multi-agent cooperative defense: tessellation-based geometric partitioning, optimization-based assignment (Hungarian algorithm), and deep reinforcement learning (DQN). All three approaches are evaluated in a shared 2D grid simulation where 20 autonomous agents must protect a central asset from continuously spawning threats.

The central question is **role allocation**: how should each agent decide whether to stay close to the asset and defend it, or move outward to intercept incoming threats? Each approach answers this differently. The tessellation approach assigns fixed roles based on geometric territory. The optimization approach dynamically reassigns roles every timestep using globally optimal matching. The RL approach learns a policy that produces emergent role specialization without any explicit assignment.

The RL agent significantly outperformed both rule-based methods, achieving 2.4x the survival time and 15x the threat neutralization rate of the random baseline. The key insight: the RL agent learned anticipatory positioning (placing itself in threat paths) rather than greedy pursuit (chasing threats from behind), which is fundamentally limited when agents and threats move at equal speed.

This was built as a course project for **Design and Control of Multi-Agent Systems** at UC Berkeley (Spring 2026).

---

## Key Findings

| Approach | Avg Survival | Avg Threats Neutralized | Role Allocation | Decentralized? |
|----------|-------------|------------------------|----------------|---------------|
| Random Baseline | 93.8 t | 6.4 | None | N/A |
| Tessellation | 108.2 t | 19.2 | Geometric (fixed) | Yes |
| Optimization | 106.8 t | 25.4 | Assignment (dynamic) | No |
| **RL (DQN)** | **222.8 t** | **97.9** | **Emergent** | **Yes*** |

\*Centralized training, decentralized execution.

---

## Environment

A 100x100 discrete grid with a 7x7 object-of-interest (OI) at the center (10 HP). Twenty defender agents protect the OI from threats that spawn at random boundary positions each timestep and move toward the OI. An agent neutralizes a threat by occupying the same cell. Both agents and threats move at one cell per timestep. The simulation runs for 500 timesteps or until the OI is destroyed.

Agents cannot enter OI cells or overlap with other agents. This collision avoidance constraint turned out to be the single most impactful design decision for the RL approach, as it forces parameter-sharing agents into different positions and enables behavioral diversity.

---

## Approaches

### 1. Tessellation-Based Coordination
Agents are deployed in two concentric rings (40% defenders, 60% interceptors). A tessellation partitions space into regions per agent. Defenders move toward their region centroid to maintain coverage. Interceptors pursue the nearest threat using greedy movement. Roles are fixed at deployment. Fully decentralized.

**Limitation:** Greedy pursuit fails at equal speeds. Interceptors that start behind a threat chase it all the way back to the OI without catching it.

### 2. Optimization-Based Coordination
At each timestep, a cost matrix (Manhattan distance from every agent to every threat) is constructed and solved using the Hungarian algorithm for optimal one-to-one matching. Assigned agents intercept, unassigned agents defend. Roles are dynamic and re-optimized every timestep. Centralized (requires global state).

**Advantage over tessellation:** Dynamic reassignment prevents sticky failed pursuits.  
**Limitation:** Swap-through problem. Agents and threats moving head-on on a discrete grid can pass through each other without neutralization.

### 3. Reinforcement Learning (DQN)
Each agent observes a local 21x21 grid window and selects from 5 actions (up/down/left/right/stay). A CNN-based Q-network processes one-hot encoded observations through two conv layers and outputs Q-values. All 20 agents share one network (parameter sharing). Training uses experience replay (200K buffer), a target network (updated every 2,000 steps), and epsilon-greedy exploration decaying over 250,000 steps.

**Training:** Centralized. All agents contribute to a shared replay buffer and a single network is updated.  
**Execution:** Decentralized. Each agent acts independently from its local observation only.

**Key finding:** Without explicit role assignment, agents naturally formed two concentric defense layers: inner agents defending and outer agents intercepting. This emerged purely from distance-scaled rewards and collision avoidance constraints.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core language |
| **PyTorch** | DQN agent (CNN Q-network, training pipeline) |
| **NumPy** | Environment logic, array operations |
| **SciPy** | Voronoi tessellation, Hungarian algorithm (linear_sum_assignment) |
| **Pygame** | Real-time visualization and figure generation |
| **Matplotlib** | Training curve plots |

---

## Project Structure

```
multi-agent-defense/
│
├── config.py                        - All tunable constants (grid size, agents, rewards, etc.)
├── environment.py                   - Shared grid environment (OI, threats, agents, collision)
│
├── rl_agent.py                      - DQN agent (Q-network, replay buffer, training logic)
├── train.py                         - RL training loop with logging and checkpointing
│
├── voronoi_controller.py            - Tessellation-based controller
├── optimization_controller.py       - Optimization-based controller (Hungarian assignment)
│
├── evaluate.py                      - Unified evaluation across all approaches (fixed seeds)
├── visualize.py                     - Real-time Pygame visualization (all modes)
│
├── assets/                       
│
└── README.md
```

---

## How to Run

**1. Clone and set up**
```bash
git clone https://github.com/yourusername/multi-agent-defense.git
cd multi-agent-defense
pip install torch numpy scipy pygame matplotlib
```

**2. Train the RL agent**
```bash
python train.py
```
Checkpoints are saved to `checkpoints/YYYYMMDD_HHMMSS/` with a config snapshot and training log.

**3. Evaluate all approaches**
```bash
python evaluate.py --mode all --checkpoint checkpoints/YYYYMMDD_HHMMSS/agent_best.pth
```
Runs random baseline, RL, tessellation, and optimization on 20 fixed-seed episodes and prints a comparison table.

**4. Visualize**
```bash
python visualize.py --mode rl --checkpoint checkpoints/YYYYMMDD_HHMMSS/agent_best.pth
python visualize.py --mode voronoi
python visualize.py --mode optimization
python visualize.py --mode random
```
Controls: SPACE (pause), R (reset), +/- (speed), ESC (quit).

---

## Training Details

The DQN agent was trained for 3,000 episodes on a 100x100 grid with 20 agents. Training takes approximately 2-3 hours on a consumer GPU.

**Reward function:**
- +0.01 per timestep (survival incentive)
- +2.0 per threat neutralized
- +0.1 x distance from OI at time of kill (encourages proactive interception)
- -2.0 per OI hit, -10.0 on OI destruction

**Hyperparameters:**
- Learning rate: 5e-4
- Discount factor: 0.99
- Replay buffer: 200,000
- Batch size: 64
- Target network update: every 2,000 steps
- Epsilon decay: 1.0 to 0.05 over 250,000 steps

The best checkpoint is saved automatically based on rolling average of threats neutralized. This is important because the agent exhibits catastrophic forgetting later in training (a known DQN issue), so the final checkpoint is not necessarily the best one.

---

## Notable Observations

**Emergent role specialization.** RL agents developed a two-layer defense structure (inner defenders, outer interceptors) without any explicit role assignment. This emerged from the distance-scaled reward and collision avoidance working together.

**Collision avoidance as coordination.** Without collision avoidance, parameter-sharing agents with similar observations took identical actions and stacked on the same cell, reducing 20 agents to effectively 3-4. Adding the constraint was the single biggest performance improvement.

**Greedy pursuit failure.** Both rule-based approaches use greedy movement (move toward target), which fundamentally fails when agent and threat speeds are equal. The RL agent discovered anticipatory positioning instead.

**Swap-through problem.** On a discrete grid with simultaneous movement, agents and threats moving head-on can exchange positions without ever sharing a cell. This affects all approaches but the RL agent learned to avoid it implicitly.

---

## References

1. Cortes, Martinez, Karatas, and Bullo. "Coverage optimization and spatial design of multi-robot systems." IEEE Trans. Robotics, 2004.
2. Kuhn. "The Hungarian method for the assignment problem." Naval Research Logistics Quarterly, 1955.
3. Samvelyan et al. "The StarCraft multi-agent challenge." AAMAS, 2019.
4. Mnih et al. "Human-level control through deep reinforcement learning." Nature, 2015.
5. Wang et al. "ROMA: Multi-agent reinforcement learning with emergent roles." ICML, 2020.
6. Chipade and Panagou. "Aerial swarm defense by stringnet herding." Frontiers in Robotics and AI, 2023.
