# Section 3: Threat Model and Problem Formalization

## 3.1 Provenance Events as Atomic Hyperedges

**Definition 1 (Provenance Hypergraph).**  
A provenance trace is a temporal hypergraph $\mathcal{H} = (\mathcal{V}, \mathcal{E}, \mathcal{T})$ where:
- $\mathcal{V}$ is the set of system entities (subjects, file objects, netflow objects, memory objects, ...).
- Each hyperedge $e \in \mathcal{E}$ corresponds to exactly one CDM event. It connects 2 or 3 nodes: the `subject`, the `predicateObject` (primary object), and optionally the `predicateObject2` (secondary object, e.g., memory region via `EVENT_MMAP`, shared memory via `EVENT_SHM`).
- $\mathcal{T} : \mathcal{E} \to \mathbb{R}^+$ assigns a nanosecond timestamp to each hyperedge.

**Empirical Grounding:** Across 44M Theia events, 97.2% are size‑2 (subject + object) and 2.8% (1.22M events) are genuine size‑3, where the secondary object is a distinct memory region or IPC channel. 

**Observation 1 (Attack Enrichment in 3‑Entity Hyperedges).**  
Attack events are **3.0× more likely** to involve genuine 3‑entity interactions than benign events (5.6% vs 1.9%). The secondary entity is almost exclusively a memory region linked via `EVENT_MMAP`—a system call heavily used in code injection, shared library loading, and process hollowing. When a pairwise graph decomposes an `EVENT_MMAP` into two edges—(process, file) and (process, memory region)—the critical fact that the process mapped *this specific file* into *that specific memory region* simultaneously is structurally lost.

**Concrete Example (from DARPA TC Theia data):**  
Consider two `EVENT_MMAP` events from the dataset. Both share a (subject, object) interaction structure but critically diverge in their `predicateObject2` (memory region).

* Malicious `EVENT_MMAP` (Label: Attack Stage):
  * Subject UUID: `B50E061A-0000-0000-0000-000000000020`
  * Object UUID: `B50E00F0-FF5B-6B7F-0000-000000000050`
  * Object2 UUID: `12000000-3242-0000-0000-000000000000` (Malicious Executable Memory Region)

* Benign `EVENT_MMAP` (Label: Background):
  * Subject UUID: `0F0DAC11-0000-0000-0000-000000000020`
  * Object UUID: `0F0D00D0-9FF2-1E7F-0000-000000000050`
  * Object2 UUID: `0100D00F-A66C-1800-0000-00003423B213` (Normal Shared Memory Segment)

Pairwise representations would yield identical edge types for the subject-object pairs, forcing the model to infer the context probabilistically. A native hyperedge maintains the discriminative 3-entity joint context natively.

---

# Section 4: The THyN Architecture

## 4.1 Continuous Entity State Bank
Instead of constructing static windowed graphs, THyN processes events in a continuous, chronological stream. To maintain temporal coherence across millions of events, we introduce the `EntityStateBank`, which stores an evolving state $s_v(t) \in \mathbb{R}^{d}$ for every entity $v \in \mathcal{V}$. 
Because the number of entities is bounded ($|\mathcal{V}| \approx 2.7M$ in Theia), the memory complexity is strictly bounded to $\mathcal{O}(|\mathcal{V}| \times d)$, requiring less than 3GB of VRAM in practice.

## 4.2 Hyperedge Aggregation (AllSetAggregator)
For an incoming hyperedge $e$ connecting entities $V_e = \{v_1, v_2, v_3\}$ (subject, object, object2), we first dynamically aggregate their states. The aggregation is conditioned on the semantic event features (event type, continuous features like size, and the process name).
We employ a dynamic attention mechanism where the event features act as the query, and the concatenated entity states and role embeddings (Subject, Object1, Object2) act as keys and values. This outputs a unified hyperedge representation $x_e$ that fuses both the graph topology and the semantic event payload.

## 4.3 Selective SSM State Update (E → V)
Once the hyperedge representation $x_e$ is computed, we selectively update the state of the participating entities using a Mamba-inspired State Space Model (SSM). 
For each participating entity $v \in V_e$:
1. **Time-Aware Discretization**: We explicitly model the elapsed time $\Delta t$ since the entity was last seen. We apply a gentle logarithmic scaling ($1 + 0.1 \log(1 + \Delta t)$) to the SSM's input-dependent discretization step $\Delta_i$, preventing massive gradient spikes from long temporal gaps.
2. **Role-Specific Propagation**: The input to the SSM is concatenated with the entity's role embedding within the hyperedge. This prevents identical updates from smoothing away the distinct graph structure (e.g., ensuring a subject receives a different state update than an object).
3. **HiPPO Initialization**: The state decay matrix $A$ uses a HiPPO-style initialization (log-spaced) to ensure stable, long-range memory retention across tens of thousands of steps.
4. **Gated Residuals**: A learned gating mechanism controls how much of the new SSM state replaces the old state, preventing catastrophic over-smoothing.

## 4.4 Model Training and Evaluation Regime
THyN is trained using Truncated Backpropagation Through Time (TBPTT). Entity states persist across event chunks within an epoch but are detached from the computational graph at chunk boundaries to bound memory.
For evaluation, we employ a **Warm Detection Regime** (Regime C): the model first processes historical training and validation data in a forward-only pass to "warm up" the Entity State Bank. This accurately mirrors a real-world continuous deployment, avoiding artificial "cold start" penalties during testing.
