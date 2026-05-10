# Paper Blueprint: Temporal Hypergraph Networks for Provenance-Based Intrusion Detection

**Target:** NDSS Fall 2027 (Aug 19, 2026) | **Backup:** USENIX Security 2027 Cycle 2 (Jan 26, 2027)
**Status:** Design locked. Implementation phase begins.

---

## 1. Title and Thesis

**Title:** "Event-as-Hyperedge: Temporal Hypergraph Networks for Provenance-Based Campaign Detection"

**One-sentence thesis:** *We prove that provenance events are natively hyperedges, show that existing pairwise graph approaches fragment multi-entity interactions and lose critical attack semantics, and propose the first temporal hypergraph network that classifies individual system events as attack stages within multi-stage campaigns.*

---

## 2. Four Novel Contributions

| # | Contribution | Status |
|---|---|---|
| **C1** | **Formalization: Provenance events are atomic hyperedges.** We show that CDM events inherently link 3 entities (subject, object, predicateObject2), that pairwise decomposition fragments a single system call into 2 edges and loses the joint interaction context, and that prior provenance IDS (Flash, KAIROS, MAGIC) operate on this lossy pairwise representation. | ✅ Backed by Survey 1 & 2 & atomic hyperedge data |
| **C2** | **Architecture: Temporal Hypergraph Network with Mamba sequence encoding.** First model to combine (a) native hyperedge representation over provenance, (b) Mamba/SSM for intra-entity temporal sequence modeling, and (c) hypergraph convolution across shared entities for campaign-level coordination. | ✅ Novelty confirmed by Surveys 1 & 5 | 
| **C3** | **Task: Hyperedge-level campaign classification.** We define a new detection granularity where individual system events are classified as benign or as stages of a specific APT campaign. Unlike reconstruction-based anomaly detection (e.g., LOHA), our approach is supervised and provides forensic stage labels. | ✅ Differentiated from LOHA and autoencoder-based work |
| **C4** | **Empirical demonstration on DARPA TC against pairwise baselines.** We evaluate against KAIROS, E-GraphSAGE, and a temporal GNN, measuring Campaign Recall, F1, and latency. We also report a negative result: naive multi-event hyperedge grouping fails due to agglomeration, motivating our atomic approach. | ✅ Pipeline ready; baseline execution planned |

---

## 3. Threat Model and Problem Formalization

### 3.1 Provenance Events as Hyperedges

**Definition 1 (Provenance Hypergraph).** A provenance trace is a temporal hypergraph $\mathcal{H} = (\mathcal{V}, \mathcal{E}, \mathcal{T})$ where:
- $\mathcal{V}$ is the set of all system entities (subjects, file objects, netflow objects, memory objects, etc.).
- Each hyperedge $e \in \mathcal{E}$ corresponds to exactly one CDM event. It connects exactly three nodes: the subject, the predicateObject (primary object), and the predicateObject2 (secondary object, e.g., a memory region, IPC channel, or network buffer).
- $\mathcal{T}: \mathcal{E} \to \mathbb{R}^+$ assigns a nanosecond timestamp to each hyperedge.

**Pairwise Decomposition Loss.** When forced into a pairwise graph, a single hyperedge becomes two edges: (subject → object) and (subject → object2). This decomposition:
1. **Loses joint semantics:** The fact that the subject interacted with both objects *simultaneously* is not captured by either edge in isolation.
2. **Introduces spurious independence:** An attacker can craft sequences where the pairwise edges appear benign individually, while the triple interaction reveals malice.
3. **Inflates graph size:** The number of edges doubles, increasing computational cost and diluting the structural signal.

**Example (from DARPA TC ground truth).** An `EVENT_EXECUTE` call where Firefox executes `/home/admin/clean` with a shared library `libdrakon.so`. Pairwise produces two edges: (Firefox, EXECUTE, clean) and (clean, LOAD_LIB, libdrakon.so). Only the hyperedge (Firefox, EXECUTE, clean, libdrakon.so) captures that this is a single atomic action loading a malicious library into a dropped binary. A GNN must reconstruct this joint event from two disconnected edges and global readout—a structurally harder learning problem.

### 3.2 Attack Model

We target Advanced Persistent Threat (APT) campaigns that execute in multiple stages: initial compromise, payload delivery, privilege escalation, lateral movement, and exfiltration. Each stage manifests as a sequence of hyperedges sharing entities (the attacking process, the dropped file, the C2 socket). Our model classifies each hyperedge as belonging to a campaign stage or to benign background activity.

### 3.3 Why Pairwise IDS Fail

**Theorem (Informal):** Any graph neural network operating on pairwise event decompositions cannot distinguish two system events that share identical pairwise edge features but differ in their triple interaction.

This follows trivially from the fact that pairwise decomposition is a surjective mapping from $\mathcal{E}$ (hyperedges) to $\mathcal{E}_{\text{pair}}$ (edge pairs), and GNNs' message passing is invariant to the joint distribution of edges sharing a node beyond what the pairwise features encode. Since CDM events have rich `predicateObject2` attributes (memory addresses, socket parameters, file descriptors), the hyperedge carries strictly more information than the sum of its constituent pairwise edges.

*We acknowledge this is a direct consequence of the data representation, not a deep mathematical theorem. The contribution is the empirical demonstration that this representational loss matters for real APT detection.*

---

## 4. Architecture: THyN (Temporal Hypergraph Network)

### 4.1 Overview

```
┌─────────────────────────────────┐
│     Raw CDM Events (Avro)       │
└───────────┬─────────────────────┘
            │ Parse & extract
            ▼
┌─────────────────────────────────┐
│   Atomic Hyperedges (size-3)    │  ← Each event = one hyperedge
│   4.7M hyperedges / shard        │
└───────────┬─────────────────────┘
            │ Feature extraction
            ▼
┌─────────────────────────────────┐
│   Hyperedge Features            │  Event type embedding, timestamp,
│   + Entity Lookup               │  subject/object UUIDs, event fields
└───────────┬─────────────────────┘
            │ Group by subject, order by time
            ▼
┌─────────────────────────────────┐
│   Temporal Sequence Encoder     │  Mamba SSM (or GRU) per subject
│   (per-entity chain)            │  Captures sequential behavior
└───────────┬─────────────────────┘
            │ Hyperedge embeddings
            ▼
┌─────────────────────────────────┐
│   Hypergraph Convolution        │  Message passing across shared
│   (across entities)             │  objects; hyperedges that share
│                                  │  entities exchange information
└───────────┬─────────────────────┘
            │ Final hyperedge representations
            ▼
┌─────────────────────────────────┐
│   Classification Head           │  FC → Sigmoid: benign vs. attack
│                                  │  FC → Softmax: campaign stage ID
└─────────────────────────────────┘
```

### 4.2 Atomic Hyperedge Construction (from CDM)

Each CDM event record maps directly to a hyperedge:
```python
hyperedge = {
    'id': event.uuid,
    'nodes': [event.subject_uuid, event.predicateObject_uuid, event.predicateObject2_uuid],
    'timestamp': event.timestampNanos,
    'type': event.type,  # EVENT_READ, EVENT_EXECUTE, etc.
    'features': [event.size, event.flags, ...],  # domain-specific parameters
}
```
No clustering, no windowing, no heuristic grouping. This is deterministic, lossless, and reproducible across any CDM-compliant provenance trace.

### 4.3 Event-Type Embedding

Each of the ~40 CDM event types is mapped to a learnable dense vector $\mathbf{e}_{\text{type}} \in \mathbb{R}^{d_t}$. Additional scalar features (event size, flags) are projected through a linear layer. The initial hyperedge feature is:
$$\mathbf{h}_e^{(0)} = \mathbf{e}_{\text{type}} \oplus \text{MLP}([size, flags, timestamp\_feat])$$
where $\oplus$ is concatenation and timestamp features include hour-of-day, day-of-week, and time since previous hyperedge on the same subject.

### 4.4 Temporal Sequence Encoder (Mamba)

For each subject entity $v$, we collect all hyperedges where $v$ is the subject, ordered by timestamp. This forms a sequence of hyperedge features $(\mathbf{h}_{e_1}^{(0)}, \mathbf{h}_{e_2}^{(0)}, ..., \mathbf{h}_{e_T}^{(0)})$.

We feed this sequence into a Mamba state-space model:
$$(\mathbf{z}_{e_1}, ..., \mathbf{z}_{e_T}) = \text{Mamba}(\mathbf{h}_{e_1}^{(0)}, ..., \mathbf{h}_{e_T}^{(0)})$$
where each $\mathbf{z}_{e_t} \in \mathbb{R}^{d_h}$ is the temporally contextualized representation of hyperedge $e_t$, now aware of the subject's behavioral history.

**Why Mamba:** Provenance traces can have hundreds of thousands of events per subject. Transformer attention is quadratic. Mamba's linear-time recurrence handles long sequences efficiently. We ablate against GRU to test if the SSM's selective state provides benefit over simpler recurrent models.

### 4.5 Hypergraph Convolution

Some subjects produce purely benign event sequences; others interleave benign and malicious events. To capture coordination across entities (e.g., the same file object being read by a benign process and then written by a malicious one), we perform hypergraph convolution.

We construct the hypergraph's incidence matrix $\mathbf{H} \in \{0,1\}^{|\mathcal{V}| \times |\mathcal{E}|}$ where $\mathbf{H}_{v,e}=1$ if entity $v$ participates in hyperedge $e$. We apply two rounds of hypergraph convolution:

$$\mathbf{h}_e^{(\ell+1)} = \sigma\left(\mathbf{W}^{(\ell)} \mathbf{h}_e^{(\ell)} + \sum_{v \in e} \sum_{e' \ni v} \alpha_{v,e'} \mathbf{U}^{(\ell)} \mathbf{z}_{e'}\right)$$

where $\alpha$ is a learned attention weight over incident hyperedges, and $\mathbf{z}_{e'}$ is the Mamba output for hyperedge $e'$. This allows information to flow between hyperedges that share entities—for instance, connecting the Firefox exploit hyperedge with the clean C2 callback hyperedge through the shared `/home/admin/clean` file.

### 4.6 Classification Head

The final hyperedge representation $\mathbf{h}_e^{(L)}$ is passed through:
- A binary classifier: $\hat{y}_e = \sigma(\mathbf{w}_b^T \mathbf{h}_e^{(L)})$ for attack vs. benign.
- Optionally, a multi-class head for campaign stage identification (exploit, delivery, C2, exfil, etc.).

### 4.7 Loss Function

Weighted binary cross-entropy to handle class imbalance:
$$\mathcal{L} = -\frac{1}{|\mathcal{E}|} \sum_{e} \left[ w_1 y_e \log \hat{y}_e + w_0 (1-y_e) \log (1-\hat{y}_e) \right]$$
with $w_1 = \frac{|\mathcal{E}_{\text{benign}}|}{|\mathcal{E}|}$ and $w_0 = \frac{|\mathcal{E}_{\text{attack}}|}{|\mathcal{E}|}$.

### 4.8 Complexity

- Mamba encoding: $O(E \cdot d_h^2)$ where $E$ is total hyperedges, linear in sequence length.
- Hypergraph convolution: $O(|\mathcal{E}| \cdot k \cdot d_h)$ where $k$ is average node degree in the hypergraph ($k \approx 3$ for our data).
- Total: linear in the number of events, scalable to production provenance volumes.

---

## 5. Experimental Design

### 5.1 Datasets

| Dataset | Role | Description |
|---------|------|-------------|
| **DARPA TC E3 Theia** | Primary | 44M events, 3 attack scenarios, full CDM schema |
| **DARPA TC E5 Theia** | Extension | Larger, multi-day, additional attack types |
| **CICIDS-2017** | Secondary | Flow-level data for generalization test |
| **Synthetic SCA** | Controlled | Perfect stealth attacks injected into benign background |

### 5.2 Baselines

| Baseline | Type | Why |
|----------|------|-----|
| **KAIROS** (S&P 2024) | Provenance GNN | State-of-the-art provenance IDS using pairwise graphs |
| **E-GraphSAGE** | Temporal GNN | Edge-level GNN with temporal features |
| **TGN** | Temporal GNN | Continuous-time dynamic graph |
| **Random Forest** | ML baseline | Handcrafted features on hyperedges (from sanity check) |
| **GRU-only** (ablation) | Our model minus hypergraph convolution | Tests necessity of cross-entity message passing |
| **No-Mamba** (ablation) | Our model with GRU instead of Mamba | Tests SSM benefit |

### 5.3 Metrics

| Metric | Definition | Why |
|--------|-----------|-----|
| **Campaign Recall** | Fraction of campaigns where ALL stages are detected | Operational requirement |
| **F1 (attack class)** | Harmonic mean on minority class | Handles imbalance |
| **AUPRC** | Area under precision-recall curve | Better for imbalanced data than ROC |
| **Detection Latency** | Time from first attack event to first alert | SOC relevance |
| **Throughput** | Events/second | Production feasibility |

### 5.4 Ablation Studies

1. **Atomic vs. naive grouped hyperedges:** Demonstrate that our deterministic atomic construction outperforms the windowed agglomeration approach (report the negative result from Survey 5).
2. **Mamba vs. GRU vs. Transformer:** Compare sequence encoders.
3. **With/without hypergraph convolution:** Show the value of cross-entity message passing.
4. **Hyperedge classification vs. node classification vs. graph classification:** Prove our task granularity is optimal.

### 5.5 Pre-Registered Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| Campaign Recall improvement over KAIROS | > 15% absolute |
| F1 on attack hyperedges | > 0.75 |
| False positive rate (per day) | < 100 alerts |
| No single feature dominates detection | Confirmed by feature importance analysis |

---

## 6. Related Work Positioning

### 6.1 Provenance-Based IDS

- **Flash** (S&P 2024): Pairwise heterogeneous GNN on provenance. Validates the importance of provenance, but limited to dyadic edges.
- **KAIROS** (S&P 2024): Temporal graph alignment on provenance. Masters the temporal dimension, but uses pairwise graphs.
- **MAGIC** (USENIX 2024): Masked graph representation learning for APT detection. Self-supervised approach on standard graphs.
- **Our contribution:** All prior provenance IDS operate on pairwise graphs. We are the first to model provenance events as native hyperedges, preserving multi-entity interactions and achieving superior campaign detection.

### 6.2 Hypergraph-Based IDS

- **APT Detection via HyperGAT** (Song et al., 2025): Hypergraph on provenance, but uses statistical community mining (LFM) and treats hyperedges as behavioral clusters, not atomic events. Lacks temporal dynamics.
- **HRNN** (Yang et al., 2024): Temporal hypergraph on network flows, not provenance. Uses KNN for hyperedge construction—unsuitable for deterministic causal tracking.
- **LOHA** (Chen et al., 2025): Hypergraph masked autoencoder for APT detection on network traffic. Uses Louvain community detection, positional encodings, and reconstruction-based anomaly detection. Differs from our work in data domain (network vs. host), hypergraph construction (statistical vs. deterministic), temporal modeling (snapshot vs. continuous sequence), and task (anomaly scoring vs. campaign classification).
- **Our contribution:** First to combine deterministically constructed hyperedges from provenance events with continuous temporal sequence modeling and supervised campaign classification.

### 6.3 The (a)(b)(c) Framing

> "While coordination features have been used heuristically in intrusion detection, no prior work has (a) formalized provenance events as atomic hyperedges and proven that pairwise decomposition loses multi-entity semantics, (b) designed a temporal hypergraph network that natively operates on this representation, and (c) demonstrated that this approach detects multi-stage APT campaigns that pairwise provenance GNNs miss."

---

## 7. Paper Section Outline (12 pages, NDSS format)

| Section | Pages | Content |
|---------|-------|---------|
| §1 Introduction | 1.5 | APT campaigns → provenance → pairwise limitation → our thesis → contributions |
| §2 Background | 1.0 | CDM provenance schema, hypergraph definitions, Mamba/SSM basics |
| §3 Threat Model & Problem | 1.5 | Definition 1 (provenance hypergraph), pairwise decomposition loss proof, motivation examples from DARPA TC |
| §4 Architecture | 2.5 | Sections 4.1-4.8 from this blueprint, with diagrams |
| §5 Evaluation | 3.0 | Experimental setup, main results, campaign recall, ablation studies, negative result on grouped hyperedges |
| §6 Discussion | 0.5 | Limitations, adversarial robustness, deployment considerations |
| §7 Related Work | 1.0 | Provenance IDS (Flash, KAIROS, MAGIC), hypergraph NIDS (HRNN, LOHA, HyperGAT), temporal GNNs |
| §8 Conclusion | 0.5 | Summary, future work |

---

## 8. Implementation Roadmap (14 weeks to NDSS deadline)

| Weeks | Phase | Key Deliverables |
|-------|-------|------------------|
| **1** (May 12-18) | **Data pipeline finalization** | Atomic hyperedge extraction from all Theia shards, feature engineering complete |
| **2** (May 19-25) | **Mamba encoder implementation** | Per-subject sequence grouping, Mamba model training on hyperedge sequences |
| **3** (May 26-Jun 1) | **Hypergraph convolution** | Incidence matrix construction, message passing implementation |
| **4-5** (Jun 2-15) | **Full model integration** | End-to-end THyN pipeline, training loop, hyperparameter tuning |
| **6** (Jun 16-22) | **Baseline implementation** | KAIROS reproduction, E-GraphSAGE, TGN setup on our data |
| **7-8** (Jun 23-Jul 6) | **Main evaluation** | All metrics on all datasets, campaign recall computation |
| **9** (Jul 7-13) | **Ablation studies** | Mamba vs. GRU, atomic vs. grouped, convolution on/off |
| **10** (Jul 14-20) | **Paper writing: §§2-5** | Draft technical sections |
| **11** (Jul 21-27) | **Paper writing: §§1,6,7,8** | Draft intro, discussion, related work, conclusion |
| **12** (Jul 28-Aug 3) | **Adversarial robustness** | Mimicry attack evaluation, noise injection |
| **13** (Aug 4-10) | **Professor review** | Full draft to professor, 1 week for feedback |
| **14** (Aug 11-17) | **Final revision** | Incorporate feedback, polish, submit Aug 19 |

**Backup plan:** If any week slips by >3 days, switch to S&P Cycle 2 (Nov 17) for 13 extra weeks of buffer.

---

## 9. Project Repository Structure

```
thyne-nids/
├── src/
│   ├── data/
│   │   ├── darpa_parser.py        # Avro → atomic hyperedges
│   │   ├── hypergraph_builder.py  # Incidence matrix construction
│   │   └── feature_extractor.py   # Hyperedge feature engineering
│   ├── model/
│   │   ├── mamba_encoder.py       # Mamba SSM sequence model
│   │   ├── hypergraph_conv.py     # Hypergraph convolution layer
│   │   ├── thyne.py               # Full THyN model
│   │   └── classifier.py          # Detection heads
│   ├── baselines/
│   │   ├── kairos_runner.py
│   │   ├── e_graphsage.py
│   │   └── tgn_runner.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── campaign_recall.py
│   └── utils/
├── experiments/
│   └── configs/
├── paper/
│   └── main.tex
└── data/  (gitignored)
```