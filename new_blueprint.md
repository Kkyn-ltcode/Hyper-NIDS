# Event-as-Hyperedge: Temporal Hypergraph Networks for Provenance-Based Campaign Detection  
**Final Blueprint v2 — NDSS Fall 2027 (Aug 19, 2026) | Backup: USENIX Security 2027 Cycle 2**

---

## 1. Title and Thesis

**Title:** *Event-as-Hyperedge: Temporal Hypergraph Networks for Provenance-Based Campaign Detection*

**One-sentence thesis:** *We prove that provenance events are natively hyperedges linking three entities, show that pairwise graph representations fragment these multi-entity interactions and lose critical attack semantics, and propose the first temporal hypergraph network that classifies individual system events as stages within advanced persistent threat campaigns.*

---

## 2. Four Clean Contributions (Updated)

| # | Contribution | Proof / Status |
|---|---|---|
| **C1** | **Formalization and enrichment: Provenance events are variable‑arity hyperedges, and attacks are enriched in 3‑entity interactions.** We show that CDM events naturally form size‑2 (subject + object) or size‑3 (subject + object + secondary object) hyperedges. While only 2.8% of events are size‑3, **attack events are 3× more likely to involve 3‑entity interactions** (5.6% vs 1.9%), driven by `EVENT_MMAP` (code injection, library loading). Pairwise graphs decompose these into two edges, losing the joint semantics. Prior provenance IDS (Flash, KAIROS, MAGIC) operate on this lossy representation. | ✅ Validated: 44M events, 1.22M genuine size‑3, 3× attack enrichment |
| **C2** | **Architecture: Temporal Hypergraph Network (THyN) with Mamba sequence encoding.** First model to combine (a) native variable‑arity hyperedge representation over provenance, (b) Mamba/SSM for per‑entity temporal sequence modeling, and (c) hypergraph convolution over the bipartite entity‑hyperedge graph to capture cross‑entity coordination. | ✅ Novelty confirmed by Surveys 1 & 5, RF temporal features justify Mamba |
| **C3** | **Task: Hyperedge‑level campaign‑stage classification.** We define a new detection granularity where individual system events are classified as benign or as belonging to specific APT campaign stages (initial compromise, payload drop, escalation, C2, reconnaissance). Unlike reconstruction‑based anomaly detection (LOHA), our approach is supervised and provides forensic stage labels. | ✅ Differentiated from LOHA; stage mapping from DARPA ground truth complete |
| **C4** | **Empirical demonstration on DARPA TC against pairwise baselines, with multi‑granularity evaluation.** We evaluate on Theia and TRACE datasets, comparing against KAIROS, E‑GraphSAGE, and TGN. We report Stage‑Level Campaign Recall, Size‑3 Recall, AUPRC, and F1 at three label granularities (broad ~24%, narrow ~1–3%, IoC‑only <0.1%). We also report the negative result that naive window‑based hyperedge grouping fails due to event agglomeration, justifying atomic construction. | ✅ Pipeline ready; TRACE data pending (est. 3 days); baselines planned |

---

## 3. Threat Model and Problem Formalization

### 3.1 Provenance Events as Atomic Hyperedges

**Definition 1 (Provenance Hypergraph).**  
A provenance trace is a temporal hypergraph \(\mathcal{H} = (\mathcal{V}, \mathcal{E}, \mathcal{T})\) where:
- \(\mathcal{V}\) is the set of system entities (subjects, file objects, netflow objects, memory objects, ...).
- Each hyperedge \(e \in \mathcal{E}\) corresponds to exactly one CDM event. It connects 2 or 3 nodes: the `subject`, the `predicateObject` (primary object), and optionally the `predicateObject2` (secondary object, e.g., memory region via `EVENT_MMAP`, shared memory via `EVENT_SHM`).
- \(\mathcal{T} : \mathcal{E} \to \mathbb{R}^+\) assigns a nanosecond timestamp to each hyperedge.

**Empirical grounding:** Across 44M Theia events, 97.2% are size‑2 (subject + object) and 2.8% (1.22M events) are genuine size‑3, where the secondary object is a distinct memory region or IPC channel. The CDM schema records a null UUID sentinel when no secondary object exists; we filter this to obtain honest hyperedge arities.

**Observation 1 (Attack Enrichment in 3‑Entity Hyperedges).**  
Attack events are **3.0× more likely** to involve genuine 3‑entity interactions than benign events (5.6% vs 1.9%). The secondary entity is almost exclusively a memory region linked via `EVENT_MMAP`—a system call heavily used in code injection, shared library loading, and process hollowing. When a pairwise graph decomposes an `EVENT_MMAP` into two edges—(process, file) and (process, memory region)—the critical fact that the process mapped *this specific file* into *that specific memory region* simultaneously is structurally lost.

This enrichment is not incidental. Attack techniques such as reflective DLL injection, code‑cave implantation, and shared‑memory IPC inherently require multi‑entity coordination. Our hypergraph representation captures this naturally; pairwise graphs cannot.

**Concrete example (from Theia data):**  
Two `EVENT_MMAP` events share the same subject (Firefox) and primary object (a library file), but one has `predicateObject2` pointing to a suspicious executable memory region (attack) and the other to a normal shared memory segment (benign). Pairwise edges would be identical; only the hyperedge retains the discriminative secondary object.

> *Note: In the paper we will insert the exact event UUIDs and object paths from the dataset.*

### 3.2 Attack Model

We target Advanced Persistent Threat (APT) campaigns that execute in multiple stages: initial compromise, payload delivery, privilege escalation, lateral movement, C2 communication, and exfiltration. Each stage manifests as a sequence of hyperedges sharing entities (e.g., the same malicious process, dropped binary, or C2 socket). Our model classifies each hyperedge as belonging to a campaign stage or to benign background activity.

### 3.3 Why Pairwise IDS Fail

**Observation 2 (Limitation of Pairwise GNNs).**  
Any graph neural network operating on pairwise event decompositions cannot distinguish two system events that share identical (subject, object) pairwise features but differ in their `predicateObject2`. This follows directly from the fact that pairwise decomposition is a surjective mapping from hyperedges to edge pairs, and GNN message passing is invariant to the joint distribution of edges beyond what the pairwise features encode. The representational loss forces downstream models to reconstruct multi‑entity semantics from fragmented evidence, a structurally harder learning problem that benefits attackers.

---

## 4. The THyN Architecture

### 4.1 Overview

```
Raw CDM Events (Avro)
        │ Parse & extract
        ▼
Atomic Hyperedges (size‑3)  ← 4.7M hyperedges / shard
        │ Feature extraction
        ▼
Hyperedge Features + Entity Lookup
        │ Group by subject, order by time
        ▼
Temporal Sequence Encoder (Mamba SSM)  ← per‑subject event chains
        ▼
Hyperedge embeddings
        │
        ▼
Hypergraph Convolution (bipartite E‑V graph)
        │ Message passing across shared entities
        ▼
Final hyperedge representations
        │
        ▼
Classification Head
   ├── Sigmoid: benign vs. attack
   └── Softmax: campaign stage ID
```

### 4.2 Atomic Hyperedge Construction

Each CDM event record directly becomes a hyperedge; no clustering, no windowing, no heuristics.

```python
hyperedge = {
    'id': event.uuid,
    'nodes': [event.subject_uuid, event.predicateObject_uuid, event.predicateObject2_uuid],
    'timestamp': event.timestampNanos,
    'type': event.type,                # e.g., EVENT_READ, EVENT_EXECUTE
    'features': [event.size, event.flags, ...]
}
```

### 4.3 Event‑Type Embedding

Each of the ~40 CDM event types is mapped to a learnable dense vector \(\mathbf{e}_{\text{type}} \in \mathbb{R}^{d_t}\). Additional scalar features (size, flags, timestamp features) are projected through a small MLP. The initial hyperedge feature is:

\[
\mathbf{h}_e^{(0)} = \mathbf{e}_{\text{type}} \;\oplus\; \text{MLP}\bigl([\text{size}, \text{flags}, \text{hour\_sin}, \text{hour\_cos}, \text{day\_of\_week}, \text{time\_since\_prev\_subject\_event}]\bigr)
\]

### 4.4 Temporal Sequence Encoder (Mamba)

For each subject entity \(v\), we collect all hyperedges where \(v\) is the subject, ordered by timestamp. This forms a sequence \((\mathbf{h}_{e_1}^{(0)}, \mathbf{h}_{e_2}^{(0)}, \dots, \mathbf{h}_{e_T}^{(0)})\).

We use a Mamba state‑space model:

\[
(\mathbf{z}_{e_1}, \dots, \mathbf{z}_{e_T}) = \text{Mamba}(\mathbf{h}_{e_1}^{(0)}, \dots, \mathbf{h}_{e_T}^{(0)})
\]

Each \(\mathbf{z}_{e_t} \in \mathbb{R}^{d_h}\) is now temporally contextualized, aware of the subject's behavioural history.

**Why Mamba:**  
- RF feature importance shows that temporal features (`subject_is_new`, `time_gap_same_subject`) dominate, but they only capture pairwise intervals. Mamba can learn multi‑step sequential patterns (e.g., write→execute→connect) that signify attack stages.  
- Linear \(O(T)\) complexity vs. quadratic attention; critical for subjects with tens of thousands of events.  
- Ablation with GRU tests the benefit of the selective state‑space mechanism.

### 4.5 Hypergraph Convolution (Bipartite Formulation)

We construct the hypergraph incidence matrix \(\mathbf{H} \in \{0,1\}^{|\mathcal{V}| \times |\mathcal{E}|}\) where \(\mathbf{H}_{v,e}=1\) if entity \(v\) participates in hyperedge \(e\).

**Acknowledgment of equivalence:**  
The hypergraph convolution defined below is algebraically equivalent to message passing on the bipartite graph between entities \(\mathcal{V}\) and hyperedges \(\mathcal{E}\). Our contribution is not a new convolution operator, but the systematic mapping of CDM provenance events to this bipartite structure, combined with Mamba‑based temporal encoding on entity‑centric event chains. Standard temporal GNNs (TGN, TGAT) cannot natively handle variable‑arity relations (size‑3 interactions) in a single time‑stamped message; our formulation resolves this.

**Convolution operation** (two rounds):

\[
\mathbf{h}_e^{(\ell+1)} = \sigma\!\left( \mathbf{W}^{(\ell)} \mathbf{h}_e^{(\ell)} \;+\; \sum_{v \in e} \;\sum_{e' \ni v} \alpha_{v,e'} \cdot \mathbf{U}^{(\ell)} \mathbf{z}_{e'} \right)
\]

where:
- \(\mathbf{z}_{e'}\) is the Mamba output for hyperedge \(e'\).
- \(\alpha_{v,e'}\) is a learned attention weight over incident hyperedges.
- \(\mathbf{W}^{(\ell)}, \mathbf{U}^{(\ell)}\) are layer parameters.

This allows information to flow between hyperedges that share entities—e.g., connecting the initial Firefox exploit hyperedge with the subsequent `clean` process C2 callback through the shared `/home/admin/clean` file.

### 4.6 Classification Head

The final hyperedge representation \(\mathbf{h}_e^{(L)}\) is fed into:

- **Binary classifier:** \(\hat{y}_e = \sigma(\mathbf{w}_b^T \mathbf{h}_e^{(L)})\)  (attack vs. benign)
- **Multi‑class stage classifier:** \(\hat{\mathbf{y}}_e = \text{softmax}(\mathbf{W}_s \mathbf{h}_e^{(L)})\)  (stage 1, 2, ..., 5 or benign)

### 4.7 Loss Function

Weighted binary cross‑entropy for attack/benign; weighted categorical cross‑entropy for stage classification. Class weights are inversely proportional to class frequency to handle heavy imbalance.

### 4.8 Complexity

- Mamba encoding: \(O(E \cdot d_h^2)\), linear in number of events per subject.
- Hypergraph convolution: \(O(|\mathcal{E}| \cdot \bar{k} \cdot d_h)\), where \(\bar{k}\) is average node degree (\(\approx 3\) for our data).
- **Total:** linear in total number of events, scalable to production provenance volumes.

---

## 5. Experimental Design

### 5.1 Datasets

| Dataset | Role | Size | Rationale |
|---------|------|------|-----------|
| **DARPA TC E3 Theia** | Primary | 44M events, 3 campaigns | Standard provenance benchmark; rich event types; ground truth mapped |
| **DARPA TC E3 TRACE** | Co‑primary | ~22M events, different attacks | Different Linux host, different benign background (Nginx, etc.) — mitigates Firefox confound |
| **DARPA TC E3 CADETS** | Cross‑OS extension | (deferred) | FreeBSD; validates cross‑platform generalisation |

*Note: CICIDS‑2017 is excluded because it lacks CDM schema (no subjects/objects/predicateObject2). Cross‑dataset generalisation within CDM (Theia → TRACE) is a stronger claim.*

### 5.2 Baselines

| Baseline | Type | Significance |
|----------|------|--------------|
| **KAIROS** (S&P 2024) | Temporal provenance GNN | State‑of‑the‑art pairwise provenance IDS |
| **E‑GraphSAGE** | Edge‑level temporal GNN | Pairwise representation with temporal features |
| **TGN** | Continuous‑time dynamic GNN | Handles temporal edges, but only pairwise |
| **Random Forest** | Handcrafted features on hyperedges | Strong non‑neural baseline; already achieved AUC 0.89 |
| **THyN‑GRU** (ablation) | Our model with GRU instead of Mamba | Tests SSM benefit |
| **THyN‑noConv** (ablation) | Our model without hypergraph convolution | Tests necessity of cross‑entity message passing |

### 5.3 Evaluation Metrics

| Metric | Definition | Why |
|--------|-----------|-----|
| **Stage‑Level Campaign Recall** | Fraction of ground‑truth attack stages detected per campaign, averaged across campaigns | Fine‑grained, statistically meaningful measure of campaign completeness |
| **AUPRC** | Area under precision‑recall curve | Robust to class imbalance; primary quantitative metric |
| **F1 (attack class)** | Harmonic mean for minority class | Reported at three granularities (broad, narrow, IoC‑only) |
| **Size‑3 Recall** | Recall on genuine 3‑entity attack hyperedges (the 5.6% subset) | Directly tests whether the hypergraph captures the signal it is designed to capture |
| **Detection Latency** | Time from first stage event to first alert | Operational relevance |
| **Throughput** | Events/second | Deployment feasibility |

**Stage‑Level Campaign Recall definition:**
For each campaign, the ground truth enumerates distinct attack stages (e.g., Stage 1: exploit delivery, Stage 2: payload drop, …). A stage is considered *detected* if all of its constituent hyperedges (or a majority, say >75%) are correctly classified as attack. The metric is the fraction of stages detected, averaged over all campaigns. This yields a continuous value with substantially greater statistical power than the original binary campaign‑level metric.

### 5.4 Multi‑Granularity Labeling

| Granularity | Attack % | Labeled Entities |
|-------------|----------|------------------|
| **Broad** | ~24% | All events involving the Firefox attack process tree |
| **Narrow** | ~1–3% | Only events directly part of exploit, payload, C2, or escalation |
| **IoC‑only** | <0.1% | Only events interacting with known‑malicious IPs/files |

F1 and AUPRC are reported at all three levels, demonstrating graceful degradation towards realistic imbalance.

### 5.5 Ablation Studies

1. **Atomic vs. naive grouped hyperedges** — report the negative result from Survey 5; show that our deterministic atomic construction is necessary.
2. **Mamba vs. GRU vs. Transformer** — compare sequence encoders.
3. **With/without hypergraph convolution** — demonstrate value of cross‑entity coordination.
4. **Hyperedge classification vs. node classification vs. graph classification** — justify our task granularity.

### 5.6 Pre‑Registered Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| Stage‑Level Campaign Recall (broad) | > 0.80 |
| Stage‑Level Campaign Recall (narrow) | > 0.60 |
| AUPRC (broad) | > 0.90 |
| F1 (narrow) | > 0.60 |
| Significant improvement over KAIROS | p < 0.05 on AUPRC |
| No single feature dominates learned representation | Verified by integrated gradients / ablation |

### 5.7 Adversarial Robustness

We evaluate against mimicry attacks: an attacker injects benign‑looking system calls that share (subject, object) pairs with attack events but differ in predicateObject2. We test whether THyN retains detection capability where pairwise models fail.

---

## 6. Related Work Positioning

### 6.1 Provenance‑Based IDS
- **Flash** (S&P 2024): Pairwise heterogeneous GNN. Validates provenance importance but uses dyadic edges.
- **KAIROS** (S&P 2024): Temporal graph alignment on provenance. Masters time, but misses multi‑entity interactions.
- **MAGIC** (USENIX 2024): Masked self‑supervised learning on standard graphs.
- **Our contribution:** All prior provenance IDS operate on pairwise graphs; we are the first to model provenance events as native variable‑arity hyperedges. We show that attack events are 3× enriched in 3‑entity interactions, providing a statistical argument for the hypergraph representation.

### 6.2 Hypergraph‑Based IDS
- **HyperGAT** (Song et al. 2025): Hypergraphs on provenance, but uses statistical community mining (LFM) and static snapshots — not atomic events, no temporal sequence modeling.
- **HRNN** (Yang et al. 2024): Temporal hypergraph on network flows, KNN‑based hyperedges; unsuitable for deterministic causal tracking.
- **LOHA** (Chen et al. 2025): Hypergraph masked autoencoder for network APT detection. Uses Louvain clustering, positional encodings, reconstruction‑based anomaly scoring. Differs in data domain (network vs. host), hyperedge construction (statistical vs. deterministic), temporal modeling (snapshot vs. continuous sequence), and task (anomaly scoring vs. stage classification). Crucially, statistical clustering buries the attack enrichment signal in 3‑entity events; only deterministic event‑level construction can reveal this structure.
- **Our contribution:** First to combine deterministically constructed provenance hyperedges with continuous temporal modeling and supervised campaign‑stage classification. First to report the 3× attack enrichment in 3‑entity provenance events.

### 6.3 The (a)(b)(c) Framing
> "While hypergraph representations have been explored for network‑level intrusion detection, no prior work has (a) shown that provenance events are naturally variable‑arity hyperedges with a 3× attack enrichment in 3‑entity interactions, (b) designed a temporal hypergraph network that natively handles this variable‑arity structure with continuous sequence modeling, and (c) demonstrated that this approach detects multi‑stage APT campaigns that pairwise provenance GNNs miss."

---

## 7. Paper Section Outline (NDSS format, 12 pages)

| § | Pages | Content |
|---|-------|---------|
| 1 Introduction | 1.5 | APT campaigns → provenance limitations → pairwise loss → our thesis → contributions |
| 2 Background | 1.0 | CDM provenance schema, hypergraph definitions, Mamba/SSM basics |
| 3 Threat Model & Problem | 1.5 | Def. 1 (provenance hypergraph), Obs. 1 (pairwise decomposition loss), concrete example, Obs. 2 (GNN limitation), attack model |
| 4 THyN Architecture | 2.5 | Sections 4.1–4.8, with diagrams; bipartite equivalence acknowledged |
| 5 Evaluation | 3.0 | Datasets (Theia, TRACE), baselines, metrics (esp. Stage‑Level Campaign Recall), multi‑granularity results, ablation, negative result on grouped hyperedges |
| 6 Discussion | 0.5 | Limitations, adversarial robustness, deployment considerations |
| 7 Related Work | 1.0 | Provenance IDS, hypergraph NIDS, temporal GNNs; LOHA deep contrast |
| 8 Conclusion | 0.5 | Summary, future work |

---

## 8. Implementation Roadmap (14 weeks to NDSS)

| Week | Dates | Phase | Key deliverable |
|------|-------|-------|-----------------|
| W1 | May 12–18 | Pipeline finalisation | Atomic hyperedge extraction for all Theia shards; feature engineering complete |
| W2 | May 19–25 | Mamba encoder | Per‑subject grouping, Mamba training on hyperedge sequences |
| W3 | May 26–Jun 1 | Hypergraph convolution | Incidence matrix construction, bipartite message passing |
| W4‑5 | Jun 2–15 | Full model integration | End‑to‑end THyN, training loop, hyperparameter tuning |
| W6 | Jun 16–22 | Baselines | KAIROS, E‑GraphSAGE, TGN on our data |
| W7‑8 | Jun 23–Jul 6 | Main evaluation | All metrics on Theia; TRACE data ingestion and evaluation |
| W9 | Jul 7–13 | Ablation studies | Mamba vs. GRU, atomic vs. grouped, convolution on/off |
| W10 | Jul 14–20 | Paper: §§2–5 | Draft technical sections |
| W11 | Jul 21–27 | Paper: §§1,6,7,8 | Draft intro, discussion, related work, conclusion |
| W12 | Jul 28–Aug 3 | Adversarial robustness | Mimicry attack evaluation |
| W13 | Aug 4–10 | Prof review | Full draft to professor |
| W14 | Aug 11–17 | Revision & submit | Incorporate feedback, final polish, submit Aug 19 |

**Fallback:** If any week slips >3 days, switch primary target to IEEE S&P 2027 Cycle 2 (Nov 17) for a 13‑week buffer.

---

## 9. Project Structure

```
thyne-nids/
├── src/
│   ├── data/
│   │   ├── darpa_parser.py          # Avro → atomic hyperedges
│   │   ├── hypergraph_builder.py    # Incidence matrix & bipartite
│   │   └── feature_extractor.py     # Hyperedge features + temporal
│   ├── model/
│   │   ├── mamba_encoder.py         # Mamba SSM sequence model
│   │   ├── hypergraph_conv.py       # Bipartite message passing
│   │   ├── thyne.py                 # Full THyN model
│   │   └── classifier.py            # Detection heads
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

---

## 10. Key Numbers for the Introduction / Evaluation

- **44,023,627** events across 10 Theia shards → **2,736,362** entities (after null UUID filtering).
- **97.2%** are size‑2 hyperedges (subject + object); **2.8%** (1,221,080) are genuine size‑3.
- **Attack events are 3.0× more likely** to be size‑3 (5.6% vs 1.9%) — driven by `EVENT_MMAP`.
- RF on atomic hyperedge features: **AUC 0.8918**, no single feature >0.70 → signal is strong but non‑trivial.
- Top discriminative features are temporal (`subject_is_new`, `time_gap_same_subject`) → justifies Mamba.
- Class imbalance: 24.3% attack (broad) → 1–3% (narrow) → <0.1% (IoC‑only); evaluation covers all levels.
- Baseline to beat: RF 0.89; expected THyN uplift >6% absolute.
- Incidence matrix: (2,736,362 × 44,023,627), 89.3M non‑zeros, density 1.1×10⁻⁶.