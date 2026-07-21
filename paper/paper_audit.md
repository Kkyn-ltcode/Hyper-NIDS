# Paper Audit: What's Missing from `main.tex`

I've read the entire paper (543 lines). It's already a **strong draft** — the architecture section, threat model, and ablation study are well-written and rigorous. Here's what's missing or needs work, organized by priority.

---

## 🔴 Critical (Must Have)

### 1. No `\bibliography` / References
The paper cites `\cite{kairos}`, `\cite{magic}`, `\cite{mamba}`, `\cite{arp2022dos}`, etc., but there is **no `\bibliographystyle{}` or `\bibliography{}` command** — the paper will compile without any references section. You need a `.bib` file.

**Missing references that need entries:**
- `kairos`, `magic`, `flash`, `threatrace`
- `mamba`, `s4`, `s6`
- `arp2022dos` (Arp et al., "Do's and Don'ts of ML in Computer Security")
- `loha`, `hypergat`, `hrnn`

### 2. No Figures At All
The paper references `Figure~\ref{fig:architecture}` (line 147) but **no figure is defined anywhere**. A top-venue paper needs at minimum:

| Figure | Priority | Description |
|--------|----------|-------------|
| **Architecture diagram** | 🔴 Must | The 5-stage pipeline (Event Encoding → State Retrieval → Aggregation → SSM Update → Classification) |
| **Attack campaign timeline** | 🔴 Must | DARPA TC E3 attack phases mapped to the chronological shard structure |
| **Training curves / convergence** | 🟡 Should | Loss + AUPRC over epochs for full vs ablation variants |
| **Precision-Recall curves** | 🟡 Should | PR curves for same-campaign vs cross-campaign |
| **Entity state visualization** | 🟢 Nice | t-SNE/UMAP of entity states for benign vs malicious entities |

### 3. TRACE Dataset Results Missing
You're currently running TRACE. Once it finishes, the paper should include TRACE results in Table 1. This would strengthen the paper significantly:
- **3 datasets** (Theia + CADETS + TRACE) instead of 2
- TRACE is the largest and most challenging (835M events, 176M entities)
- Demonstrates scalability claim

### 4. No `\label{sec:setup}` Section
Line 178 references `Section~\ref{sec:setup}` for the process name vocabulary definition, but **this section doesn't exist**. It should describe:
- How process names are extracted and vocabulary is built
- The Z-score normalization of continuous features
- How `entity_vocab.npz` is constructed
- The `build_graph.py` / `build_event_features.py` pipeline

---

## 🟡 Important (Should Have)

### 5. No Data Preprocessing Section
The paper jumps from architecture directly to evaluation without describing:
- The `build_graph.py` pipeline that constructs the provenance hypergraph
- Feature engineering (which 7 continuous features, which 35 binary features)
- How Z-score normalization is fitted on train shards only
- Entity vocabulary construction
- Shard structure and chronological ordering

> [!IMPORTANT]  
> Reviewers will want to know the complete feature set. The 7 continuous features and the normalization procedure are implementation details that affect reproducibility.

### 6. Experimental Setup Incomplete (Section 5.1)
- **Training hyperparameters** not stated: learning rate, optimizer, epochs, chunk size, d_model, weight decay, early stopping criteria
- **Hardware** not specified: GPU type, RAM, training time per epoch
- **Reproducibility**: no mention of seeds, variance across runs, or code availability
- **CADETS setup** not detailed: which split, which shards, label type

### 7. No Throughput Comparison with Baselines
Table 3 shows HyperMamba throughput but doesn't compare against baseline systems' computational costs. Even a qualitative comparison (e.g., "KAIROS requires offline graph construction before inference") would strengthen the operational argument.

### 8. Entity State Bank Section Outdated
Section 4.3 (line 209) says:
> "For the Theia dataset (|V| ≈ 7.1 × 10^6, d = 256), this amounts to 6.8 GB in float32, fitting within a single commodity GPU."

This is now inaccurate — the bank lives on **CPU** with the new pinned-memory optimization. The paper should describe the CPU bank + per-chunk GPU gather/scatter approach, as it's actually a technical contribution (enabling TRACE-scale datasets).

---

## 🟢 Nice to Have (Polish)

### 9. No Ethical Considerations / Broader Impact
Many venues now require or expect this. Brief paragraph on:
- Surveillance implications of fine-grained event monitoring
- False positive consequences in operational settings
- Dataset is from a controlled DARPA engagement, not real user data

### 10. No Appendix
Could include:
- Full hyperparameter table
- Extended results per-shard
- Additional ablation variants (e.g., `event_only`)
- The `event_only` ablation (both state AND cross-entity disabled) from your code isn't in the paper

### 11. Abstract Could Mention TRACE
Once results are in, update abstract to mention 3 datasets instead of 2.

### 12. Conclusion is Weak on Future Work
The conclusion (line 540) just restates results. Add 2-3 sentences on future directions:
- Self-supervised pretraining to reduce label dependency
- Multi-host lateral movement detection
- State compression for extreme-scale deployments
- Adversarial robustness evaluation

---

## Summary: Action Items

| # | Item | Priority | Effort |
|---|------|----------|--------|
| 1 | Create `.bib` file with all references | 🔴 | Low |
| 2 | Architecture diagram (Figure 1) | 🔴 | Medium |
| 3 | Add TRACE results to Table 1 | 🔴 | Low (after run) |
| 4 | Write Section 5.1 "Setup" with hyperparams + features | 🔴 | Medium |
| 5 | Add data preprocessing subsection | 🟡 | Medium |
| 6 | Complete experimental setup (hardware, seeds) | 🟡 | Low |
| 7 | Attack campaign timeline figure | 🟡 | Medium |
| 8 | Update Entity State Bank section (CPU bank) | 🟡 | Low |
| 9 | PR curve / training curve figures | 🟡 | Medium |
| 10 | Ethics paragraph | 🟢 | Low |
| 11 | Future work in conclusion | 🟢 | Low |
| 12 | Appendix with full hyperparams | 🟢 | Low |

> [!NOTE]
> Which items would you like me to start working on? I'd suggest tackling the `.bib` file and the experimental setup section first since those don't require waiting for TRACE results, and the figures can be created in parallel.
