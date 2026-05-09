# HyperMamba-NIDS

**"Catching Campaigns, Not Connections: Coordination-Aware Hypergraph Networks for Multi-Stage Intrusion Detection"**

Target: NDSS Fall 2027 (August 19, 2026)

## Project Status

- [x] Project setup
- [ ] Week 1: Paper reading (KAIROS, ORTHRUS, Mamba, k-HNN)
- [ ] Week 2: Write §2-§3 drafts + environment config
- [ ] Week 3: Kill switch experiment (SCA-degree measurement) → GO/NO-GO
- [ ] Week 4-5: Data pipeline (DARPA TC, CICIDS, hypergraph construction)
- [ ] Week 6-7: Model implementation
- [ ] Week 8-9: Baselines + evaluation
- [ ] Week 10-11: Ablations, adversarial robustness, writing
- [ ] Week 12-14: Polish, professor review, submit

## Setup

```bash
# Create environment
conda create -n hypermamba python=3.10 -y
conda activate hypermamba

# Install Phase 1 dependencies (data + kill switch)
pip install numpy pandas scikit-learn scipy tqdm pyyaml jsonlines
```

## Project Structure

```
src/
├── data/           # Dataset loading and preprocessing
├── coordination/   # CD computation, hypergraph construction, SCA-degree
├── model/          # CD-weighted HGNN + temporal encoder
├── baselines/      # KAIROS wrapper, E-GraphSAGE, TGN
├── evaluation/     # Metrics, Campaign Recall, adversarial
└── utils/          # Config, logging
```
