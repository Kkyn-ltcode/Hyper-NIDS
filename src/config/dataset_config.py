"""
Centralized dataset configuration for HyperMamba-NIDS.

This module defines the training, validation, and test splits for all datasets,
ensuring consistency across feature normalization and model evaluation.
"""

DATASET_CONFIG = {
    "theia": {
        "splits": {
            # Same-campaign (Campaign 1 only). Fast iteration.
            "small": {"train": list(range(7)), "val": [7], "test": [8, 9]},
            
            # Mixed-campaign. Both campaigns in training, last shards held out.
            # This is the standard PIDS evaluation — matches how baselines are tested.
            "full":  {"train": list(range(8)) + list(range(12, 20)),
                      "val": [8, 9, 20, 21], "test": [10, 22, 23, 24]},
            
            # Cross-campaign generalization. Train Campaign 1, test Campaign 2.
            # Hardest setting — tests whether supervised detection transfers.
            "cross": {"train": list(range(9)), "val": [9, 10],
                      "test": list(range(12, 25))},
            
            # KAIROS/baseline-compatible split: matches the train/val/test boundaries
            # used in KAIROS Table 12 (train=Apr 3-5, val=Apr 9, test=Apr 10-12).
            # Use with --label_type broad for direct comparison against published
            # baseline results that evaluate ALL attack nodes (not crossprocess+).
            "kairos": {"train": list(range(0, 9)), "val": [9, 10],
                       "test": list(range(11, 20))},
        },
        # Shards used exclusively to fit the Z-score normalization (mean/std).
        # We use the entire Campaign 1 period (shards 0-10) before the 92-hour gap.
        "norm_train_shards": list(range(11)),
    },
    
    "trace": {
        "splits": {
            # Quick iteration split. TRACE has 211 shards total.
            "small": {"train": list(range(8)), "val": [8, 9], "test": [10, 11, 12, 13]},
            # Memory-bounded split for machines that can't load the full 211
            # shards (~835M events) at once, and that also sidesteps a labeling
            # anomaly: shards 154-203 are ~99% labeled crossprocess+ (vs. a
            # normal <0.5% elsewhere), which looks like a taint-propagation
            # bug rather than real ground truth. "full"'s val/test ranges
            # (150-180, 180-211) are 84-89% inside that block, which is
            # probably not what you want to evaluate against.
            # Train: shards 0-59, the earliest ~222M clean events (Apr 2-6).
            # Val:   shards 60-69, ~34M clean events immediately after train.
            # Test:  shards 204-210, ~22M clean events — the tail *after*
            #        the anomalous block, still chronologically post-Apr-10.
            # ~278M events total, ~26 GB in ChronoDataset's numeric fields
            # after the event_uuids/raw_nanos fix — should fit machines with
            # roughly 64GB+ free RAM. Scale the ranges up/down from there.
            "partial": {"train": list(range(60)), "val": list(range(60, 70)), "test": list(range(204, 211))},
            
            # Focused cross-campaign split: 50 strategically selected shards (~200M events).
            # Covers full temporal range (April 2-13) with entity re-indexing.
            # Train: 25 shards spanning April 2-9 (clean baseline + 4 attack waves)
            # Val:    7 shards at April 9-10 boundary (transition period)
            # Test:  18 shards April 10-13 (onset → peak narrow+ → late recovery)
            "cross": {
                "train": [0, 3, 6, 8, 9, 10, 11, 12, 13, 18, 29, 35,
                          48, 49, 52, 55, 56, 60, 62, 69, 77, 84, 91, 95, 99],
                "val":   [105, 107, 112, 113, 114, 115, 116],
                "test":  [121, 122, 123, 124, 125, 126, 136, 137, 139,
                          154, 160, 170, 180, 190, 200, 204, 207, 210],
            },
            
            # KAIROS/baseline-compatible split (pre-attack train, early attack test)
            "kairos": {"train": list(range(0, 8)), "val": [8, 9, 10],
                       "test": list(range(11, 20))},
        },
        # Shards 0-7 are the completely clean period before the first attacks on April 3.
        # This provides a clean baseline for normalization.
        "norm_train_shards": list(range(8)),
    },
    
    "cadets": {
        "splits": {
            # Preliminary small split. CADETS has 10 shards total.
            "small": {"train": [0, 1, 2, 3, 4, 5], "val": [6], "test": [7, 8, 9]},
            "full": {"train": [0, 1, 2, 3, 4, 5, 6], "val": [7], "test": [8, 9]},
            "cross": {"train": [0, 1, 2, 3, 4, 5], "val": [6, 7], "test": [8, 9]},
            
            # KAIROS/baseline-compatible split (train=Apr 3-5, val=Apr 9, test=Apr 10-12)
            "kairos": {"train": [0, 1, 2], "val": [5], "test": [6, 7, 8]},
        },
        # Use first 7 shards for normalization fitting.
        "norm_train_shards": [0, 1, 2, 3, 4, 5, 6],
    },
}
