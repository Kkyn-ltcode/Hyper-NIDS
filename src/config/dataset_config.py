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
            # Preliminary small split. TRACE has 211 shards total.
            "small": {"train": list(range(8)), "val": [8, 9], "test": [10, 11, 12, 13]},
        },
        # Shards 0-7 are the completely clean period before the first attacks on April 3.
        # This provides a clean baseline for normalization.
        "norm_train_shards": list(range(8)),
    },
    
    "cadets": {
        "splits": {
            # Preliminary small split. CADETS has 10 shards total.
            "small": {"train": [0, 1, 2, 3, 4, 5], "val": [6], "test": [7, 8, 9]},
        },
        # Use first 7 shards for normalization fitting.
        "norm_train_shards": [0, 1, 2, 3, 4, 5, 6],
    },
}
