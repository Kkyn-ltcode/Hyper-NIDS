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

            # ──────────────────────────────────────────────────────────────
            # IMPORTANT: TRACE shards are NOT strictly chronological!
            # Shard 16 starts at 15:30 but shard 15 ends at 20:09;
            # shard 34 jumps back to 14:46, shard 55 to 14:27, etc.
            # Processing out-of-order shards corrupts dt = t_curr - last_seen
            # for the SSM state updates, producing cascading NaN under AMP.
            #
            # The splits below restrict to the first 20 shards (0-19),
            # which is the same evaluation window that KAIROS, MAGIC, FLASH,
            # and ThreaTrace use. This gives ~74M events, avoids temporal
            # overlap issues, and enables direct comparison with baselines.
            # ──────────────────────────────────────────────────────────────

            # Supervised cross-campaign split on the baseline-compatible window.
            # Shards 0-7 are clean (no attacks), 8-13 are first attack wave
            # (~6K xproc+ each), 14-19 are mostly clean aftermath.
            # Train includes the clean baseline AND the first attack wave so
            # the supervised model sees positive examples.
            # Test: aftermath + any residual attacks (shard 18 has 1.5K xproc+).
            #
            # | Split  | Shards | Events  | xproc+  |
            # |--------|--------|---------|---------|
            # | train  |  0-10  |  39.6M  | 17,742  |
            # | val    | 11-13  |  10.3M  | 18,388  |
            # | test   | 14-19  |  21.0M  |  1,496  |
            "cross": {
                "train": list(range(0, 11)),   # shards 0-10: clean + first attack wave
                "val":   [11, 12, 13],          # shards 11-13: continued attack wave
                "test":  list(range(14, 20)),   # shards 14-19: aftermath (shard 18 has residual)
            },

            # KAIROS/baseline-compatible split (unsupervised-friendly:
            # pre-attack train, early attack test). For supervised models,
            # prefer "cross" above since train has 0 positive examples here.
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
