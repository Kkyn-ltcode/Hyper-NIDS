"""
Investigate the null UUID and predicateObject2 semantics.

Answers:
1. Where does 00000000-... appear? (subject? obj? obj2?)
2. How many events have a REAL (non-null) predicateObject2?
3. Are attack events more likely to have a real obj2?
4. What are the FE/FD prefix entities?

Usage:
    python -m src.pipeline.investigate_null_uuid --dataset theia
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "darpa_tc_e3"
)

NULL_UUID = "00000000-0000-0000-0000-000000000000"

# Other suspicious high-degree entities from the graph stats
SUSPICIOUS_UUIDS = [
    NULL_UUID,
    "FEFFFFFF-0000-FFFF-FFFF-000000000040",
    "FDFFFFFF-0000-FFFF-FFFF-000000000040",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="theia")
    args = parser.parse_args()

    labeled_dir = DATA_ROOT / args.dataset / "labeled"
    shard_files = sorted(labeled_dir.glob("labeled_shard*.parquet"))

    print("=" * 60)
    print(f"NULL UUID INVESTIGATION: {args.dataset.upper()}")
    print("=" * 60)

    # Counters
    total = 0
    null_as_subject = 0
    null_as_obj = 0
    null_as_obj2 = 0
    real_obj2 = 0  # events with non-null, non-placeholder obj2
    attack_with_real_obj2 = 0
    benign_with_real_obj2 = 0
    attack_total = 0
    benign_total = 0

    # Track suspicious UUIDs
    suspicious_counts = {u: {"subject": 0, "obj": 0, "obj2": 0}
                         for u in SUSPICIOUS_UUIDS}
    suspicious_set = set(SUSPICIOUS_UUIDS)

    # Event type breakdown for real obj2
    real_obj2_by_type = {}
    null_obj2_by_type = {}

    for f in shard_files:
        print(f"  Scanning {f.stem}...")
        df = pd.read_parquet(f, columns=[
            "subject_uuid", "predicate_object_uuid",
            "predicate_object2_uuid", "label_broad", "type",
        ])
        n = len(df)
        total += n

        sub = df["subject_uuid"].values
        obj = df["predicate_object_uuid"].values
        obj2 = df["predicate_object2_uuid"].values
        labels = df["label_broad"].values
        types = df["type"].values

        # Null UUID appearances
        null_as_subject += (sub == NULL_UUID).sum()
        null_as_obj += (obj == NULL_UUID).sum()
        null_as_obj2 += (obj2 == NULL_UUID).sum()

        # Suspicious UUID appearances
        for u in SUSPICIOUS_UUIDS:
            suspicious_counts[u]["subject"] += (sub == u).sum()
            suspicious_counts[u]["obj"] += (obj == u).sum()
            suspicious_counts[u]["obj2"] += (obj2 == u).sum()

        # Real obj2: non-null AND not in suspicious set
        is_real_obj2 = ~pd.Series(obj2).isin(suspicious_set).values
        n_real = int(is_real_obj2.sum())
        real_obj2 += n_real

        # Attack vs benign with real obj2
        is_atk = labels == 1
        attack_with_real_obj2 += int((is_atk & is_real_obj2).sum())
        benign_with_real_obj2 += int((~is_atk & is_real_obj2).sum())
        attack_total += int(is_atk.sum())
        benign_total += int((~is_atk).sum())

        # Event type breakdown
        for t in np.unique(types):
            mask_t = types == t
            n_real_t = int((mask_t & is_real_obj2).sum())
            n_null_t = int((mask_t & ~is_real_obj2).sum())
            real_obj2_by_type[t] = real_obj2_by_type.get(t, 0) + n_real_t
            null_obj2_by_type[t] = null_obj2_by_type.get(t, 0) + n_null_t

        del df
        gc.collect()

    # Report
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    print(f"\n  Total events: {total:,}")

    print(f"\n  Null UUID ({NULL_UUID[:20]}...):")
    print(f"    As subject:  {null_as_subject:,} "
          f"({100*null_as_subject/total:.2f}%)")
    print(f"    As obj:      {null_as_obj:,} "
          f"({100*null_as_obj/total:.2f}%)")
    print(f"    As obj2:     {null_as_obj2:,} "
          f"({100*null_as_obj2/total:.2f}%)")

    print(f"\n  All suspicious UUIDs:")
    for u, counts in suspicious_counts.items():
        total_u = counts["subject"] + counts["obj"] + counts["obj2"]
        print(f"    {u[:36]}:")
        print(f"      subject={counts['subject']:,}  "
              f"obj={counts['obj']:,}  "
              f"obj2={counts['obj2']:,}  "
              f"total={total_u:,}")

    print(f"\n  Events with REAL obj2 (excluding all suspicious UUIDs):")
    print(f"    Real obj2:   {real_obj2:,} ({100*real_obj2/total:.1f}%)")
    print(f"    Null/sentinel obj2: {total-real_obj2:,} "
          f"({100*(total-real_obj2)/total:.1f}%)")

    print(f"\n  CRITICAL: Attack events with real obj2:")
    atk_pct = 100 * attack_with_real_obj2 / max(attack_total, 1)
    ben_pct = 100 * benign_with_real_obj2 / max(benign_total, 1)
    print(f"    Attack events with real obj2: "
          f"{attack_with_real_obj2:,} / {attack_total:,} ({atk_pct:.1f}%)")
    print(f"    Benign events with real obj2: "
          f"{benign_with_real_obj2:,} / {benign_total:,} ({ben_pct:.1f}%)")
    if atk_pct > ben_pct:
        print(f"    → Attack events are {atk_pct/max(ben_pct,0.01):.1f}x "
              f"more likely to have real obj2!")
    else:
        print(f"    → No significant difference.")

    print(f"\n  Event types with most REAL obj2:")
    sorted_types = sorted(real_obj2_by_type.items(), key=lambda x: -x[1])
    print(f"    {'Type':35s} {'Real obj2':>10s} {'Null obj2':>10s} "
          f"{'%Real':>6s}")
    print(f"    {'─'*65}")
    for t, n_real in sorted_types[:15]:
        n_null = null_obj2_by_type.get(t, 0)
        total_t = n_real + n_null
        pct = 100 * n_real / max(total_t, 1)
        print(f"    {t:35s} {n_real:>10,} {n_null:>10,} {pct:>5.1f}%")


if __name__ == "__main__":
    main()
