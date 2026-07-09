import gc
import os
import resource
import time
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

from src.data.ground_truth import (
    load_ground_truth,
    build_attack_subject_uuids,
    build_attack_object_uuids,
    build_child_only_subject_uuids,
)

DATA_ROOT = Path("data/processed/darpa_tc_e3/trace")

def print_mem(step_name):
    # ru_maxrss is in bytes on macOS
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    print(f"[{step_name}] Max RSS: {mem_mb:.1f} MB")

def main():
    print_mem("Start")
    gt = load_ground_truth("trace")
    
    sub_schema = pq.read_schema(DATA_ROOT / "subjects.parquet").names
    obj_schema = pq.read_schema(DATA_ROOT / "objects.parquet").names

    sub_cols = [c for c in ["uuid", "process_path", "cmd_line", "parent_uuid"] if c in sub_schema]
    obj_cols = [c for c in ["uuid", "filename", "remote_address", "local_address"] if c in obj_schema]

    print("Loading subjects...")
    t0 = time.time()
    subjects_df = pd.read_parquet(DATA_ROOT / "subjects.parquet", columns=sub_cols)
    print(f"Subjects loaded in {time.time()-t0:.1f}s. Rows: {len(subjects_df):,}")
    print_mem("After loading subjects")
    
    print("Loading objects...")
    t0 = time.time()
    objects_df = pd.read_parquet(DATA_ROOT / "objects.parquet", columns=obj_cols)
    print(f"Objects loaded in {time.time()-t0:.1f}s. Rows: {len(objects_df):,}")
    print_mem("After loading objects")

    print("Building attack subjects...")
    t0 = time.time()
    attack_sub = build_attack_subject_uuids(subjects_df, gt)
    print(f"Attack subjects built in {time.time()-t0:.1f}s. Found: {len(attack_sub)}")
    print_mem("After build_attack_subject_uuids")

    print("Building attack objects...")
    t0 = time.time()
    attack_obj = build_attack_object_uuids(objects_df, gt)
    print(f"Attack objects built in {time.time()-t0:.1f}s. Found: {len(attack_obj)}")
    print_mem("After build_attack_object_uuids")

    print("Building child subjects...")
    t0 = time.time()
    child_sub = build_child_only_subject_uuids(subjects_df, gt)
    print(f"Child subjects built in {time.time()-t0:.1f}s. Found: {len(child_sub)}")
    print_mem("After build_child_only_subject_uuids")

if __name__ == "__main__":
    main()
