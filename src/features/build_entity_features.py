"""Build per-entity static feature vectors from subjects.parquet and objects.parquet.

Produces entity_features.npz with shape (num_entities, n_entity_features).
Features are looked up by entity_id at runtime — zero compute per event.

Feature vector layout (per entity):
  [0:4]   entity_type one-hot: PROCESS, FILE, MEMORY, NETFLOW
  [4:8]   path_zone one-hot: SYSTEM, USER, TEMP, OTHER
  [8]     path_depth (normalized)
  [9]     is_temp_path (binary)
  [10:74] path_token_hash (64-dim hashed bag-of-tokens from path)
  [74:78] port_bucket one-hot: WELLKNOWN, REGISTERED, EPHEMERAL, NONE
  [78]    is_loopback (binary)
  [79]    is_private_ip (binary)
  [80]    is_external_ip (binary)
  Total: 81 features
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────
N_PATH_HASH_BUCKETS = 64
ENTITY_TYPE_MAP = {"SUBJECT_PROCESS": 0, "FILE": 1, "MEMORY": 2, "NETFLOW": 3}
PATH_ZONE_SYSTEM = 0
PATH_ZONE_USER = 1
PATH_ZONE_TEMP = 2
PATH_ZONE_OTHER = 3

SYSTEM_PREFIXES = ("/usr/", "/bin/", "/sbin/", "/lib/", "/etc/", "/boot/", "/opt/")
USER_PREFIXES = ("/home/",)
TEMP_PREFIXES = ("/tmp/", "/var/tmp/", "/dev/shm/", "/var/run/", "/run/")

PRIVATE_PREFIXES = ("10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.",
                    "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.",
                    "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.")
N_FEATURES = 81


def hash_path_tokens(path_str: str, n_buckets: int = N_PATH_HASH_BUCKETS) -> np.ndarray:
    """Hash each token in a path to a fixed-size bag-of-tokens vector."""
    vec = np.zeros(n_buckets, dtype=np.float32)
    if not path_str or path_str == "N/A" or path_str == "None":
        return vec
    # Split by /, spaces, and common delimiters
    tokens = path_str.replace("/", " ").replace("-", " ").replace(".", " ").replace("_", " ").split()
    for token in tokens:
        if token:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16) % n_buckets
            vec[h] = 1.0
    return vec


def classify_path_zone(path_str: str) -> int:
    """Classify a path into a zone category."""
    if not path_str or path_str == "N/A" or path_str == "None":
        return PATH_ZONE_OTHER
    p = path_str.lower()
    if any(p.startswith(pf) for pf in TEMP_PREFIXES):
        return PATH_ZONE_TEMP
    if any(p.startswith(pf) for pf in SYSTEM_PREFIXES):
        return PATH_ZONE_SYSTEM
    if any(p.startswith(pf) for pf in USER_PREFIXES):
        return PATH_ZONE_USER
    return PATH_ZONE_OTHER


def classify_port(port: float) -> int:
    """Classify a port into a bucket: 0=wellknown, 1=registered, 2=ephemeral, 3=none."""
    if np.isnan(port):
        return 3
    port = int(port)
    if port < 1024:
        return 0
    elif port < 49152:
        return 1
    else:
        return 2


def classify_ip(addr: str) -> tuple[bool, bool, bool]:
    """Returns (is_loopback, is_private, is_external)."""
    if not addr or addr == "None":
        return False, False, False
    if addr.startswith("127.") or addr == "::1":
        return True, False, False
    if any(addr.startswith(p) for p in PRIVATE_PREFIXES) or addr.startswith("fd") or addr.startswith("fe80"):
        return False, True, False
    return False, False, True


def build_entity_features(data_root: str) -> None:
    data_root = Path(data_root)
    graph_dir = data_root / "graph"

    # Load entity vocabulary
    vocab = np.load(graph_dir / "entity_vocab.npz", allow_pickle=True)
    uuids = vocab["uuids"]
    num_entities = int(vocab["num_entities"])
    uuid_to_id = {str(u): i for i, u in enumerate(uuids)}

    print(f"Entity vocab: {num_entities} entities")

    # Initialize feature matrix
    features = np.zeros((num_entities, N_FEATURES), dtype=np.float32)

    # ── Process subjects ──────────────────────────────────────────────
    sub = pd.read_parquet(data_root / "subjects.parquet")
    n_matched_sub = 0

    for _, row in sub.iterrows():
        uuid_str = str(row["uuid"])
        if uuid_str not in uuid_to_id:
            continue
        eid = uuid_to_id[uuid_str]
        n_matched_sub += 1

        # Entity type: PROCESS
        features[eid, 0] = 1.0

        # Path features
        path = str(row.get("process_path", "")) if pd.notna(row.get("process_path")) else ""
        cmd = str(row.get("cmd_line", "")) if pd.notna(row.get("cmd_line")) else ""

        # Use process_path preferentially, fall back to cmd_line
        effective_path = path if path and path != "N/A" else cmd

        # Path zone
        zone = classify_path_zone(effective_path)
        features[eid, 4 + zone] = 1.0

        # Path depth
        if effective_path and effective_path != "N/A":
            depth = effective_path.count("/")
            features[eid, 8] = min(depth / 10.0, 1.0)  # normalize

        # Is temp
        features[eid, 9] = 1.0 if zone == PATH_ZONE_TEMP else 0.0

        # Path token hash (combine process_path + cmd_line for richer signal)
        combined = f"{effective_path} {cmd}"
        features[eid, 10:74] = hash_path_tokens(combined)

    print(f"  Subjects matched: {n_matched_sub} / {len(sub)}")

    # ── Process objects ───────────────────────────────────────────────
    obj = pd.read_parquet(data_root / "objects.parquet")
    n_matched_obj = 0

    for _, row in obj.iterrows():
        uuid_str = str(row["uuid"])
        if uuid_str not in uuid_to_id:
            continue
        eid = uuid_to_id[uuid_str]
        n_matched_obj += 1

        obj_type = str(row.get("object_type", ""))

        # Entity type
        if obj_type == "FILE":
            features[eid, 1] = 1.0
        elif obj_type == "MEMORY":
            features[eid, 2] = 1.0
        elif obj_type in ("NETFLOW", "SRCSINK"):
            features[eid, 3] = 1.0

        # File features
        filename = str(row.get("filename", "")) if pd.notna(row.get("filename")) else ""
        if filename and filename != "None":
            zone = classify_path_zone(filename)
            features[eid, 4 + zone] = 1.0
            features[eid, 8] = min(filename.count("/") / 10.0, 1.0)
            features[eid, 9] = 1.0 if zone == PATH_ZONE_TEMP else 0.0
            features[eid, 10:74] = hash_path_tokens(filename)

        # Network features
        remote_addr = str(row.get("remote_address", "")) if pd.notna(row.get("remote_address")) else ""
        remote_port = row.get("remote_port", float("nan"))
        if not isinstance(remote_port, (int, float)):
            remote_port = float("nan")

        # Port bucket
        port_bucket = classify_port(float(remote_port))
        features[eid, 74 + port_bucket] = 1.0

        # IP classification
        is_loop, is_priv, is_ext = classify_ip(remote_addr)
        features[eid, 78] = float(is_loop)
        features[eid, 79] = float(is_priv)
        features[eid, 80] = float(is_ext)

    print(f"  Objects matched: {n_matched_obj} / {len(obj)}")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = graph_dir / "entity_features.npz"
    np.savez_compressed(out_path, features=features)
    print(f"\nSaved entity features: {out_path}")
    print(f"  Shape: {features.shape}")
    print(f"  Non-zero rows: {(features.sum(axis=1) > 0).sum()} / {num_entities}")

    # Quick sanity check: what do attack-related entities look like?
    # Firefox processes should have distinctive path features
    firefox_mask = sub["process_path"].str.contains("firefox", case=False, na=False)
    firefox_uuids = sub.loc[firefox_mask, "uuid"].astype(str)
    firefox_ids = [uuid_to_id[u] for u in firefox_uuids if u in uuid_to_id]
    if firefox_ids:
        ff_feats = features[firefox_ids]
        print(f"\n  Firefox entities ({len(firefox_ids)}): mean feature norm = {np.linalg.norm(ff_feats, axis=1).mean():.2f}")

    # Temp path entities
    temp_mask = features[:, 9] == 1.0
    print(f"  Entities with is_temp=1: {temp_mask.sum()}")
    ext_mask = features[:, 80] == 1.0
    print(f"  Entities with external_ip=1: {ext_mask.sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="data/processed/darpa_tc_e3/theia")
    args = parser.parse_args()
    build_entity_features(args.data_root)
