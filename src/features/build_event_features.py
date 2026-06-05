import pandas as pd
import numpy as np
import mmh3
import os
import argparse
from tqdm import tqdm
import pyarrow.parquet as pq

# Define constants
TYPE_PROCESS = 0
TYPE_FILE = 1
TYPE_MEMORY = 2
TYPE_NETFLOW = 3

ZONE_SYSTEM = 0
ZONE_USER = 1
ZONE_TEMP = 2
ZONE_OTHER = 3

def parse_args():
    parser = argparse.ArgumentParser(description="Build per-event HFH features")
    parser.add_argument("--data_dir", type=str, default="data/processed/darpa_tc_e3/theia", help="Path to processed data")
    return parser.parse_args()

def classify_path_zone(path):
    if not path or pd.isna(path):
        return ZONE_OTHER
    
    path = str(path).lower()
    
    if path.startswith(('/usr/', '/bin/', '/sbin/', '/lib/', '/etc/')):
        return ZONE_SYSTEM
    elif path.startswith('/home/'):
        return ZONE_USER
    elif path.startswith(('/tmp/', '/var/tmp/', '/dev/shm/')):
        return ZONE_TEMP
    else:
        return ZONE_OTHER

def get_hfh_buckets(path, max_buckets=16):
    if not path or pd.isna(path):
        return []
        
    path = str(path)
    # Handle absolute paths properly
    is_absolute = path.startswith('/')
    parts = [p for p in path.split('/') if p]
    
    buckets = []
    current_prefix = ""
    for part in parts:
        if current_prefix == "":
            current_prefix = ("/" if is_absolute else "") + part
        else:
            current_prefix = current_prefix + "/" + part
            
        bucket = mmh3.hash(current_prefix, seed=42) % max_buckets
        buckets.append(bucket)
        
    return buckets

def process_network_features(remote_address, remote_port):
    # Returns [is_loopback, is_private, is_external, wellknown, registered, ephemeral] -> 6 dims
    # Or 5 dims as planned: [is_loopback, is_private, is_external, port_bucket1, port_bucket2] (one-hot for port bucket 0, 1, 2)
    # Let's do: is_loopback, is_private, is_external, port_is_wellknown, port_is_registered, port_is_ephemeral
    # Wait, the plan says 5 dims. 
    # `is_loopback`, `is_private`, `is_external` -> 3 dims
    # `port_bucket`: 0, 1, 2 -> 3 dims. Total 6 dims. Let's just use 6 dims. It's safer.
    
    is_loopback = 0.0
    is_private = 0.0
    is_external = 0.0
    
    is_wellknown = 0.0
    is_registered = 0.0
    is_ephemeral = 0.0
    
    if pd.isna(remote_address) or not str(remote_address):
        pass
    else:
        addr = str(remote_address)
        if addr.startswith('127.'):
            is_loopback = 1.0
        elif addr.startswith('10.') or addr.startswith('192.168.') or (addr.startswith('172.') and 16 <= int(addr.split('.')[1]) <= 31):
            is_private = 1.0
        else:
            is_external = 1.0
            
    if pd.isna(remote_port):
        pass
    else:
        try:
            port = int(float(remote_port))
            if 0 <= port <= 1023:
                is_wellknown = 1.0
            elif 1024 <= port <= 49151:
                is_registered = 1.0
            elif port >= 49152:
                is_ephemeral = 1.0
        except ValueError:
            pass
            
    return [is_loopback, is_private, is_external, is_wellknown, is_registered, is_ephemeral]


def main():
    args = parse_args()
    
    print("Loading metadata...")
    # Load metadata
    sub_df = pd.read_parquet(f'{args.data_dir}/subjects.parquet', columns=['uuid', 'process_path'])
    obj_df = pd.read_parquet(f'{args.data_dir}/objects.parquet', columns=['uuid', 'object_type', 'filename', 'remote_address', 'remote_port'])
    
    # We will need dictionaries for fast lookup
    print("Building metadata lookups...")
    
    # Subjects (always PROCESS)
    # Map UUID -> { 'path': path }
    sub_dict = {}
    for row in tqdm(sub_df.itertuples(), total=len(sub_df)):
        sub_dict[row.uuid] = {
            'path': row.process_path
        }
        
    # Objects (can be FILE, MEMORY, NETFLOW)
    obj_dict = {}
    for row in tqdm(obj_df.itertuples(), total=len(obj_df)):
        # Determine object type code
        if row.object_type == 'FILE':
            type_code = TYPE_FILE
        elif row.object_type == 'MEMORY':
            type_code = TYPE_MEMORY
        elif row.object_type == 'NETFLOW':
            type_code = TYPE_NETFLOW
        else:
            type_code = TYPE_FILE # fallback
            
        obj_dict[row.uuid] = {
            'type_code': type_code,
            'path': row.filename,
            'remote_address': row.remote_address,
            'remote_port': row.remote_port
        }
        
    # Process each shard
    shard_dir = f'{args.data_dir}/labeled'
    
    num_shards = 10
    
    for sid in range(num_shards):
        shard_path = f"{shard_dir}/labeled_shard{sid}.parquet"
        if not os.path.exists(shard_path):
            print(f"Skipping {shard_path}, not found.")
            continue
            
        print(f"Processing shard {sid}...")
        df = pd.read_parquet(shard_path, columns=['subject_uuid', 'predicate_object_uuid', 'predicate_object2_uuid'])
        
        N = len(df)
        
        # Dimensions:
        # Group 1: 3 roles * 4 object types = 12 dims
        # Group 2: 2 roles (obj, obj2) * 4 zones = 8 dims
        # Group 3: 2 roles (obj, obj2) * 16 HFH buckets = 32 dims
        # Group 4: network features = 6 dims
        # Total = 12 + 8 + 32 + 6 = 58 dims
        out_features = np.zeros((N, 58), dtype=np.float32)
        
        # We will iterate and build the features
        # Vectorizing this perfectly is hard due to lookups, but we can do it efficiently
        
        # Get raw UUIDs
        sub_uuids = df['subject_uuid'].values
        obj_uuids = df['predicate_object_uuid'].values
        obj2_uuids = df['predicate_object2_uuid'].values
        
        # Null UUID
        null_uuid = "00000000-0000-0000-0000-000000000000"
        
        for i in tqdm(range(N)):
            # Subject is always PROCESS
            sub_id = sub_uuids[i]
            obj_id = obj_uuids[i]
            obj2_id = obj2_uuids[i]
            
            # --- Group 1: Object Types (Cols 0-11) ---
            # Subject type
            if sub_id != null_uuid:
                out_features[i, TYPE_PROCESS] = 1.0 # 0..3
            
            # Object type
            if obj_id != null_uuid and obj_id in obj_dict:
                obj_meta = obj_dict[obj_id]
                out_features[i, 4 + obj_meta['type_code']] = 1.0 # 4..7
            
            # Object2 type
            if obj2_id != null_uuid and obj2_id in obj_dict:
                obj2_meta = obj_dict[obj2_id]
                out_features[i, 8 + obj2_meta['type_code']] = 1.0 # 8..11
                
            # --- Group 2: Path Zones (Cols 12-19) ---
            # Obj Zone (12-15)
            if obj_id != null_uuid and obj_id in obj_dict and obj_dict[obj_id]['type_code'] == TYPE_FILE:
                zone = classify_path_zone(obj_dict[obj_id]['path'])
                out_features[i, 12 + zone] = 1.0
            
            # Obj2 Zone (16-19)
            if obj2_id != null_uuid and obj2_id in obj_dict and obj_dict[obj2_id]['type_code'] == TYPE_FILE:
                zone = classify_path_zone(obj_dict[obj2_id]['path'])
                out_features[i, 16 + zone] = 1.0
                
            # --- Group 3: HFH (Cols 20-51) ---
            # Obj HFH (20-35)
            if obj_id != null_uuid and obj_id in obj_dict and obj_dict[obj_id]['type_code'] == TYPE_FILE:
                buckets = get_hfh_buckets(obj_dict[obj_id]['path'])
                for b in buckets:
                    out_features[i, 20 + b] += 1.0
                    
            # Obj2 HFH (36-51)
            if obj2_id != null_uuid and obj2_id in obj_dict and obj_dict[obj2_id]['type_code'] == TYPE_FILE:
                buckets = get_hfh_buckets(obj_dict[obj2_id]['path'])
                for b in buckets:
                    out_features[i, 36 + b] += 1.0
                    
            # --- Group 4: Network (Cols 52-57) ---
            # Only for NETFLOW objects. We check both obj and obj2, and max them (or just combine)
            net_feats = np.zeros(6, dtype=np.float32)
            if obj_id != null_uuid and obj_id in obj_dict and obj_dict[obj_id]['type_code'] == TYPE_NETFLOW:
                f = process_network_features(obj_dict[obj_id]['remote_address'], obj_dict[obj_id]['remote_port'])
                net_feats = np.maximum(net_feats, f)
            if obj2_id != null_uuid and obj2_id in obj_dict and obj_dict[obj2_id]['type_code'] == TYPE_NETFLOW:
                f = process_network_features(obj_dict[obj2_id]['remote_address'], obj_dict[obj2_id]['remote_port'])
                net_feats = np.maximum(net_feats, f)
                
            out_features[i, 52:58] = net_feats

        # Save the enriched features
        out_path = f"{shard_dir}/enriched_shard{sid}.npz"
        np.savez_compressed(out_path, features=out_features)
        print(f"Saved {out_path} with shape {out_features.shape}")

if __name__ == "__main__":
    main()
