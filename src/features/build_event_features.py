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
        try:
            if addr.startswith('127.'):
                is_loopback = 1.0
            elif addr.startswith('10.') or addr.startswith('192.168.'):
                is_private = 1.0
            elif addr.startswith('172.'):
                second_octet = int(addr.split('.')[1])
                if 16 <= second_octet <= 31:
                    is_private = 1.0
                else:
                    is_external = 1.0
            else:
                is_external = 1.0
        except (ValueError, IndexError):
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
    
    # Map string object types to numeric codes for fast access
    type_map = {'FILE': TYPE_FILE, 'MEMORY': TYPE_MEMORY, 'NETFLOW': TYPE_NETFLOW}
    obj_df['type_code'] = obj_df['object_type'].map(type_map).fillna(TYPE_FILE).astype(int)
    
    # Process each shard
    shard_dir = f'{args.data_dir}/labeled'
    
    import glob
    labeled_files = glob.glob(f"{shard_dir}/labeled_shard*.parquet")
    all_indices = sorted([int(os.path.basename(f).replace("labeled_shard", "").replace(".parquet", "")) for f in labeled_files])
    
    null_uuid = "00000000-0000-0000-0000-000000000000"
    
    for sid in all_indices:
        shard_path = f"{shard_dir}/labeled_shard{sid}.parquet"
        if not os.path.exists(shard_path):
            print(f"Skipping {shard_path}, not found.")
            continue
        
        # ---- NEW: Skip if enriched output already exists ----
        out_path = f"{shard_dir}/enriched_shard{sid}.npz"
        if os.path.exists(out_path):
            print(f"Shard {sid} already processed, skipping.")
            continue
        # -----------------------------------------------------

        print(f"Processing shard {sid}...")
        df = pd.read_parquet(shard_path, columns=['subject_uuid', 'predicate_object_uuid', 'predicate_object2_uuid'])
        
        # Merge sub_df
        df = df.merge(sub_df, left_on='subject_uuid', right_on='uuid', how='left')
        df.rename(columns={'process_path': 'sub_path'}, inplace=True)
        df.drop(columns=['uuid'], inplace=True)
        
        # Merge obj_df (predicate_object_uuid)
        df = df.merge(obj_df, left_on='predicate_object_uuid', right_on='uuid', how='left')
        df.rename(columns={'type_code': 'obj_type_code', 'filename': 'obj_path', 'remote_address': 'obj_ip', 'remote_port': 'obj_port'}, inplace=True)
        df.drop(columns=['uuid', 'object_type'], inplace=True)
        
        # If obj_type_code is NaN, maybe it's a subject?
        df = df.merge(sub_df, left_on='predicate_object_uuid', right_on='uuid', how='left')
        # If it was a subject, uuid will not be null
        df['obj_is_sub'] = df['uuid'].notna()
        df.drop(columns=['uuid', 'process_path'], inplace=True)
        
        # Merge obj_df (predicate_object2_uuid)
        df = df.merge(obj_df, left_on='predicate_object2_uuid', right_on='uuid', how='left')
        df.rename(columns={'type_code': 'obj2_type_code', 'filename': 'obj2_path', 'remote_address': 'obj2_ip', 'remote_port': 'obj2_port'}, inplace=True)
        df.drop(columns=['uuid', 'object_type'], inplace=True)
        
        df = df.merge(sub_df, left_on='predicate_object2_uuid', right_on='uuid', how='left')
        df['obj2_is_sub'] = df['uuid'].notna()
        df.drop(columns=['uuid', 'process_path'], inplace=True)

        N = len(df)
        out_features = np.zeros((N, 62), dtype=np.float32)
        
        sub_uuids = df['subject_uuid'].values
        obj_uuids = df['predicate_object_uuid'].values
        obj2_uuids = df['predicate_object2_uuid'].values
        
        sub_paths = df['sub_path'].values
        
        obj_type_codes = df['obj_type_code'].values
        obj_paths = df['obj_path'].values
        obj_ips = df['obj_ip'].values
        obj_ports = df['obj_port'].values
        obj_is_subs = df['obj_is_sub'].values
        
        obj2_type_codes = df['obj2_type_code'].values
        obj2_paths = df['obj2_path'].values
        obj2_ips = df['obj2_ip'].values
        obj2_ports = df['obj2_port'].values
        obj2_is_subs = df['obj2_is_sub'].values
        
        # Using a raw loop over zip is much faster than itertuples!
        from tqdm import tqdm
        for i, (su, ou, o2u, sp, otc, op, oip, oport, osub, o2tc, o2p, o2ip, o2port, o2sub) in tqdm(enumerate(zip(
            sub_uuids, obj_uuids, obj2_uuids, sub_paths, 
            obj_type_codes, obj_paths, obj_ips, obj_ports, obj_is_subs,
            obj2_type_codes, obj2_paths, obj2_ips, obj2_ports, obj2_is_subs
        )), total=N):
            
            # Sub type
            if su != null_uuid:
                out_features[i, TYPE_PROCESS] = 1.0
                
            # Obj type
            if ou != null_uuid:
                if not np.isnan(otc):
                    out_features[i, 4 + int(otc)] = 1.0
                elif osub:
                    out_features[i, 4 + TYPE_PROCESS] = 1.0
                    
            # Obj2 type
            if o2u != null_uuid:
                if not np.isnan(o2tc):
                    out_features[i, 8 + int(o2tc)] = 1.0
                elif o2sub:
                    out_features[i, 8 + TYPE_PROCESS] = 1.0
                    
            # Sub zone
            if su != null_uuid and not pd.isna(sp):
                zone = classify_path_zone(sp)
                out_features[i, 12 + zone] = 1.0
                
            # Obj zone
            if ou != null_uuid and not np.isnan(otc) and int(otc) == TYPE_FILE and not pd.isna(op):
                zone = classify_path_zone(op)
                out_features[i, 16 + zone] = 1.0
                
            # Obj2 zone
            if o2u != null_uuid and not np.isnan(o2tc) and int(o2tc) == TYPE_FILE and not pd.isna(o2p):
                zone = classify_path_zone(o2p)
                out_features[i, 20 + zone] = 1.0
                
            # Obj HFH
            if ou != null_uuid and not np.isnan(otc) and int(otc) == TYPE_FILE and not pd.isna(op):
                buckets = get_hfh_buckets(op)
                for b in buckets:
                    out_features[i, 24 + b] += 1.0
                
                hfh_norm = out_features[i, 24:40].sum()
                if hfh_norm > 0:
                    out_features[i, 24:40] /= hfh_norm
                    
            # Obj2 HFH
            if o2u != null_uuid and not np.isnan(o2tc) and int(o2tc) == TYPE_FILE and not pd.isna(o2p):
                buckets = get_hfh_buckets(o2p)
                for b in buckets:
                    out_features[i, 40 + b] += 1.0
                    
                hfh2_norm = out_features[i, 40:56].sum()
                if hfh2_norm > 0:
                    out_features[i, 40:56] /= hfh2_norm
                    
            # Net features
            net_feats = np.zeros(6, dtype=np.float32)
            if ou != null_uuid and not np.isnan(otc) and int(otc) == TYPE_NETFLOW:
                f = process_network_features(oip, oport)
                net_feats = np.maximum(net_feats, f)
            if o2u != null_uuid and not np.isnan(o2tc) and int(o2tc) == TYPE_NETFLOW:
                f = process_network_features(o2ip, o2port)
                net_feats = np.maximum(net_feats, f)
                
            out_features[i, 56:62] = net_feats
            
        # Save the enriched features
        np.savez_compressed(out_path, features=out_features)
        print(f"Saved {out_path} with shape {out_features.shape}")

if __name__ == "__main__":
    main()
