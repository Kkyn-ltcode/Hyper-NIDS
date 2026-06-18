import argparse
import os
import glob
import pandas as pd
import numpy as np
import pickle
import psycopg2

def extract_node_mapping_from_postgres(dbname="tc_data", user="postgres", password="password", host="localhost", port=5432):
    """
    Connect to PIDSMaker Postgres DB and extract index_id -> UUID mapping.
    """
    try:
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
        cur = conn.cursor()
        
        mapping = {}
        tables = ["subject_node_table", "file_node_table", "netflow_node_table"]
        
        for table in tables:
            try:
                # The columns are [uuid, hashstr, ..., index_id]
                # index_id is always the last column. uuid is the first column.
                cur.execute(f"SELECT uuid, index_id FROM {table};")
                rows = cur.fetchall()
                for row in rows:
                    mapping[int(row[1])] = str(row[0])
            except Exception as e:
                print(f"Warning: Could not read table {table}: {e}")
                conn.rollback()
                
        cur.close()
        conn.close()
        return mapping
    except Exception as e:
        print(f"Failed to connect to Postgres: {e}")
        return None

def parse_pidsmaker_outputs(edge_losses_dir, node_mapping):
    """
    Parse PIDSMaker edge loss CSVs and aggregate to node-level max scores.
    """
    csv_files = glob.glob(os.path.join(edge_losses_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        print(f"No CSV files found in {edge_losses_dir}")
        return {}
        
    print(f"Found {len(csv_files)} CSV files. Parsing...")
    
    node_max_scores = {}
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Find the score column
            score_col = None
            for col in ['loss', 'score', 'anomaly_score', 'reconstruction_loss']:
                if col in df.columns:
                    score_col = col
                    break
            
            # Find node columns
            src_col = None
            dst_col = None
            for col in ['src_id', 'source', 'src']:
                if col in df.columns:
                    src_col = col
                    break
            for col in ['dst_id', 'target', 'dst']:
                if col in df.columns:
                    dst_col = col
                    break
                    
            if score_col is None:
                # Fallback to the last column if it's numeric
                score_col = df.columns[-1]
            if src_col is None:
                src_col = df.columns[0]
            if dst_col is None:
                dst_col = df.columns[1]
                
            scores = df[score_col].values
            src_ids = df[src_col].values
            dst_ids = df[dst_col].values
            
            for i in range(len(scores)):
                s = float(scores[i])
                src = src_ids[i]
                dst = dst_ids[i]
                
                src_uuid = node_mapping.get(src, str(src))
                dst_uuid = node_mapping.get(dst, str(dst))
                
                if src_uuid not in node_max_scores or s > node_max_scores[src_uuid]:
                    node_max_scores[src_uuid] = s
                if dst_uuid not in node_max_scores or s > node_max_scores[dst_uuid]:
                    node_max_scores[dst_uuid] = s
                    
        except Exception as e:
            print(f"Error parsing {f}: {e}")
            
    return node_max_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse PIDSMaker baseline outputs")
    parser.add_argument("--edge_losses_dir", type=str, required=True, help="Directory containing PIDSMaker CSV outputs")
    parser.add_argument("--mapping_csv", type=str, default=None, help="Path to node_id,uuid mapping CSV")
    parser.add_argument("--out_file", type=str, default="parsed_pidsmaker_scores.pkl", help="Output file")
    args = parser.parse_args()
    
    mapping = {}
    if args.mapping_csv and os.path.exists(args.mapping_csv):
        print(f"Loading mapping from {args.mapping_csv}")
        df = pd.read_csv(args.mapping_csv)
        # Assume columns are node_id, uuid
        mapping = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    else:
        print("Attempting to connect to PIDSMaker Postgres DB for node mapping...")
        db_mapping = extract_node_mapping_from_postgres()
        if db_mapping:
            print(f"Successfully extracted {len(db_mapping)} node mappings from Postgres.")
            mapping = db_mapping
        else:
            print("Warning: Could not get node mapping. Node scores will be keyed by integer IDs.")
            
    node_scores = parse_pidsmaker_outputs(args.edge_losses_dir, mapping)
    
    print(f"Aggregated scores for {len(node_scores)} unique nodes.")
    with open(args.out_file, "wb") as f:
        pickle.dump(node_scores, f)
    print(f"Saved to {args.out_file}")
