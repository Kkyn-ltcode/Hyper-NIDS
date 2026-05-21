# Inside __init__, replace the label loading block with:

label_col = f"label_{label_type}"
print(f"  Loading features for shards {shard_ids} (labels={label_type})...")

Xs, ys = [], []
for sid in shard_ids:
    d = np.load(features_dir / f"thyne_shard{sid}.npz")
    Xs.append(d["X"])
    
    if label_type == "broad":
        y_arr = d["y_broad"]
    else:
        # Load any other label from parquet (e.g., label_l1, label_narrow, etc.)
        ldf = pd.read_parquet(labeled_dir / f"labeled_shard{sid}.parquet", columns=[label_col])
        y_arr = ldf[label_col].values.astype(np.int64)
        del ldf
    # Safety check
    unique_vals = np.unique(y_arr)
    print(f"    Shard {sid} label {label_col} unique values: {unique_vals}")
    if (y_arr == -1).all():
        raise ValueError(f"label_{label_type} is all -1 in shard {sid}.")
    ys.append(y_arr)