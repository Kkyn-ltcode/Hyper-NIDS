import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix

def aggregate_node_scores(entity_ids, scores, uuid_mapping=None):
    """
    Aggregate event-level anomaly scores into node-level scores.
    The anomaly score for a node is the MAX score of any event it participates in.
    
    Args:
        entity_ids: (N, 3) array of entity IDs (subject, object, object2) for N events.
                    If uuid_mapping is provided, these are integer IDs.
        scores: (N,) array of event anomaly scores (e.g., logits or reconstruction loss).
        uuid_mapping: Dict mapping integer IDs to UUID strings. Optional if entity_ids are already strings.
        
    Returns:
        Dict mapping UUID strings to their max anomaly score.
    """
    node_scores = {}
    
    # Flatten the entity arrays and repeat scores
    # entity_ids is shape (N, 3)
    # We ignore null/padding entities (-1 or 0 usually)
    for i in range(len(scores)):
        score = float(scores[i])
        for ent in entity_ids[i]:
            if ent < 0:  # Skip invalid/padding IDs
                continue
                
            uuid = str(ent)
            if uuid_mapping is not None:
                if ent in uuid_mapping:
                    uuid = uuid_mapping[ent]
                elif str(ent) in uuid_mapping:
                    uuid = uuid_mapping[str(ent)]
            
            # Skip the '00000000-0000-0000-0000-000000000000' null UUID if present
            if uuid == "00000000-0000-0000-0000-000000000000":
                continue
                
            if uuid not in node_scores or score > node_scores[uuid]:
                node_scores[uuid] = score
                
    return node_scores

def evaluate_node_level(node_scores, ground_truth_uuids):
    """
    Compute AUPRC, Best F1, Precision @ 0.1% FPR, and Precision @ 0.01% FPR.
    
    Args:
        node_scores: Dict mapping UUID string to max anomaly score.
        ground_truth_uuids: Set of malicious UUID strings.
        
    Returns:
        Dict of computed metrics.
    """
    uuids = list(node_scores.keys())
    scores = np.array([node_scores[u] for u in uuids])
    y_true = np.array([1 if u in ground_truth_uuids else 0 for u in uuids])
    
    # If no positives or no negatives, metrics are undefined
    if sum(y_true) == 0 or sum(y_true) == len(y_true):
        return {"auprc": 0.0, "best_f1": 0.0, "prec_at_0.1%_fpr": 0.0, "prec_at_0.01%_fpr": 0.0}
        
    auprc = average_precision_score(y_true, scores)
    
    prec, rec, thresholds = precision_recall_curve(y_true, scores)
    # Compute F1 for all thresholds
    f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_f1 = float(np.max(f1_scores)) if len(f1_scores) > 0 else 0.0
    
    # Compute metrics at specific FPRs
    # Sort by score descending
    sorted_indices = np.argsort(scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    scores_sorted = scores[sorted_indices]
    
    total_negatives = np.sum(y_true == 0)
    
    def get_precision_at_fpr(target_fpr):
        target_fp_count = max(1, int(total_negatives * target_fpr))
        fp_count = 0
        tp_count = 0
        
        # We walk down the sorted list (decreasing threshold)
        for y in y_true_sorted:
            if y == 1:
                tp_count += 1
            else:
                fp_count += 1
                
            if fp_count >= target_fp_count:
                break
                
        if tp_count + fp_count == 0:
            return 0.0
        return float(tp_count / (tp_count + fp_count))

    prec_at_01_fpr = get_precision_at_fpr(0.001)
    prec_at_001_fpr = get_precision_at_fpr(0.0001)
    
    return {
        "auprc": float(auprc),
        "best_f1": best_f1,
        "prec_at_0.1%_fpr": prec_at_01_fpr,
        "prec_at_0.01%_fpr": prec_at_001_fpr
    }
