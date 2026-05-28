import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import json
import gc

data_dir = 'data/processed/darpa_tc_e3/theia/labeled'
subjects_path = 'data/processed/darpa_tc_e3/theia/subjects.parquet'

print("Loading subjects...")
subjects = pd.read_parquet(subjects_path, columns=['uuid', 'parent_uuid', 'process_path'])
subjects['basename'] = subjects['process_path'].apply(lambda x: os.path.basename(x) if pd.notnull(x) else '')
subject_dict = subjects.set_index('uuid').to_dict('index')

results = {
    'check1': {},
    'check2': {},
    'check3': {},
    'check4': {},
    'check5': {}
}

# Check 1: Attack Label Density Per Shard (L1* Labels)
print("Running Check 1...")
for s in range(10):
    df = pd.read_parquet(f"{data_dir}/labeled_shard{s}.parquet", columns=['subject_uuid', 'predicate_object_uuid', 'label_l1'])
    total = len(df)
    attack_events = df[df['label_l1'] == 1]
    att_total = len(attack_events)
    att_rate = att_total / total if total > 0 else 0
    unique_subjs = attack_events['subject_uuid'].nunique()
    unique_objs = attack_events['predicate_object_uuid'].nunique()
    
    results['check1'][f"shard_{s}"] = {
        'total_events': total,
        'attack_events': att_total,
        'attack_rate': float(att_rate),
        'unique_attack_subjects': int(unique_subjs),
        'unique_attack_objects': int(unique_objs)
    }
    del df
    gc.collect()

# Check 2: Firefox Children's Label Distribution in Train/Val
print("Running Check 2...")
firefox_uuids = set(subjects[subjects['basename'].str.lower() == 'firefox']['uuid'])
firefox_children_uuids = set(subjects[subjects['parent_uuid'].isin(firefox_uuids)]['uuid'])

check2_attack_children = 0
check2_event_types = {}

for s in range(8):
    df = pd.read_parquet(f"{data_dir}/labeled_shard{s}.parquet", columns=['subject_uuid', 'type', 'label_l1'])
    children_events = df[df['subject_uuid'].isin(firefox_children_uuids)]
    attack_children = children_events[children_events['label_l1'] == 1]
    check2_attack_children += len(attack_children)
    
    counts = attack_children['type'].value_counts()
    for t, c in counts.items():
        check2_event_types[t] = check2_event_types.get(t, 0) + int(c)
    del df
    gc.collect()

results['check2'] = {
    'total_attack_events_from_firefox_children': int(check2_attack_children),
    'event_types': check2_event_types
}

# Check 3: Behavioral Pattern Consistency (Event-Type Distributions)
print("Running Check 3...")
train_event_types = {}
test_event_types = {}

for s in range(7):
    df = pd.read_parquet(f"{data_dir}/labeled_shard{s}.parquet", columns=['type', 'label_l1'])
    counts = df[df['label_l1'] == 1]['type'].value_counts()
    for t, c in counts.items():
        train_event_types[t] = train_event_types.get(t, 0) + int(c)
    del df
    
for s in [8, 9]:
    df = pd.read_parquet(f"{data_dir}/labeled_shard{s}.parquet", columns=['type', 'label_l1'])
    counts = df[df['label_l1'] == 1]['type'].value_counts()
    for t, c in counts.items():
        test_event_types[t] = test_event_types.get(t, 0) + int(c)
    del df
    gc.collect()

all_types = list(set(train_event_types.keys()).union(set(test_event_types.keys())))
p = np.array([train_event_types.get(t, 0) for t in all_types], dtype=float)
q = np.array([test_event_types.get(t, 0) for t in all_types], dtype=float)

if p.sum() > 0: p /= p.sum()
if q.sum() > 0: q /= q.sum()

jsd = jensenshannon(p, q)
results['check3'] = {
    'train_event_types': train_event_types,
    'test_event_types': test_event_types,
    'jsd': float(jsd) if not np.isnan(jsd) else "NaN"
}

# Check 4: Entity Overlap Between Splits
print("Running Check 4...")
train_entities = set()
test_entities = set()

for s in range(7):
    df = pd.read_parquet(f"{data_dir}/labeled_shard{s}.parquet", columns=['predicate_object_uuid', 'predicate_object2_uuid', 'label_l1'])
    att = df[df['label_l1'] == 1]
    train_entities.update(att['predicate_object_uuid'].dropna().unique())
    train_entities.update(att['predicate_object2_uuid'].dropna().unique())
    del df
    
for s in [8, 9]:
    df = pd.read_parquet(f"{data_dir}/labeled_shard{s}.parquet", columns=['predicate_object_uuid', 'predicate_object2_uuid', 'label_l1'])
    att = df[df['label_l1'] == 1]
    test_entities.update(att['predicate_object_uuid'].dropna().unique())
    test_entities.update(att['predicate_object2_uuid'].dropna().unique())
    del df
    gc.collect()

intersection = train_entities.intersection(test_entities)
union = train_entities.union(test_entities)
jaccard = len(intersection) / len(union) if len(union) > 0 else 0

results['check4'] = {
    'train_entities_count': len(train_entities),
    'test_entities_count': len(test_entities),
    'overlap_count': len(intersection),
    'jaccard_index': float(jaccard)
}

# Check 5: Firefox Baseline Behavior in Train/Val
print("Running Check 5...")
firefox_train_event_types = {}
firefox_train_objects = set()
firefox_test_event_types = {}
firefox_test_objects = set()

for s in range(8):
    df = pd.read_parquet(f"{data_dir}/labeled_shard{s}.parquet", columns=['subject_uuid', 'type', 'predicate_object_uuid', 'label_l1'])
    fx_events = df[(df['subject_uuid'].isin(firefox_uuids)) & (df['label_l1'] == 0)]
    
    counts = fx_events['type'].value_counts()
    for t, c in counts.items():
        firefox_train_event_types[t] = firefox_train_event_types.get(t, 0) + int(c)
    
    firefox_train_objects.update(fx_events['predicate_object_uuid'].dropna().unique())
    del df
    gc.collect()
    
for s in [8, 9]:
    df = pd.read_parquet(f"{data_dir}/labeled_shard{s}.parquet", columns=['subject_uuid', 'type', 'predicate_object_uuid', 'label_l1'])
    fx_events = df[(df['subject_uuid'].isin(firefox_uuids)) & (df['label_l1'] == 1)]
    
    counts = fx_events['type'].value_counts()
    for t, c in counts.items():
        firefox_test_event_types[t] = firefox_test_event_types.get(t, 0) + int(c)
        
    firefox_test_objects.update(fx_events['predicate_object_uuid'].dropna().unique())
    del df
    gc.collect()

results['check5'] = {
    'train_benign_firefox': {
        'event_types': firefox_train_event_types,
        'unique_objects_count': len(firefox_train_objects)
    },
    'test_attack_firefox': {
        'event_types': firefox_test_event_types,
        'unique_objects_count': len(firefox_test_objects)
    }
}

with open('theia_checks_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Done. Saved to theia_checks_results.json")
