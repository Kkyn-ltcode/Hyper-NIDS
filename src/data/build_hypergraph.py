"""
Hypergraph construction from CDM provenance events.

Builds the incidence structure (entity vocabulary + incidence list)
needed for hypergraph convolution. Each CDM event = one hyperedge
connecting 3 entities (subject, object, object2).

Usage:
    from src.data.build_hypergraph import build_incidence
    entity_vocab, incidence = build_incidence(events_df)
"""

import numpy as np
import pandas as pd
from collections import OrderedDict


def build_entity_vocab(
    events_df: pd.DataFrame,
) -> dict[str, int]:
    """
    Build a vocabulary mapping entity UUIDs to contiguous integer IDs.

    Collects all unique UUIDs from subject_uuid, predicate_object_uuid,
    and predicate_object2_uuid columns.

    Returns:
        entity_vocab: dict mapping UUID string -> int ID
    """
    all_uuids = set()

    for col in ["subject_uuid", "predicate_object_uuid",
                "predicate_object2_uuid"]:
        if col in events_df.columns:
            uuids = events_df[col].dropna().unique()
            all_uuids.update(uuids)

    # Sort for reproducibility
    entity_vocab = {uuid: idx for idx, uuid
                    in enumerate(sorted(all_uuids))}
    return entity_vocab


def build_incidence(
    events_df: pd.DataFrame,
    entity_vocab: dict[str, int] | None = None,
) -> tuple[dict[str, int], list[list[int]]]:
    """
    Build the hypergraph incidence structure.

    Each event becomes a hyperedge connecting its entity IDs.
    The incidence list is aligned with events_df.index — incidence[i]
    corresponds to events_df.iloc[i].

    Args:
        events_df: Events DataFrame with entity UUID columns
        entity_vocab: Optional pre-built vocabulary. Built if None.

    Returns:
        entity_vocab: dict mapping UUID -> int ID
        incidence: list of lists, each inner list contains the integer
                   entity IDs participating in that hyperedge
    """
    if entity_vocab is None:
        entity_vocab = build_entity_vocab(events_df)

    sub_uuids = events_df["subject_uuid"].values
    obj_uuids = events_df["predicate_object_uuid"].values
    obj2_uuids = (events_df["predicate_object2_uuid"].values
                  if "predicate_object2_uuid" in events_df.columns
                  else [None] * len(events_df))

    incidence = []
    for i in range(len(events_df)):
        nodes = []
        for uuid in [sub_uuids[i], obj_uuids[i], obj2_uuids[i]]:
            if pd.notna(uuid) and uuid in entity_vocab:
                nodes.append(entity_vocab[uuid])
        incidence.append(nodes)

    return entity_vocab, incidence


def incidence_stats(
    entity_vocab: dict[str, int],
    incidence: list[list[int]],
) -> dict:
    """Compute summary statistics of the hypergraph."""
    sizes = [len(nodes) for nodes in incidence]
    n_entities = len(entity_vocab)
    n_hyperedges = len(incidence)

    # Node degree: how many hyperedges each entity participates in
    degree = np.zeros(n_entities, dtype=np.int64)
    for nodes in incidence:
        for nid in nodes:
            degree[nid] += 1

    return {
        "n_entities": n_entities,
        "n_hyperedges": n_hyperedges,
        "he_size_mean": np.mean(sizes),
        "he_size_min": min(sizes),
        "he_size_max": max(sizes),
        "node_degree_mean": degree.mean(),
        "node_degree_median": np.median(degree),
        "node_degree_max": degree.max(),
    }
