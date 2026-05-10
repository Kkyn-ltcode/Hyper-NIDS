"""
THyN: Temporal Hypergraph Network for Provenance-Based IDS.

v0 architecture:
    Input projection → Sequence encoder (GRU/Mamba) → [HypergraphConv] → Classifier

All components are in this file for v0. Will be split into
separate modules when complexity warrants it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphConv(nn.Module):
    """
    Lightweight hypergraph convolution via bipartite message passing.

    For each entity, aggregates embeddings of incident hyperedges (mean).
    For each hyperedge, updates by summing its entities' aggregated embeddings.
    One round of entity↔hyperedge message passing.
    """

    def __init__(self, d_hidden: int):
        super().__init__()
        self.linear = nn.Linear(d_hidden, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)

    def forward(self, he_emb, entity_ids, mask):
        """
        Args:
            he_emb:     (B, L, D) hyperedge embeddings
            entity_ids: (B, L, 3) int64, entity IDs per event (-1 = invalid)
            mask:       (B, L) float, 1=real, 0=pad

        Returns:
            he_emb_updated: (B, L, D)
        """
        B, L, D = he_emb.shape

        # Flatten batch for scatter operations
        he_flat = he_emb.reshape(B * L, D)              # (B*L, D)
        ent_flat = entity_ids.reshape(B * L, 3)         # (B*L, 3)
        mask_flat = mask.reshape(B * L)                  # (B*L,)

        # Find all unique entity IDs in this batch
        valid_ent = ent_flat[mask_flat.bool()]            # (N_valid, 3)
        all_ent_ids = valid_ent.reshape(-1)               # (N_valid*3,)
        all_ent_ids = all_ent_ids[all_ent_ids >= 0]       # filter -1

        if len(all_ent_ids) == 0:
            return he_emb

        unique_ents, inv_map = torch.unique(all_ent_ids, return_inverse=True)
        n_ents = len(unique_ents)

        # Build entity→local_id mapping for this batch
        ent_to_local = torch.full(
            (unique_ents.max().item() + 1,), -1,
            dtype=torch.long, device=he_emb.device)
        ent_to_local[unique_ents] = torch.arange(
            n_ents, device=he_emb.device)

        # Step 1: Aggregate HE embeddings → entity embeddings (mean)
        ent_emb = torch.zeros(n_ents, D, device=he_emb.device)
        ent_count = torch.zeros(n_ents, 1, device=he_emb.device)

        for col in range(3):
            col_ids = ent_flat[:, col]             # (B*L,)
            valid = (col_ids >= 0) & mask_flat.bool()
            if not valid.any():
                continue
            local_ids = ent_to_local[col_ids[valid]]
            ent_emb.scatter_add_(0, local_ids.unsqueeze(1).expand(-1, D),
                                 he_flat[valid])
            ent_count.scatter_add_(0, local_ids.unsqueeze(1),
                                   torch.ones(valid.sum(), 1,
                                              device=he_emb.device))

        ent_emb = ent_emb / ent_count.clamp(min=1)

        # Step 2: Update HE embeddings ← sum of entity embeddings
        he_update = torch.zeros_like(he_flat)
        for col in range(3):
            col_ids = ent_flat[:, col]
            valid = (col_ids >= 0) & mask_flat.bool()
            if not valid.any():
                continue
            local_ids = ent_to_local[col_ids[valid]]
            he_update[valid] += ent_emb[local_ids]

        he_update = he_update.reshape(B, L, D)
        he_out = self.norm(he_emb + self.linear(he_update))
        return he_out


class THyN(nn.Module):
    """
    Temporal Hypergraph Network v0.

    Architecture:
        1. Linear projection: (n_features) → (d_model)
        2. Sequence encoder: GRU or Mamba over per-subject windows
        3. Optional HypergraphConv: bipartite message passing
        4. Classifier: Linear → sigmoid for binary attack/benign
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        d_hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_conv: bool = True,
        encoder_type: str = "gru",  # "gru" or "mamba"
    ):
        super().__init__()
        self.use_conv = use_conv
        self.encoder_type = encoder_type

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Sequence encoder
        if encoder_type == "gru":
            self.encoder = nn.GRU(
                input_size=d_model,
                hidden_size=d_hidden,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
                bidirectional=False,
            )
        elif encoder_type == "mamba":
            try:
                from mamba_ssm import Mamba
                self.encoder = nn.ModuleList([
                    Mamba(d_model=d_hidden, d_state=16, d_conv=4, expand=2)
                    for _ in range(n_layers)
                ])
            except ImportError:
                raise ImportError(
                    "mamba-ssm not installed. Use encoder_type='gru' "
                    "or install: pip install mamba-ssm")
        else:
            raise ValueError(f"Unknown encoder: {encoder_type}")

        # Optional hypergraph convolution
        if use_conv:
            self.hg_conv = HypergraphConv(d_hidden)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

    def encode_sequence(self, x):
        """Run sequence encoder."""
        if self.encoder_type == "gru":
            out, _ = self.encoder(x)
            return out
        elif self.encoder_type == "mamba":
            h = x
            for layer in self.encoder:
                h = layer(h)
            return h

    def forward(self, X, entity_ids=None, mask=None):
        """
        Args:
            X:          (B, L, n_features) float
            entity_ids: (B, L, 3) int64, optional
            mask:       (B, L) float, optional

        Returns:
            logits: (B, L) — raw logits (apply sigmoid for probabilities)
        """
        # 1. Project input
        h = self.input_proj(X)             # (B, L, d_model)

        # 2. Sequence encoding
        h = self.encode_sequence(h)        # (B, L, d_hidden)

        # 3. Optional hypergraph convolution
        if self.use_conv and entity_ids is not None and mask is not None:
            h = self.hg_conv(h, entity_ids, mask)

        # 4. Classify each position
        logits = self.classifier(h).squeeze(-1)  # (B, L)

        return logits
