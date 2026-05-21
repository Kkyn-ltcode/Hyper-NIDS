"""
THyN: Temporal Hypergraph Network for Provenance-Based IDS.

Supports multiple configurations via model_type:
    - "thyn":       Mamba/GRU + HypergraphConv (3-entity, full model)
    - "baseline_a": Mamba/GRU + PairwiseConv (2-entity, drops obj2)
    - "baseline_b": Mamba/GRU only (no graph conv)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphConv(nn.Module):
    """Hypergraph convolution (3‑entity)."""
    def __init__(self, d_hidden: int, n_entities: int = 3):
        super().__init__()
        self.n_entities = n_entities
        self.linear = nn.Linear(d_hidden, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)

    def forward(self, he_emb, entity_ids, mask):
        B, L, D = he_emb.shape
        he_flat = he_emb.reshape(B * L, D)
        ent_flat = entity_ids.reshape(B * L, self.n_entities)
        mask_flat = mask.reshape(B * L).bool()

        valid_ent = ent_flat[mask_flat][:, :self.n_entities].reshape(-1)
        valid_ent = valid_ent[valid_ent >= 0]
        if len(valid_ent) == 0:
            return he_emb

        unique_ents = torch.unique(valid_ent)
        n_ents = len(unique_ents)
        ent_to_local = torch.full((unique_ents.max().item() + 1,), -1,
                                  dtype=torch.long, device=he_emb.device)
        ent_to_local[unique_ents] = torch.arange(n_ents, device=he_emb.device)

        # Entity ← mean(HE)
        ent_emb = torch.zeros(n_ents, D, device=he_emb.device)
        ent_count = torch.zeros(n_ents, 1, device=he_emb.device)
        for col in range(self.n_entities):
            col_ids = ent_flat[:, col]
            valid = (col_ids >= 0) & mask_flat
            if not valid.any():
                continue
            local = ent_to_local[col_ids[valid]]
            ent_emb.scatter_add_(0, local.unsqueeze(1).expand(-1, D), he_flat[valid])
            ent_count.scatter_add_(0, local.unsqueeze(1),
                                   torch.ones(valid.sum(), 1, device=he_emb.device))
        ent_emb = ent_emb / ent_count.clamp(min=1)

        # HE ← sum(entity embeddings)
        he_update = torch.zeros_like(he_flat)
        for col in range(self.n_entities):
            col_ids = ent_flat[:, col]
            valid = (col_ids >= 0) & mask_flat
            if not valid.any():
                continue
            local = ent_to_local[col_ids[valid]]
            he_update[valid] += ent_emb[local]

        he_update = he_update.reshape(B, L, D)
        return self.norm(he_emb + self.linear(he_update))


class PairwiseConv(HypergraphConv):
    """Baseline A: only subject & object (2 entities)."""
    def __init__(self, d_hidden: int):
        super().__init__(d_hidden, n_entities=2)

    def forward(self, he_emb, entity_ids, mask):
        return super().forward(he_emb, entity_ids[:, :, :2], mask)


class THyN(nn.Module):
    def __init__(
        self,
        n_cont_features: int = 8,
        num_event_types: int = 18,
        d_type: int = 16,
        d_model: int = 64,
        d_hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        model_type: str = "thyn",
        encoder_type: str = "gru",
    ):
        super().__init__()
        self.model_type = model_type
        self.encoder_type = encoder_type

        self.type_emb = nn.Embedding(num_event_types, d_type, padding_idx=0)

        self.input_proj = nn.Sequential(
            nn.Linear(d_type + n_cont_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

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
                self.mamba_proj = nn.Linear(d_model, d_hidden) if d_model != d_hidden else nn.Identity()
                self.mamba_layers = nn.ModuleList([
                    Mamba(d_model=d_hidden, d_state=16, d_conv=4, expand=2)
                    for _ in range(n_layers)
                ])
                # Optional: add LayerNorm after each Mamba block for stability
                self.mamba_norms = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(n_layers)])
            except ImportError:
                raise ImportError("mamba-ssm not installed. Use encoder_type='gru'.")
        else:
            raise ValueError(f"Unknown encoder: {encoder_type}")

        if model_type == "thyn":
            self.graph_conv = HypergraphConv(d_hidden)
        elif model_type == "baseline_a":
            self.graph_conv = PairwiseConv(d_hidden)
        elif model_type == "baseline_b":
            self.graph_conv = None
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.classifier = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

    def encode_sequence(self, x):
        if self.encoder_type == "gru":
            out, _ = self.encoder(x)
            return out
        else:  # mamba
            h = self.mamba_proj(x)
            for layer, norm in zip(self.mamba_layers, self.mamba_norms):
                h = layer(h)
                h = norm(h)
                h = h.clamp(-50, 50)   # critical for numerical stability
            return h

    def forward(self, X_cont, event_type, entity_ids=None, mask=None):
        type_emb = self.type_emb(event_type)
        h = torch.cat([type_emb, X_cont], dim=-1)
        h = self.input_proj(h)
        h = self.encode_sequence(h)
        if self.graph_conv is not None and entity_ids is not None:
            h = self.graph_conv(h, entity_ids, mask)
        logits = self.classifier(h).squeeze(-1)
        return logits