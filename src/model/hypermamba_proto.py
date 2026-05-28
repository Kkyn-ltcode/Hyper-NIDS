"""
HyperMamba Minimal Prototype — Cross-Entity State Propagation.

Vectorized version: the entire chunk is processed in ~10 GPU operations
instead of a Python for-loop. States propagate ACROSS chunks (via the
persistent bank) but not WITHIN a chunk. This matches TGN's batching
strategy and is ~100-1000x faster than the sequential version.

Key trade-off: if entity A appears at positions 5 and 900 in the same
chunk, position 900 sees A's state from the PREVIOUS chunk, not the
update from position 5. With chunk_size=4096 out of 32M events, this
is a negligible loss.
"""

import torch
import torch.nn as nn


class HyperMambaProto(nn.Module):

    def __init__(self, num_entities, n_cont_features, num_event_types,
                 d_model=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_entities = num_entities

        # --- Input encoding ---
        self.etype_emb = nn.Embedding(num_event_types, d_model, padding_idx=0)
        self.cont_proj = nn.Linear(n_cont_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # --- State update (gated residual) ---
        self.update_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.gate = nn.Linear(d_model, d_model)
        self.state_norm = nn.LayerNorm(d_model)

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # --- State bank ---
        self.register_buffer("bank", torch.zeros(num_entities, d_model))

    def reset_bank(self):
        """Call at the start of each epoch."""
        self.bank = torch.zeros(
            self.num_entities, self.d_model,
            device=self.bank.device, dtype=self.bank.dtype)

    def detach_bank(self):
        """Call after each chunk to cut the BPTT graph."""
        self.bank = self.bank.detach()

    def forward(self, X_cont, event_type, entity_ids):
        """
        Fully vectorized forward pass — NO Python for-loop.

        Args:
            X_cont:     (1, C, n_cont)
            event_type: (1, C)
            entity_ids: (1, C, 3)

        Returns:
            logits: (1, C)
        """
        X_cont = X_cont.squeeze(0)          # (C, n_cont)
        event_type = event_type.squeeze(0)  # (C,)
        entity_ids = entity_ids.squeeze(0)  # (C, 3)
        C = X_cont.size(0)

        # 1. ENCODE all event features at once
        feat = self.input_norm(
            self.cont_proj(X_cont) + self.etype_emb(event_type)
        )  # (C, d)

        # 2. GATHER: fetch states for all entities in all events
        valid_mask = entity_ids >= 0                        # (C, 3) bool
        safe_ids = entity_ids.clamp(min=0)                  # (C, 3) — safe for indexing
        all_states = self.bank[safe_ids]                    # (C, 3, d)
        all_states = all_states * valid_mask.unsqueeze(-1)  # zero out invalid slots

        # 3. AGGREGATE: mean over valid entity slots per event
        n_valid = valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # (C, 1)
        agg_states = all_states.sum(dim=1) / n_valid  # (C, d)

        # 4. COMBINE event features with aggregated entity context
        x_event = agg_states + feat  # (C, d)

        # 5. CLASSIFY all events at once
        logits = self.classifier(x_event).squeeze(-1)  # (C,)

        # 6. PROPAGATE: compute gated state updates
        update = self.update_mlp(x_event)           # (C, d)
        g = torch.sigmoid(self.gate(x_event))       # (C, d)

        # Scatter updates back to bank for each entity slot
        for slot in range(3):
            slot_ids = entity_ids[:, slot]           # (C,)
            slot_valid = slot_ids >= 0               # (C,)
            if not slot_valid.any():
                continue

            v_ids = slot_ids[slot_valid]             # (N,)
            old_states = self.bank[v_ids]            # (N, d)
            slot_g = g[slot_valid]                   # (N, d)
            slot_upd = update[slot_valid]            # (N, d)

            new_states = self.state_norm(
                (1.0 - slot_g) * old_states + slot_g * slot_upd
            )  # (N, d)

            # Write back — for duplicate IDs, last write wins
            # (equivalent to TGN's "most recent message" strategy)
            self.bank[v_ids] = new_states

        return logits.unsqueeze(0)  # (1, C)
