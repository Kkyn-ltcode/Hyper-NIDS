"""
HyperMamba Minimal Prototype — Cross-Entity State Propagation.

v3: Fixed state corruption with concat (not add) + bank decay.

Key changes from v2:
  - Concat entity state with event features (not add). This lets the
    classifier learn to IGNORE state when it's noisy early in training.
  - Bank decay (0.95 per chunk) prevents unbounded state accumulation.
  - --no_state ablation flag to measure the contribution of cross-entity
    state propagation vs. event-features-only.
"""

import torch
import torch.nn as nn


class HyperMambaProto(nn.Module):

    def __init__(self, num_entities, n_cont_features, num_event_types,
                 d_model=128, dropout=0.1, use_state=True, bank_decay=0.95):
        super().__init__()
        self.d_model = d_model
        self.num_entities = num_entities
        self.use_state = use_state
        self.bank_decay = bank_decay

        # --- Input encoding ---
        self.etype_emb = nn.Embedding(num_event_types, d_model, padding_idx=0)
        self.cont_proj = nn.Linear(n_cont_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # --- Classifier: takes concatenated [state || feat] or [feat] ---
        clf_input_dim = 2 * d_model if use_state else d_model
        self.classifier = nn.Sequential(
            nn.Linear(clf_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        if use_state:
            # --- State update (gated residual) ---
            self.update_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.gate = nn.Linear(d_model, d_model)
            self.state_norm = nn.LayerNorm(d_model)

            # --- State bank ---
            self.register_buffer("bank", torch.zeros(num_entities, d_model))

    def reset_bank(self):
        if self.use_state:
            self.bank = torch.zeros(
                self.num_entities, self.d_model,
                device=self.bank.device, dtype=self.bank.dtype)

    def detach_bank(self):
        if self.use_state:
            # Decay old states so they fade naturally
            self.bank = self.bank.detach() * self.bank_decay

    def forward(self, X_cont, event_type, entity_ids):
        """
        Fully vectorized forward pass.

        Args:
            X_cont:     (1, C, n_cont)
            event_type: (1, C)
            entity_ids: (1, C, 3)

        Returns:
            logits: (1, C)
        """
        X_cont = X_cont.squeeze(0)
        event_type = event_type.squeeze(0)
        entity_ids = entity_ids.squeeze(0)
        C = X_cont.size(0)

        # 1. ENCODE all event features
        feat = self.input_norm(
            self.cont_proj(X_cont) + self.etype_emb(event_type)
        )  # (C, d)

        if not self.use_state:
            # No-state ablation: classify on features only
            logits = self.classifier(feat).squeeze(-1)
            return logits.unsqueeze(0)

        # 2. GATHER entity states
        valid_mask = entity_ids >= 0
        safe_ids = entity_ids.clamp(min=0)
        all_states = self.bank[safe_ids]                    # (C, 3, d)
        all_states = all_states * valid_mask.unsqueeze(-1)

        n_valid = valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        agg_states = all_states.sum(dim=1) / n_valid        # (C, d)

        # 3. CONCATENATE state + features (NOT add)
        x_combined = torch.cat([agg_states, feat], dim=-1)  # (C, 2d)

        # 4. CLASSIFY
        logits = self.classifier(x_combined).squeeze(-1)    # (C,)

        # 5. PROPAGATE: gated state updates
        # The update is based on features only (not combined), to prevent
        # the update from depending on potentially noisy old states
        update = self.update_mlp(feat)                      # (C, d)
        g = torch.sigmoid(self.gate(feat))                  # (C, d)

        for slot in range(3):
            slot_ids = entity_ids[:, slot]
            slot_valid = slot_ids >= 0
            if not slot_valid.any():
                continue

            v_ids = slot_ids[slot_valid]
            old_states = self.bank[v_ids]
            slot_g = g[slot_valid]
            slot_upd = update[slot_valid]

            new_states = self.state_norm(
                (1.0 - slot_g) * old_states + slot_g * slot_upd
            )

            self.bank[v_ids] = new_states

        return logits.unsqueeze(0)
