"""
HyperMamba Minimal Prototype — Cross-Entity State Propagation.

Tests whether a global entity state bank with simple message passing
improves L1* novel-binary detection over per-subject-only models.

Key design:
  - Sequential processing: events processed one-at-a-time in chronological order
  - State bank: each entity (process/file/socket) has a hidden state vector
  - On each event: gather states of participating entities, aggregate,
    classify, then propagate updated state back
  - Gated update with LayerNorm to prevent state explosion
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
        # Gate: controls how much new info enters the state (sigmoid → [0,1])
        self.gate = nn.Linear(d_model, d_model)
        self.state_norm = nn.LayerNorm(d_model)

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # --- State bank (managed manually, NOT a parameter) ---
        # We store it as a buffer so it moves to the right device,
        # but we'll replace it with a fresh tensor each epoch.
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
        Process a chronological chunk of events sequentially.

        Args:
            X_cont:     (1, chunk_size, n_cont_features)
            event_type: (1, chunk_size)
            entity_ids: (1, chunk_size, 3)  — [subj, obj, obj2]

        Returns:
            logits: (1, chunk_size)
        """
        # Remove batch dim (must be 1 for chronological processing)
        X_cont = X_cont.squeeze(0)          # (C, n_cont)
        event_type = event_type.squeeze(0)  # (C,)
        entity_ids = entity_ids.squeeze(0)  # (C, 3)
        chunk_size = X_cont.size(0)

        # Encode all event features at once (this IS parallelizable)
        feat = self.input_norm(
            self.cont_proj(X_cont) + self.etype_emb(event_type)
        )  # (C, d_model)

        # Pre-allocate output
        logits = torch.zeros(chunk_size, device=X_cont.device)
        zero_state = torch.zeros(self.d_model, device=X_cont.device)

        # Sequential state propagation
        active_states = {}
        
        for i in range(chunk_size):
            ids = entity_ids[i].tolist()  # (3,)
            valid_ids = [idx for idx in ids if idx >= 0]
            
            # 1. GATHER: fetch current states for participating entities
            if valid_ids:
                # Fetch from active_states dict (which has gradients) or fallback to global bank (detached)
                states_list = [active_states.get(idx, self.bank[idx]) for idx in valid_ids]
                states = torch.stack(states_list)   # (num_valid, d_model)
                agg_state = states.mean(dim=0)      # (d_model,)
            else:
                states = None
                agg_state = zero_state

            # 2. COMBINE: merge entity context with event features
            x_event = agg_state + feat[i]           # (d_model,)

            # 3. CLASSIFY
            logits[i] = self.classifier(x_event).squeeze(-1)

            # 4. PROPAGATE: gated state update for each participating entity
            if valid_ids:
                update = self.update_mlp(x_event)          # (d_model,)
                g = torch.sigmoid(self.gate(x_event))      # (d_model,)

                # new_state = LayerNorm((1-g) * old_state + g * update)
                new_states = self.state_norm(
                    (1.0 - g).unsqueeze(0) * states +
                    g.unsqueeze(0) * update.unsqueeze(0)
                )  # (num_valid, d_model)

                # Write to active_states dictionary (maintains computation graph, no 1.7GB memory copy!)
                for j, idx in enumerate(valid_ids):
                    active_states[idx] = new_states[j]

        # End of chunk: Write detached states back to global bank
        for idx, state in active_states.items():
            self.bank[idx] = state.detach()

        return logits.unsqueeze(0)  # (1, chunk_size)
