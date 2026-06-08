"""Supervised KAIROS Baseline — GRU-based Temporal Graph Network for Provenance Graphs.

Adapts the core KAIROS architecture (Cheng et al., 2024 S&P) for supervised
event-level classification on the same chronological shard pipeline used by
HyperMamba. This enables a fair, apples-to-apples comparison.

Key KAIROS design choices preserved:
  - Per-entity GRU hidden states (vs. HyperMamba's Selective SSM)
  - Edge-level processing: each event updates source and destination entities
    independently (vs. HyperMamba's 3-role hyperedge aggregation)
  - Hierarchical Feature Hashing for node attributes (shared preprocessing)
  - Temporal state maintenance across chronological chunks (TBPTT)

Key differences from original KAIROS:
  - Supervised BCE classification (original uses unsupervised encoder-decoder)
  - Same HFH features flow through cont_proj (shared with HyperMamba)
  - Same entity state bank infrastructure for fair memory/compute comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KAIROSEntityBank(nn.Module):
    """Entity state bank for KAIROS — same infrastructure as HyperMamba's
    EntityStateBank but stores GRU hidden states instead of SSM states."""
    
    def __init__(self, num_entities, d_model):
        super().__init__()
        self.num_entities = num_entities
        self.d_model = d_model
        self.register_buffer("states", torch.zeros(num_entities, d_model))
        self.register_buffer("last_seen_time", torch.full((num_entities,), -1.0))
    
    def reset(self):
        self.states.zero_()
        self.last_seen_time.fill_(-1.0)
    
    def detach_(self):
        self.states = self.states.detach()
        self.states = torch.nan_to_num(self.states, nan=0.0)
        self.last_seen_time = self.last_seen_time.detach()


class KAIROSTemporalEncoder(nn.Module):
    """KAIROS-style temporal encoder: GRU update of entity states.
    
    For each event, the source and destination entity states are updated
    through a GRU cell conditioned on the event features and time delta.
    This is the core difference from HyperMamba's Selective SSM updater.
    
    KAIROS processes edges (src, dst) not hyperedges (subj, obj, obj2).
    To handle the 3-role provenance events fairly, we process:
      - Subject entity (always updated)
      - Object entity (updated if valid)
      - Object2 entity (updated if valid)
    Each role gets the same GRU but with a role embedding mixed in.
    """
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Role embeddings (same concept as HyperMamba, for fair comparison)
        self.role_emb = nn.Embedding(3, d_model)
        
        # GRU cell: input is [event_features || role_emb || time_encoding]
        # hidden state is the entity state
        self.gru = nn.GRUCell(
            input_size=d_model * 2 + d_model + d_model,  # event + role + time
            hidden_size=d_model
        )
        
        # Time encoding: project log_dt to d_model
        self.time_proj = nn.Linear(1, d_model)
    
    def forward(self, event_features, entity_states, log_dt, valid_mask):
        """
        event_features: (C, d_event) — encoded event
        entity_states:  (C, 3, d_model) — current states for [subj, obj, obj2]
        log_dt:         (C, 3, 1) — log time since last seen
        valid_mask:     (C, 3) — boolean, which entities are valid
        
        Returns: (C, 3, d_model) — updated states
        """
        C = event_features.size(0)
        device = event_features.device
        
        # Time encoding
        t_enc = self.time_proj(log_dt)  # (C, 3, d_model)
        
        # Role embeddings
        roles = torch.arange(3, device=device).unsqueeze(0).expand(C, 3)
        r_emb = self.role_emb(roles)  # (C, 3, d_model)
        
        # Expand event features for all 3 roles
        e_exp = event_features.unsqueeze(1).expand(-1, 3, -1)  # (C, 3, d_event)
        
        # GRU input: [event || role || time]
        gru_input = torch.cat([e_exp, r_emb, t_enc], dim=-1)  # (C, 3, d_event + 2*d_model)
        
        # Process all roles through the GRU
        # Reshape to (C*3, ...) for GRUCell
        gru_in_flat = gru_input.view(C * 3, -1)
        h_flat = entity_states.view(C * 3, -1)
        
        new_h_flat = self.gru(gru_in_flat, h_flat)  # (C*3, d_model)
        new_states = new_h_flat.view(C, 3, self.d_model)
        
        # Only update valid entities — keep old state for invalid ones
        new_states = torch.where(
            valid_mask.unsqueeze(-1),
            new_states,
            entity_states
        )
        
        return new_states


class SupervisedKAIROS(nn.Module):
    """Supervised KAIROS baseline for fair comparison with HyperMamba.
    
    Architecture:
      1. Event Encoder (shared design): event_type embedding + continuous features
      2. Entity State Bank: GRU hidden states per entity (vs. SSM states)
      3. GRU Temporal Update: update entity states per event (vs. Selective SSM)
      4. Classification: [aggregated_states || event_features] → MLP → logit
    
    Key difference from HyperMamba:
      - No AllSetAggregator (attention-based hyperedge aggregation)
      - GRU cell instead of Selective SSM (A_log, B, delta)
      - Entity states updated independently per role (no cross-entity mixing)
    
    This isolates the contribution of HyperMamba's two innovations:
      1. Selective SSM vs. GRU for temporal state
      2. Cross-entity hyperedge aggregation vs. independent updates
    """
    
    def __init__(self, num_entities, n_cont_features, num_event_types, d_model=128):
        super().__init__()
        self.d_model = d_model
        
        # Event Encoder — IDENTICAL to HyperMamba for fair comparison
        self.event_emb = nn.Embedding(num_event_types, d_model)
        self.cont_proj = nn.Linear(n_cont_features, d_model)
        self.input_norm = nn.LayerNorm(d_model * 2)
        self.d_event = d_model * 2
        
        # Entity State Bank
        self.bank = KAIROSEntityBank(num_entities, d_model)
        
        # KAIROS GRU-based temporal encoder
        self.temporal_encoder = KAIROSTemporalEncoder(d_model)
        
        # Project event features from d_event to d_model for classifier
        self.event_proj = nn.Linear(self.d_event, d_model)
        
        # Classifier Head — IDENTICAL to HyperMamba for fair comparison
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
        self.state_norm = nn.LayerNorm(d_model)
    
    def reset_bank(self):
        self.bank.reset()
    
    def detach_bank(self):
        self.bank.detach_()
    
    def forward(self, x_cont, event_type, entity_ids, timestamps):
        """
        Same interface as HyperMambaFull.forward() for drop-in compatibility.
        
        Args:
            x_cont:     (1, C, n_cont_features)
            event_type: (1, C)
            entity_ids: (1, C, 3)
            timestamps: (1, C)
        Returns:
            logits: (1, C)
        """
        x_cont = x_cont.squeeze(0)
        event_type = event_type.squeeze(0)
        entity_ids = entity_ids.squeeze(0)
        timestamps = timestamps.squeeze(0)
        
        C = x_cont.size(0)
        device = x_cont.device
        
        # 1. Encode event features (IDENTICAL to HyperMamba)
        e_emb = self.event_emb(event_type)
        c_emb = self.cont_proj(x_cont)
        f_e = self.input_norm(torch.cat([e_emb, c_emb], dim=-1))  # (C, d_event)
        
        # 2. Gather entity states and compute Δt
        safe_ids = entity_ids.clamp(min=0)
        valid_mask = entity_ids >= 0  # (C, 3)
        
        states = self.bank.states[safe_ids.view(-1)].view(C, 3, self.d_model)
        states = F.layer_norm(states, [self.d_model])
        states = states * valid_mask.unsqueeze(-1)
        
        last_seen = self.bank.last_seen_time[safe_ids.view(-1)].view(C, 3)
        t_curr = timestamps.unsqueeze(1).expand(C, 3)
        
        is_first = (last_seen < 0)
        dt = t_curr - last_seen
        dt = torch.where(is_first, torch.zeros_like(dt), dt)
        dt = torch.clamp(dt, min=0.0)
        log_dt = torch.log1p(dt).unsqueeze(-1)  # (C, 3, 1)
        
        # 3. GRU temporal update (KAIROS's core mechanism)
        new_states = self.temporal_encoder(f_e, states, log_dt, valid_mask)
        
        # 4. Scatter updated states back to bank
        flat_new_states = new_states.view(-1, self.d_model)
        flat_ids = safe_ids.view(-1)
        flat_valid = valid_mask.view(-1)
        
        valid_ids = flat_ids[flat_valid]
        valid_st = flat_new_states[flat_valid]
        
        valid_st_safe = torch.nan_to_num(valid_st.detach(), nan=0.0, posinf=1.0, neginf=-1.0)
        self.bank.states.scatter_(0, valid_ids.unsqueeze(1).expand(-1, self.d_model), valid_st_safe)
        self.bank.last_seen_time.scatter_(0, valid_ids, t_curr.reshape(-1)[flat_valid])
        
        # 5. Classification
        n_valid = valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        agg_states = (new_states * valid_mask.unsqueeze(-1)).sum(dim=1) / n_valid
        agg_states = self.state_norm(agg_states)
        
        event_feat = self.event_proj(f_e)
        cls_input = torch.cat([agg_states, event_feat], dim=-1)
        logits = self.classifier(cls_input).squeeze(-1)
        
        return logits.unsqueeze(0)
