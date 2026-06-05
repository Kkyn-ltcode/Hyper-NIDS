"""HyperMamba Full Architecture — SSM-Driven Taint Propagation on Provenance Hypergraphs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EntityStateBank(nn.Module):
    def __init__(self, num_entities, d_model):
        super().__init__()
        self.num_entities = num_entities
        self.d_model = d_model
        
        # State tensors (not parameters, updated in-place)
        self.register_buffer("states", torch.zeros(num_entities, d_model))
        self.register_buffer("last_seen_time", torch.full((num_entities,), -1.0))
        
    def reset(self):
        self.states.zero_()
        self.last_seen_time.fill_(-1.0)
        
    def detach_(self):
        """Detach states from computation graph at TBPTT boundary.
        
        NO bank_decay: the SSM's A matrix provides principled, learned temporal
        decay. Adding a global 0.95 multiplier was killing long-range taint
        propagation (0.95^8 = 0.66 signal loss over the 8-chunk attack gap).
        """
        self.states = self.states.detach()
        # Clean any NaN that slipped through — prevents cascade across chunks
        self.states = torch.nan_to_num(self.states, nan=0.0)
        self.last_seen_time = self.last_seen_time.detach()


class AllSetAggregator(nn.Module):
    """V→E: Aggregate entity states into a hyperedge representation using
    dynamic attention conditioned on event features."""
    
    def __init__(self, d_model, d_event, n_entity_features=0, n_roles=3):
        super().__init__()
        self.d_model = d_model
        self.n_entity_features = n_entity_features
        # h_i = [state || role_emb || dt_enc || entity_feat]
        self.d_h = d_model * 2 + 1 + n_entity_features
        
        self.role_emb = nn.Embedding(n_roles, d_model)
        
        # Dynamic query from event features
        self.q_proj = nn.Linear(d_event, d_model)
        
        # Key and Value projections for entities
        self.k_proj = nn.Linear(self.d_h, d_model)
        self.v_proj = nn.Linear(self.d_h, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5
        
    def forward(self, event_features, entity_states, log_dt, valid_mask, entity_feat=None):
        """
        event_features: (C, d_event)
        entity_states:  (C, 3, d_model)
        log_dt:         (C, 3, 1)
        valid_mask:     (C, 3) boolean mask
        entity_feat:    (C, 3, n_entity_features) or None

        Returns: x_e (C, d_model), r_emb (C, 3, d_model)
        """
        C = event_features.size(0)
        device = event_features.device
        
        roles = torch.arange(3, device=device).unsqueeze(0).expand(C, 3)
        r_emb = self.role_emb(roles)  # (C, 3, d_model)
        
        # h_i = [state || role || log_dt || entity_feat]
        parts = [entity_states, r_emb, log_dt]
        if entity_feat is not None:
            parts.append(entity_feat)
        h = torch.cat(parts, dim=-1)  # (C, 3, d_h)
        
        # Dynamic query from event features
        q = self.q_proj(event_features).unsqueeze(1)  # (C, 1, d_model)
        k = self.k_proj(h)  # (C, 3, d_model)
        v = self.v_proj(h)  # (C, 3, d_model)
        
        # Scaled dot-product attention
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (C, 1, 3)
        scores = scores.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        # In case all entities are invalid (shouldn't happen, but just in case), softmax gives NaNs.
        # Replace NaNs with 0.
        attn = torch.nan_to_num(attn, nan=0.0)
        agg = torch.bmm(attn, v).squeeze(1)  # (C, d_model)
        
        x_e = self.out_proj(agg)
        return x_e, r_emb


class SelectiveSSMUpdater(nn.Module):
    """E→V: Update entity states using Selective SSM equations.
    
    Key design decisions:
    - A initialized with HiPPO-style values for stable long-range memory
    - Delta bias initialized small so initial discretization steps are small
    - NO LayerNorm on output (it kills A_log gradient by normalizing away the decay)
    - Residual connection: new_state = gate * ssm_update + (1-gate) * old_state
    """
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # A: HiPPO-style initialization (log-spaced from -1 to -d)
        # This gives a range of time scales: some dimensions forget fast, others persist
        A_init = torch.log(torch.linspace(0.1, 2.0, d_model))
        self.A_log = nn.Parameter(A_init)
        
        # B: input projection, now role-specific to prevent over-smoothing
        self.proj_B = nn.Linear(d_model * 2, d_model)
        
        # Delta: input-dependent discretization step
        # Initialize bias to produce small delta values (around 0.1-1.0)
        self.proj_delta = nn.Linear(d_model * 2, d_model)
        with torch.no_grad():
            self.proj_delta.bias.fill_(-2.0)  # softplus(-2) ≈ 0.13
        
        # Gating: how much of the SSM update vs old state to keep
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        with torch.no_grad():
            # CRITICAL: Initialize gate to be mostly closed (sigmoid(-3) ≈ 0.04)
            # This forces the model to actively LEARN to propagate state, preventing
            # catastrophic over-smoothing (diffusion of state) at initialization.
            self.gate_proj.bias.fill_(-3.0)
        
    def forward(self, x_e, r_emb, entity_states, log_dt):
        """
        x_e:            (C, d_model)  — hyperedge representation
        r_emb:          (C, 3, d_model) — role embeddings
        entity_states:  (C, 3, d_model) — current states
        log_dt:         (C, 3, 1) — log-scaled time since last seen
        Returns:        (C, 3, d_model) — updated states
        """
        # A: negative decay rates, clamped for stability
        A = -torch.exp(self.A_log.clamp(-4.0, 4.0))  # (d_model,)
        
        x_e_exp = x_e.unsqueeze(1).expand(-1, 3, -1)  # (C, 3, d_model)
        
        # B: input-to-state projection (Role-Specific)
        # If we don't include r_emb here, all 3 entities get the EXACT same
        # update direction, forcing them to become identical and destroying graph structure.
        B_input = torch.cat([x_e_exp, r_emb], dim=-1)  # (C, 3, 2*d_model)
        B = self.proj_B(B_input)  # (C, 3, d_model)
        
        # Delta: input-dependent discretization step
        delta_raw = F.softplus(self.proj_delta(B_input))  # (C, 3, d_model), positive
        
        # Incorporate time elapsed — but scale gently
        # Use log_dt as an additive offset, not multiplicative
        # This prevents extreme amplification for large time gaps
        dt_scale = 1.0 + 0.1 * log_dt  # (C, 3, 1) — gentle scaling, ≈1.0 to 1.7
        delta_i = (delta_raw * dt_scale).clamp(max=5.0)  # cap for numerical stability
        
        # Discretize A and B
        A_bar = torch.exp(A.view(1, 1, -1) * delta_i)  # (C, 3, d_model), in (0, 1)
        B_bar = delta_i * B
        
        # SSM update: s(t) = A_bar * s(t-1) + B_bar * x_e
        ssm_state = A_bar * entity_states + B_bar * x_e_exp
        
        # Gated residual: blend SSM update with old state
        # This gives the model a stable path early in training
        gate_input = torch.cat([x_e_exp, r_emb], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))  # (C, 3, d_model)
        
        new_states = gate * ssm_state + (1.0 - gate) * entity_states
        
        return new_states


class HyperMambaFull(nn.Module):
    def __init__(self, num_entities, n_cont_features, num_event_types, d_model=128,
                 use_state=True, cross_entity=True):
        super().__init__()
        self.d_model = d_model
        self.use_state = use_state
        self.cross_entity = cross_entity
        
        # Event Encoder
        self.event_emb = nn.Embedding(num_event_types, d_model)
        self.cont_proj = nn.Linear(n_cont_features, d_model)
        self.input_norm = nn.LayerNorm(d_model * 2)
        self.d_event = d_model * 2
        
        # State Bank
        self.bank = EntityStateBank(num_entities, d_model)
        
        # Architecture Components
        self.aggregator = AllSetAggregator(d_model, self.d_event)
        self.updater = SelectiveSSMUpdater(d_model)
        
        # Project event features from d_event to d_model for classifier
        self.event_proj = nn.Linear(self.d_event, d_model)
        
        # Classifier Head: [agg_states, event_feat]
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Normalize aggregated states before classifier
        self.state_norm = nn.LayerNorm(d_model)
        
    def reset_bank(self):
        self.bank.reset()
        
    def detach_bank(self):
        self.bank.detach_()
        
    def forward(self, x_cont, event_type, entity_ids, timestamps):
        """
        All inputs arrive as (1, C, ...) from DataLoader with batch_size=1.
        We squeeze dim 0, process as (C, ...), and unsqueeze back.

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
        
        # 1. Encode event features
        e_emb = self.event_emb(event_type)
        c_emb = self.cont_proj(x_cont)
        f_e = self.input_norm(torch.cat([e_emb, c_emb], dim=-1))  # (C, d_event)
        
        # 2. Gather entity states and compute Δt
        safe_ids = entity_ids.clamp(min=0)
        valid_mask = entity_ids >= 0  # (C, 3)
        
        states = self.bank.states[safe_ids.view(-1)].view(C, 3, self.d_model)
        # Normalize gathered states to prevent unbounded accumulation from
        # causing attention score overflow (exp(>88) = inf in float32).
        # Hub entities (pid 1, shell) get updated ~157k times; without this,
        # k_proj produces scores that overflow softmax.
        states = F.layer_norm(states, [self.d_model])
        states = states * valid_mask.unsqueeze(-1)
        
        last_seen = self.bank.last_seen_time[safe_ids.view(-1)].view(C, 3)
        t_curr = timestamps.unsqueeze(1).expand(C, 3)
        
        is_first = (last_seen < 0)
        dt = t_curr - last_seen
        dt = torch.where(is_first, torch.zeros_like(dt), dt)
        dt = torch.clamp(dt, min=0.0)
        log_dt = torch.log1p(dt).unsqueeze(-1)  # (C, 3, 1)
        
        if not self.use_state:
            # Ablation: Pure event feature classification, no state tracking
            event_feat = self.event_proj(f_e)
            # Create dummy states to keep classifier size consistent
            dummy_state = torch.zeros_like(states[:, 0, :])
            cls_input = torch.cat([dummy_state, event_feat], dim=-1)
            logits = self.classifier(cls_input).squeeze(-1)
            return logits.unsqueeze(0)
            
        # 3. Hyperedge Aggregation (V → E)
        if self.cross_entity:
            x_e, r_emb = self.aggregator(f_e, states, log_dt, valid_mask)
        else:
            # Ablation: No cross-entity aggregation.
            # Represent hyperedge purely by event features.
            # We still need x_e and r_emb for the updater.
            x_e = self.event_proj(f_e)
            roles = torch.arange(3, device=device).unsqueeze(0).expand(C, 3)
            r_emb = self.aggregator.role_emb(roles)
        
        # 4. Selective SSM State Update (E → V)
        new_states = self.updater(x_e, r_emb, states, log_dt)
        
        # 5. Scatter updated states back to bank
        flat_new_states = new_states.view(-1, self.d_model)
        flat_ids = safe_ids.view(-1)
        flat_valid = valid_mask.view(-1)
        
        valid_ids = flat_ids[flat_valid]
        valid_st = flat_new_states[flat_valid]
        
        # MUST detach before saving to bank to truncate BPTT at chunk boundary!
        # Guard against NaN before scatter — once NaN enters the bank, it cascades
        # to every future chunk that reads from the corrupted entity.
        valid_st_safe = torch.nan_to_num(valid_st.detach(), nan=0.0, posinf=1.0, neginf=-1.0)
        self.bank.states.scatter_(0, valid_ids.unsqueeze(1).expand(-1, self.d_model), valid_st_safe)
        self.bank.last_seen_time.scatter_(0, valid_ids, t_curr.reshape(-1)[flat_valid])
        
        # 6. Classification: use POST-UPDATE states so gradient flows through
        #    the SSM updater and Aggregator. Using pre-update `states` (from the
        #    buffer) would have requires_grad=False, cutting all gradient to the
        #    SSM (A_log, proj_B, proj_delta, gate_proj) and Aggregator (q/k/v_proj).
        n_valid = valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        # Mask out invalid entities BEFORE summing — the SSM still computes
        # new_states for slot 2 (null UUID) via gate*B_bar*x_e, producing
        # non-zero values. Without masking, we sum 3 entities but divide by 2.
        agg_states = (new_states * valid_mask.unsqueeze(-1)).sum(dim=1) / n_valid  # (C, d_model)
        
        # Apply LayerNorm to stabilize states before classifier
        if self.use_state:
            agg_states = self.state_norm(agg_states)
            
        event_feat = self.event_proj(f_e)                # (C, d_model)
        
        cls_input = torch.cat([agg_states, event_feat], dim=-1)  # (C, 2*d_model)
        logits = self.classifier(cls_input).squeeze(-1)  # (C,)
        
        return logits.unsqueeze(0)  # (1, C)
