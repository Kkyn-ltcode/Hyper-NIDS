"""HyperMamba Full Architecture — SSM-Driven Taint Propagation on Provenance Hypergraphs."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EntityStateBank(nn.Module):
    def __init__(self, num_entities, d_model):
        super().__init__()
        self.num_entities = num_entities
        self.d_model = d_model
        
        # State tensors (not parameters, updated in-place)
        self.register_buffer("states", torch.zeros(num_entities, d_model))
        # Initialize last_seen_time to -1.0 so we know if an entity is seen for the first time
        self.register_buffer("last_seen_time", torch.full((num_entities,), -1.0))
        
    def reset(self):
        self.states.zero_()
        self.last_seen_time.fill_(-1.0)
        
    def detach_(self):
        self.states = self.states.detach()
        self.last_seen_time = self.last_seen_time.detach()


class AllSetAggregator(nn.Module):
    def __init__(self, d_model, d_event, n_roles=3):
        super().__init__()
        # h_i = [state, role_emb, dt_enc] -> total dimension = d_model + d_model + 1
        self.d_h = d_model * 2 + 1
        
        self.role_emb = nn.Embedding(n_roles, d_model)
        
        # Dynamic query generator from event features
        self.q_proj = nn.Linear(d_event, d_model)
        
        # Key and Value projections for entities
        self.k_proj = nn.Linear(self.d_h, d_model)
        self.v_proj = nn.Linear(self.d_h, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, event_features, entity_states, log_dt):
        """
        event_features: (batch, d_event)
        entity_states: (batch, 3, d_model)
        log_dt: (batch, 3, 1)
        """
        batch_size = event_features.size(0)
        device = event_features.device
        
        # 1. Role embeddings
        roles = torch.arange(3, device=device).unsqueeze(0).expand(batch_size, 3) # (batch, 3)
        r_emb = self.role_emb(roles) # (batch, 3, d_model)
        
        # 2. Construct entity inputs h_i
        # h_i shape: (batch, 3, 2*d_model + 1)
        h = torch.cat([entity_states, r_emb, log_dt], dim=-1)
        
        # 3. Dynamic query Q(t)
        # q shape: (batch, 1, d_model)
        q = self.q_proj(event_features).unsqueeze(1)
        
        # 4. Keys and Values
        k = self.k_proj(h) # (batch, 3, d_model)
        v = self.v_proj(h) # (batch, 3, d_model)
        
        # 5. Attention
        # scores: (batch, 1, 3)
        scores = torch.bmm(q, k.transpose(1, 2)) / (k.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        
        # 6. Aggregate
        # agg: (batch, 1, d_model)
        agg = torch.bmm(attn, v)
        
        # x_e: (batch, d_model)
        x_e = self.out_proj(agg.squeeze(1))
        
        return x_e, r_emb


class SelectiveSSMUpdater(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # A: learned diagonal state decay matrix
        # Initialize so that A = -exp(A_log) is in [-0.1, -1.0] range
        # This ensures moderate decay rates at initialization
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model) * 0.9 + 0.1))
        
        # B: input projection matrix
        self.proj_B = nn.Linear(d_model, d_model)
        
        # Delta: input-dependent discretization step
        self.proj_delta = nn.Linear(d_model * 2, d_model)
        
        # LayerNorm on the output to prevent state drift
        self.out_norm = nn.LayerNorm(d_model)
        
    def forward(self, x_e, r_emb, entity_states, log_dt):
        """
        x_e: (batch, d_model)
        r_emb: (batch, 3, d_model)
        entity_states: (batch, 3, d_model)
        log_dt: (batch, 3, 1)
        """
        batch_size = x_e.size(0)
        
        # A matrix: clamp A_log to prevent extreme decay values
        A = -torch.exp(self.A_log.clamp(max=2.0))
        
        # Expand x_e to match entities: (batch, 3, d_model)
        x_e_expand = x_e.unsqueeze(1).expand(-1, 3, -1)
        
        # B matrix: (batch, d_model) -> expand to (batch, 3, d_model)
        B = self.proj_B(x_e).unsqueeze(1).expand(-1, 3, -1)
        
        # Compute Delta_i using x_e and role embedding
        # delta_input: (batch, 3, 2*d_model)
        delta_input = torch.cat([x_e_expand, r_emb], dim=-1)
        
        # delta_i: (batch, 3, d_model)
        # softplus ensures delta is positive
        delta_raw = F.softplus(self.proj_delta(delta_input))
        
        # Incorporate actual time elapsed (log_dt) into the discretization step
        # Cap log_dt to prevent extreme discretization steps for large time gaps
        log_dt_clamped = log_dt.clamp(max=10.0)
        delta_i = delta_raw * (log_dt_clamped + 1.0)
        
        # Cap delta_i to prevent exp() overflow in A_bar computation
        delta_i = delta_i.clamp(max=10.0)
        
        # Discrete A: exp(A * Delta)
        # A is negative, delta_i is positive, so A*delta_i is negative -> A_bar in (0, 1)
        A_bar = torch.exp(A.view(1, 1, -1) * delta_i)
        
        # Discrete B approximation: Delta * B
        B_bar = delta_i * B
        
        # SSM Update: S(t) = A_bar * S(t-1) + B_bar * x_e
        new_states = A_bar * entity_states + B_bar * x_e_expand
        
        # Normalize to prevent unbounded state growth across chunks
        C = new_states.size(0)
        new_states = self.out_norm(new_states.view(-1, self.d_model)).view(C, 3, self.d_model)
        
        return new_states


class HyperMambaFull(nn.Module):
    BANK_DECAY = 0.999  # Gradual state fade to prevent unbounded accumulation
    
    def __init__(self, num_entities, n_cont_features, num_event_types, d_model=128):
        super().__init__()
        self.d_model = d_model
        
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
        
        # Classifier Head: x_e + updated subj state + updated obj state = 3 * d_model
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1)
        )
        
    def reset_bank(self):
        self.bank.reset()
        
    def detach_bank(self):
        self.bank.detach_()
        # Apply decay to prevent unbounded state accumulation across chunks
        self.bank.states.mul_(self.BANK_DECAY)
        # Clamp states as a safety net against any residual drift
        self.bank.states.clamp_(-10.0, 10.0)
        
    def forward(self, x_cont, event_type, entity_ids, timestamps):
        """
        All inputs arrive as (1, C, ...) from DataLoader with batch_size=1.
        We squeeze dim 0, process as (C, ...), and unsqueeze back on output.

        Args:
            x_cont:     (1, C, n_cont_features)
            event_type: (1, C)
            entity_ids: (1, C, 3) - [subj, obj, obj2]
            timestamps: (1, C) - event time in seconds

        Returns:
            logits: (1, C)
        """
        # Squeeze the DataLoader batch dimension
        x_cont = x_cont.squeeze(0)         # (C, n_cont)
        event_type = event_type.squeeze(0)  # (C,)
        entity_ids = entity_ids.squeeze(0)  # (C, 3)
        timestamps = timestamps.squeeze(0)  # (C,)
        
        C = x_cont.size(0)
        device = x_cont.device
        
        # 1. Encode Event
        e_emb = self.event_emb(event_type)  # (C, d_model)
        c_emb = self.cont_proj(x_cont)      # (C, d_model)
        f_e = self.input_norm(torch.cat([e_emb, c_emb], dim=-1))  # (C, d_event)
        
        # 2. Gather entity states and compute Delta t
        # Clamp entity IDs to valid range for indexing (mask invalid later)
        safe_ids = entity_ids.clamp(min=0)
        valid_mask = entity_ids >= 0  # (C, 3)
        
        # States: (C, 3, d_model)
        states = self.bank.states[safe_ids.view(-1)].view(C, 3, self.d_model)
        states = states * valid_mask.unsqueeze(-1)  # zero out invalid entities
        
        # Timestamps: (C, 3)
        last_seen = self.bank.last_seen_time[safe_ids.view(-1)].view(C, 3)
        t_curr = timestamps.unsqueeze(1).expand(C, 3)
        
        # Compute dt: time since entity was last seen
        is_first = (last_seen < 0)
        dt = t_curr - last_seen
        dt = torch.where(is_first, torch.zeros_like(dt), dt)
        dt = torch.clamp(dt, min=0.0)
        
        # Log-scale to handle range from microseconds to days
        log_dt = torch.log1p(dt).unsqueeze(-1)  # (C, 3, 1)
        
        # 3. Hyperedge Aggregation (V -> E)
        x_e, r_emb = self.aggregator(f_e, states, log_dt)
        
        # 4. Selective SSM State Update (E -> V)
        new_states = self.updater(x_e, r_emb, states, log_dt)
        
        # 5. Scatter updated states and timestamps back to bank
        flat_new_states = new_states.view(-1, self.d_model)
        flat_ids = safe_ids.view(-1)
        flat_valid = valid_mask.view(-1)
        
        valid_ids = flat_ids[flat_valid]
        valid_states = flat_new_states[flat_valid]
        
        # Clamp states before writing to bank to prevent NaN propagation
        valid_states = valid_states.clamp(-10.0, 10.0)
        
        self.bank.states.scatter_(0, valid_ids.unsqueeze(1).expand(-1, self.d_model), valid_states)
        self.bank.last_seen_time.scatter_(0, valid_ids, t_curr.reshape(-1)[flat_valid])
        
        # 6. Classification
        subj_state = new_states[:, 0, :]  # (C, d_model)
        obj_state = new_states[:, 1, :]   # (C, d_model)
        
        cls_input = torch.cat([x_e, subj_state, obj_state], dim=-1)  # (C, 3*d_model)
        logits = self.classifier(cls_input).squeeze(-1)  # (C,)
        
        return logits.unsqueeze(0)  # (1, C)
