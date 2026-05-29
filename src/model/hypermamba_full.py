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
        
        # A: learned diagonal state decay matrix (initialized negative for stability)
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model) * 0.5 + 0.5))
        
        # B: input projection matrix
        self.proj_B = nn.Linear(d_model, d_model)
        
        # Delta: input-dependent discretization step
        self.proj_delta = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x_e, r_emb, entity_states, log_dt):
        """
        x_e: (batch, d_model)
        r_emb: (batch, 3, d_model)
        entity_states: (batch, 3, d_model)
        log_dt: (batch, 3, 1)
        """
        batch_size = x_e.size(0)
        
        # A matrix: (d_model,)
        A = -torch.exp(self.A_log)
        
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
        # If log_dt is 0, we still want a minimum step size, so we add 1.0 (since log_dt is log(1+dt))
        # log_dt has shape (batch, 3, 1). We broadcast it to (batch, 3, d_model)
        # Ensure log_dt is strictly positive to prevent 0 step sizes in initialization
        delta_i = delta_raw * (log_dt + 1.0)
        
        # Discrete A: exp(A * Delta)
        # A is (d_model,), delta_i is (batch, 3, d_model) -> elementwise multiply
        A_bar = torch.exp(A.view(1, 1, -1) * delta_i)
        
        # Discrete B approximation: Delta * B
        B_bar = delta_i * B
        
        # SSM Update: S(t) = A_bar * S(t-1) + B_bar * x_e
        new_states = A_bar * entity_states + B_bar * x_e_expand
        
        return new_states


class HyperMambaFull(nn.Module):
    def __init__(self, num_entities, n_cont_features, num_event_types, d_model=128):
        super().__init__()
        self.d_model = d_model
        
        # Event Encoder
        self.event_emb = nn.Embedding(num_event_types, d_model)
        self.cont_proj = nn.Linear(n_cont_features, d_model)
        self.d_event = d_model * 2
        
        # State Bank
        self.bank = EntityStateBank(num_entities, d_model)
        
        # Architecture Components
        self.aggregator = AllSetAggregator(d_model, self.d_event)
        self.updater = SelectiveSSMUpdater(d_model)
        
        # Classifier Head (takes x_e + subject state + object state)
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
        
    def forward(self, x_cont, event_type, entity_ids, timestamps):
        """
        x_cont: (batch, n_cont_features)
        event_type: (batch,)
        entity_ids: (batch, 3) - [subj, obj, obj2]
        timestamps: (batch,) - current event time in seconds
        """
        batch_size = x_cont.size(0)
        device = x_cont.device
        
        # 1. Encode Event
        e_emb = self.event_emb(event_type)
        c_emb = self.cont_proj(x_cont)
        f_e = torch.cat([e_emb, c_emb], dim=-1) # (batch, d_event)
        
        # 2. Gather entity states and compute Delta t
        # States
        states_flat = self.bank.states[entity_ids.view(-1)]
        states = states_flat.view(batch_size, 3, self.d_model)
        
        # Timestamps
        last_seen_flat = self.bank.last_seen_time[entity_ids.view(-1)]
        last_seen = last_seen_flat.view(batch_size, 3)
        
        # Expand timestamps to (batch, 3)
        t_curr = timestamps.unsqueeze(1).expand(batch_size, 3)
        
        # Calculate dt. If last_seen is -1.0, it's the first time, set dt to 0.0
        is_first = (last_seen < 0)
        dt = t_curr - last_seen
        dt = torch.where(is_first, torch.zeros_like(dt), dt)
        
        # Cap negative dt (can happen due to minor out-of-order logs in real systems)
        dt = torch.clamp(dt, min=0.0)
        
        # Log scaling to handle huge range of time (microseconds to days)
        log_dt = torch.log1p(dt).unsqueeze(-1) # (batch, 3, 1)
        
        # 3. Hyperedge Aggregation (V -> E)
        x_e, r_emb = self.aggregator(f_e, states, log_dt)
        
        # 4. Selective SSM State Update (E -> V)
        new_states = self.updater(x_e, r_emb, states, log_dt)
        
        # 5. Scatter updated states and timestamps back to bank
        # We need to handle duplicate entities within the same batch.
        # Since we are doing TBPTT with batch_size=1, this is trivial, 
        # but scatter handles it safely anyway.
        flat_new_states = new_states.view(-1, self.d_model)
        flat_entity_ids = entity_ids.view(-1)
        
        # Ignore padding entities (-1)
        valid_mask = flat_entity_ids >= 0
        valid_ids = flat_entity_ids[valid_mask]
        valid_states = flat_new_states[valid_mask]
        
        self.bank.states.scatter_(0, valid_ids.unsqueeze(1).expand(-1, self.d_model), valid_states)
        self.bank.last_seen_time.scatter_(0, valid_ids, t_curr.reshape(-1)[valid_mask])
        
        # 6. Classification
        # We classify based on the event (x_e) and the NEW states of the primary actors (subj, obj)
        subj_state = new_states[:, 0, :]
        obj_state = new_states[:, 1, :]
        
        cls_input = torch.cat([x_e, subj_state, obj_state], dim=-1)
        logits = self.classifier(cls_input)
        
        return logits
