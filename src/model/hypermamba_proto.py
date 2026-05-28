import torch
import torch.nn as nn

class HyperMambaProto(nn.Module):
    """
    Minimal prototype to test cross-entity state propagation.
    Uses sequential for-loop to update states chronologically.
    """
    def __init__(self, num_entities, n_cont_features, num_event_types, d_model=128):
        super().__init__()
        self.d_model = d_model
        
        self.etype_emb = nn.Embedding(num_event_types, d_model, padding_idx=0)
        self.cont_proj = nn.Linear(n_cont_features, d_model)
        
        # Simple update rule
        self.update_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Global state bank. Registered as buffer so it moves to device automatically.
        self.register_buffer("bank", torch.zeros(num_entities, d_model))
        
    def reset_bank(self):
        self.bank.zero_()
        
    def forward(self, X_c, et, entity_ids):
        """
        X_c: (batch, chunk_size, n_cont_features)
        et: (batch, chunk_size)
        entity_ids: (batch, chunk_size, 3)
        """
        assert X_c.size(0) == 1, "Prototype only supports batch_size=1"
        
        chunk_size = X_c.size(1)
        
        X_c = X_c.squeeze(0)
        et = et.squeeze(0)
        entity_ids = entity_ids.squeeze(0)
        
        feat_emb = self.cont_proj(X_c) + self.etype_emb(et)
        
        active_states = {}
        logits = []
        
        zero_state = torch.zeros(self.d_model, device=X_c.device)
        
        for i in range(chunk_size):
            subj, obj, obj2 = entity_ids[i].tolist()
            
            # 1. Gather
            s_subj = active_states.get(subj, self.bank[subj]) if subj >= 0 else zero_state
            s_obj  = active_states.get(obj, self.bank[obj])   if obj >= 0  else zero_state
            s_obj2 = active_states.get(obj2, self.bank[obj2]) if obj2 >= 0 else zero_state
                
            # 2. Aggregate
            valid_entities = (subj >= 0) + (obj >= 0) + (obj2 >= 0)
            if valid_entities > 0:
                agg_state = (s_subj + s_obj + s_obj2) / valid_entities
            else:
                agg_state = zero_state
                
            x_e = agg_state + feat_emb[i]
            
            # 3. Classify
            logits.append(self.classifier(x_e))
            
            # 4. Update
            dx = self.update_proj(x_e)
            
            if subj >= 0: active_states[subj] = s_subj + dx
            if obj >= 0:  active_states[obj]  = s_obj + dx
            if obj2 >= 0: active_states[obj2] = s_obj2 + dx
                
        # 5. Write back to global bank (detach to cut BPTT graph)
        for idx, state in active_states.items():
            self.bank[idx] = state.detach()
            
        return torch.stack(logits).unsqueeze(0).squeeze(-1)  # (1, chunk_size)
