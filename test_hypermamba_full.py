import torch
from src.model.hypermamba_full import HyperMambaFull

def main():
    batch_size = 16
    d_model = 128
    
    model = HyperMambaFull(
        num_entities=100,
        n_cont_features=35,
        num_event_types=20,
        d_model=d_model
    )
    
    # Dummy inputs
    x_cont = torch.randn(batch_size, 35)
    event_type = torch.randint(0, 20, (batch_size,))
    entity_ids = torch.randint(0, 100, (batch_size, 3))
    
    # Simulate chronological timestamps (in seconds)
    timestamps = torch.arange(batch_size).float() * 0.1
    
    # Forward pass
    logits = model(x_cont, event_type, entity_ids, timestamps)
    print("Logits shape:", logits.shape)
    
    # Backward pass
    loss = logits.sum()
    loss.backward()
    print("Gradient check (event_emb):", model.event_emb.weight.grad is not None)
    print("Gradient check (updater A):", model.updater.A_log.grad is not None)
    
if __name__ == "__main__":
    main()
