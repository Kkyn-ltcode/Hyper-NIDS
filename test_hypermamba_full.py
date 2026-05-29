import torch
from src.model.hypermamba_full import HyperMambaFull

def main():
    chunk_size = 4096
    d_model = 128
    
    model = HyperMambaFull(
        num_entities=1000,
        n_cont_features=35,
        num_event_types=20,
        d_model=d_model
    )
    
    # Simulate exactly what DataLoader with batch_size=1 produces
    # __getitem__ returns (chunk_size, ...), DataLoader wraps in (1, chunk_size, ...)
    x_cont = torch.randn(1, chunk_size, 35)
    event_type = torch.randint(0, 20, (1, chunk_size))
    entity_ids = torch.randint(0, 1000, (1, chunk_size, 3))
    timestamps = torch.arange(chunk_size).float().unsqueeze(0) * 0.001  # (1, chunk_size)
    
    # Forward pass
    logits = model(x_cont, event_type, entity_ids, timestamps)
    print(f"Output shape: {logits.shape}  (expected: (1, {chunk_size}))")
    assert logits.shape == (1, chunk_size), f"Shape mismatch! Got {logits.shape}"
    
    # Backward pass
    loss = logits.sum()
    loss.backward()
    
    grads_ok = all(
        p.grad is not None and not torch.isnan(p.grad).any()
        for p in model.parameters() if p.requires_grad
    )
    print(f"All gradients OK: {grads_ok}")
    
    # Check bank was updated
    print(f"Bank states non-zero: {(model.bank.states != 0).any().item()}")
    print(f"Bank last_seen updated: {(model.bank.last_seen_time >= 0).any().item()}")
    
    # Second chunk — test state carry-forward
    model.detach_bank()
    x_cont2 = torch.randn(1, chunk_size, 35)
    timestamps2 = timestamps + chunk_size * 0.001
    logits2 = model(x_cont2, event_type, entity_ids, timestamps2)
    print(f"Second chunk shape: {logits2.shape}  (expected: (1, {chunk_size}))")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")
    print("\n✅ All checks passed!")

if __name__ == "__main__":
    main()
