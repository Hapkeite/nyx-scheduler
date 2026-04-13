import torch
import torch.nn as nn
import time

# Setup model and dummy data
device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Transformer(nhead=8, num_encoder_layers=6).to(device)
src = torch.rand((32, 10, 512)).to(device)
tgt = torch.rand((32, 10, 512)).to(device)

print("--- Starting Transformer Benchmark ---")
# Warmup
_ = model(src, tgt)

start_time = time.time()
for i in range(20):
    output = model(src, tgt)
    if i % 5 == 0:
        print(f"Batch {i}/20 processed...")

torch.cuda.synchronize()
end_time = time.time()

print(f"Total Job Completion Time (JCT): {end_time - start_time:.2f} seconds")
