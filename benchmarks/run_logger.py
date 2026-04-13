import torch
import torchvision.models as models
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.vgg16().to(device)
dummy_input = torch.randn(16, 3, 224, 224).to(device)

print("--- Starting VGG (Metrics Logger) Benchmark ---")
# Warmup
_ = model(dummy_input)

start_time = time.time()
for i in range(20):
    output = model(dummy_input)
    if i % 5 == 0:
        print(f"Batch {i}/20 processed...")

torch.cuda.synchronize()
end_time = time.time()

print(f"Total Job Completion Time (JCT): {end_time - start_time:.2f} seconds")
