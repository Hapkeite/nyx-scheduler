import torch

print(">>> [PYTHON] Starting PyTorch script...")

x = torch.randn(2000, 2000, device='cuda')
y = torch.randn(2000, 2000, device='cuda')
z = x @ y

print(">>> [PYTHON] Matrix math done. Deleting tensors...")

# Delete the variables so PyTorch knows they aren't needed
del x
del y
del z

# Force PyTorch to actually return the memory to the GPU (Triggers cudaFree!)
torch.cuda.empty_cache()

print(">>> [PYTHON] Cache emptied. Exiting!")
