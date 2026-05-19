
import torch
print(torch.cuda.current_device()) # Returns the active index (e.g., 0)
print(torch.cuda.get_device_name(0)) # Verifies the name of GPU at index 0

