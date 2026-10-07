import torch

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')
else:
    print('CUDA version: N/A')
    print('GPU: N/A')
    print('Total VRAM: 0 GB')
