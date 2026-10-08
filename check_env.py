"""Print the active PyTorch and CUDA environment."""

import torch


def main() -> None:
    print("--- PyTorch Environment Diagnostic ---")
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        device_index = torch.cuda.current_device()
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device ID: {device_index}")
        print(f"Current CUDA device name: {torch.cuda.get_device_name(device_index)}")
        print(f"PyTorch CUDA build: {torch.version.cuda}")
    print("--- End of Diagnostic ---")


if __name__ == "__main__":
    main()
