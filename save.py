import torch
from model import SimpleFNN


def save_fp16_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    input_dim = 1024
    hidden_dims = [16384, 16384, 16384, 16384, 16384, 16384]
    output_dim = 1
    
    model = SimpleFNN(input_dim, hidden_dims, output_dim)
    model = model.half().to(device)
    
    model_path = "/mnt/d/python/GEMM/model/model_weights.pth"
    torch.save(model.state_dict(), model_path)
    print(f"FP16 model saved to {model_path}")
    
    model_size = torch.load(model_path, weights_only=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1024**2:.2f} M")


if __name__ == "__main__":
    save_fp16_model()
