import torch
import torch.nn as nn

class SimpleFNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

if __name__ == '__main__':
    model = SimpleFNN(4096, [16384, 16384, 16384, 16384, 16384, 16384, 16384], 4096)
    print(model)
    x = torch.randn(32, 4096)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1024**2:.2f} M")
