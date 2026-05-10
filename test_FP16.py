import torch
import pandas as pd
from model import SimpleFNN


def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    input_dim = 1024
    hidden_dims = [16384, 16384, 16384, 16384, 16384, 16384]
    output_dim = 1
    
    model = SimpleFNN(input_dim, hidden_dims, output_dim).half().to(device)
    model_path = "/mnt/d/python/GEMM/model/model_weights.pth"
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Loaded model weights from {model_path}")
    model.eval()
    
    test_input = torch.ones(1, input_dim).to(device).half()
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    import time
    start_time = time.perf_counter()
    
    with torch.no_grad():
        predictions = model(test_input)
    
    end_time = time.perf_counter()
    inference_time = (end_time - start_time) * 1000
    
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\n=== Inference Results ===")
        print(f"Inference time: {inference_time:.2f} ms")
        print(f"Peak GPU memory: {peak_memory:.2f} MB")
    else:
        print(f"\n=== Inference Results ===")
        print(f"Inference time: {inference_time:.2f} ms")
    
    input_df = pd.DataFrame(test_input.cpu().numpy()[:1, :10], columns=[f"input_{i}" for i in range(10)])
    output_df = pd.DataFrame(predictions.cpu().numpy(), columns=["output"])
    
    result_df = pd.concat([input_df, output_df], axis=1)
    
    csv_path = "/mnt/d/python/GEMM/test_results.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    print("\nFirst 10 inputs and outputs:")
    print(result_df)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()) / 1024**2:.2f} M")


if __name__ == "__main__":
    test_model()
