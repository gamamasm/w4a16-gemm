#include <torch/extension.h>        // PyTorch C++扩展头文件，提供torch::Tensor等类型
#include <cuda.h>                    // CUDA运行时API头文件
#include <cuda_fp16.h>              // FP16半精度浮点数支持
#include <iostream>                  // C++标准输入输出流


// W4A16 GEMM CUDA核函数
// weight_int4: 量化后的INT4权重（每字节存储2个INT4）
// scale: 量化缩放因子
// input: FP16输入 activation
// bias: 偏置（可选）
// batch_size: batch大小
// in_features: 输入维度
// out_features: 输出维度
// num_groups: 量化组数量
// group_size: 每组大小
__global__ void w4a16_gemm_kernel(
    const uint8_t* __restrict__ weight_int4,   // 量化权重（INT4打包成UINT8）
    const at::Half* __restrict__ scale,        // 量化缩放因子（FP16）
    const at::Half* __restrict__ input,        // 输入activation（FP16）
    const at::Half* __restrict__ bias,          // 偏置（FP16，可选）
    at::Half* __restrict__ output,             // 输出结果（FP16）
    int batch_size,                             // batch大小
    int in_features,                            // 输入维度
    int out_features,                           // 输出维度
    int num_groups,                             // 量化组数量
    int group_size                              // 每组大小
) {
    // 计算当前线程处理的输出位置
    const int row = blockIdx.y * blockDim.y + threadIdx.y;  // batch维度
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // 输出维度
    
    // 越界检查
    if (row >= batch_size || col >= out_features) return;
    
    // 累加器，使用float提高精度
    float sum = 0.0f;
    
    // 如果有偏置，先加上偏置
    if (bias != nullptr) {
        sum = __half2float(bias[col]);  // 将FP16偏置转换为float
    }
    
    // 计算填充后的输入维度（向上取整到偶数）
    const int padded_in_features = (in_features + 1) / 2 * 2;
    
    // 遍历每个量化组
    for (int g = 0; g < num_groups; ++g) {
        // 计算当前组的scale偏移（per-channel + per-group）
        const int scale_offset = col * num_groups + g;
        at::Half scale_val = scale[scale_offset];  // 获取当前组的scale
        
        // 遍历组内的每个权重元素
        for (int i = 0; i < group_size; ++i) {
            // 计算当前列在总权重中的索引
            const int col_idx = g * group_size + i;
            if (col_idx >= in_features) break;  // 越界检查
            
            // 计算打包后权重中的字节索引（每2个INT4打包成1个INT8）
            const int packed_idx = col_idx / 2;
            // 计算位偏移（低4位或高4位）
            const int bit_shift = (col_idx % 2) * 4;
            
            // 计算权重在内存中的偏移量
            const int weight_offset = col * ((padded_in_features + 1) / 2) + packed_idx;
            // 计算输入在内存中的偏移量
            const int input_offset = row * in_features + col_idx;
            
            // 从内存读取打包的权重（1个字节=2个INT4）
            uint8_t packed = weight_int4[weight_offset];
            // 解包：右移相应位数并取低4位，得到INT4值
            uint8_t w_int4 = (packed >> bit_shift) & 0x0F;
            
            // 将INT4转换为FP32：先减去零点偏移8.0，再乘以scale
            float w_fp = (static_cast<float>(w_int4) - 8.0f) * __half2float(scale_val);
            // 将输入从FP16转换为FP32
            float x_fp = __half2float(input[input_offset]);
            
            // 累加矩阵乘法结果
            sum += w_fp * x_fp;
        }
    }
    
    // 将结果写回内存（FP32转FP16）
    output[row * out_features + col] = __float2half(sum);
}

// PyTorch调用接口
torch::Tensor w4a16_gemm(
    torch::Tensor weight_int4,   // 量化权重
    torch::Tensor scale,          // 缩放因子
    torch::Tensor input,          // 输入
    torch::Tensor bias,           // 偏置
    int in_features               // 输入维度
) {
    // 获取维度信息
    const int batch_size = input.size(0);           // batch大小
    const int out_features = weight_int4.size(0);   // 输出维度
    const int num_groups = scale.size(1);           // 量化组数量
    const int group_size = in_features / num_groups; // 每组大小
    
    // 创建输出tensor
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    // 设置CUDA线程块和网格大小
    const dim3 block(16, 16);  // 每个block 16x16 = 256个线程
    const dim3 grid(
        (out_features + block.x - 1) / block.x,  // 输出维度网格数
        (batch_size + block.y - 1) / block.y     // batch维度网格数
    );
    
    // 启动CUDA核函数
    w4a16_gemm_kernel<<<grid, block>>>(
        weight_int4.data_ptr<uint8_t>(),     // 权重指针
        scale.data_ptr<at::Half>(),          // scale指针
        input.data_ptr<at::Half>(),          // 输入指针
        bias.defined() ? bias.data_ptr<at::Half>() : nullptr,  // 偏置指针（检查是否定义）
        output.data_ptr<at::Half>(),         // 输出指针
        batch_size,                          // batch大小
        in_features,                         // 输入维度
        out_features,                        // 输出维度
        num_groups,                          // 量化组数量
        group_size                           // 每组大小
    );
    
    // 等待CUDA操作完成
    cudaDeviceSynchronize();
    
    // 返回输出tensor
    return output;
}

// PyBind11模块绑定（使Python可以调用）
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w4a16_gemm", &w4a16_gemm, "W4A16 GEMM kernel");
}
