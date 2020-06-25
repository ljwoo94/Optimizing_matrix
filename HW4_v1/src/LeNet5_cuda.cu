#include "LeNet5_cuda.h"

    /* NOTE: unless you want to make a major change to this class structure, 
    *  you need to write your output to the device memory d_output 
    *  so that classify() can handle the rest.
    */

__global__
void normalize(const uint8_t* const image, double* input) {
  //Initialize variables
  float max_int = 255.0L;
  float mean = 0.5L;
  float var = 0.5L;

  //i = 0 to batch*input_channel*input_size*input_size
  //each block have DimBlock size, and there are DimGrid blocks
  //ref: Cuda Thread indexing cheetsheet
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int i = blockId *(blockDim.x * blockDim.y) 
          + (threadIdx.y * blockDim.x) + threadIdx.x;
  
  input[i] = image[i] / max_int;
  input[i] = (input[i] - mean) / var;
}

__global__
void conv(double* input, double* output, double* weight,
          double* bias, int B, int H, int W, int IC, int OC,
          int K) {
  //Initialize variable
  //oc = blockIdx.x, b = blockIdx.y
  //w = threadIdx.x, h = threadIdx.y
  
  int output_index = blockIdx.y * (gridDim.x * blockDim.y * blockDim.x)
                    + blockIdx.x * (blockDim.y * blockDim.x)
                    + (int)fmaf(threadIdx.y,blockDim.x,threadIdx.x);
  double tmp = bias[blockIdx.x];

  #pragma unroll
  for(int ic = 0; ic < IC; ic++) {
    int input_base = blockIdx.y * (IC * H * W) + ic * (H * W) 
                  + (int)fmaf(threadIdx.y,W, threadIdx.x);
    int kernel_base = blockIdx.x * (IC * K * K) + ic * (K * K);

    //let out exit from loop due to threading.
    #pragma unroll
    for (int kh = 0; kh < K; kh++)
      for (int kw = 0; kw < K; kw++) {
        tmp += input[(int)fmaf(kh,W,input_base) + kw] *
              weight[(int)fmaf(kh,K,kernel_base) + kw];
    }
  }
  output[output_index] = tmp;
}

__global__
void relu(double* feature_map, int size) {
  //relu
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int i = blockId * (blockDim.x * blockDim.y)
            + (threadIdx.y * blockDim.x) + threadIdx.x;
  feature_map[i] = feature_map[i]>0.0 ? feature_map[i]:0.0;
  //feature_map[i] = fmaxf(feature_map[i], 0.0);
}

//Max Pooling
__global__
void pool(double* input, double* output, int B, int C, int H, int W) {

  int scale = 2;
  int H_OUT = H / scale;
  int W_OUT = W / scale;
  
  int input_base = blockIdx.y * (C * H * W)
                  + blockIdx.x * (H * W)
                  + (threadIdx.y*2) * (W)
                  + (threadIdx.x*2);
  //Init values
  double max_val = 0.0;

  //Find Maximum
  for(int sh = 0; sh < scale; sh++)
    for(int sw = 0; sw < scale; sw++) {
      double val = input[input_base + sh*(W) + sw];
      if(val > max_val) {
        max_val = val;
      }
    }

    int output_index = blockIdx.y * (C * H_OUT * W_OUT)
                    + blockIdx.x * (H_OUT * W_OUT)
                    + (threadIdx.y) * (W_OUT)
                    + (threadIdx.x);

    output[output_index] = max_val;
}

__global__
void fc(double* input, double* output, double* weight, double* bias,
        int B, int IC, int OC) {
  
  //Fully Connected
  int idx = (int)fmaf(blockIdx.x,blockDim.x,threadIdx.x);
  double tmp = bias[threadIdx.x];

  // __shared__ double sw[IC][IC];
  // sw[threadIdx.y][threadIdx.x] = weight[threadIdx.x*IC];

  // for(int ic = 0; ic < IC; ic++){
  //   tmp += sw[threadIdx.y][ic] * input[(int)fmaf(blockIdx.x,IC,ic)];
  // }
  #pragma unroll
  for(int ic = 0; ic < IC; ic++){
    tmp += weight[(int)fmaf(threadIdx.x, IC, ic)] * input[(int)fmaf(blockIdx.x,IC,ic)];
  }
  output[idx] = tmp;
}

void LeNet5_cuda::predict(int batch) {

    // TODO: conv1 - relu - pool1
    // TODO: conv2 - relu - pool2
    // TODO: fc1 - relu - fc2 - relu - fc3

    //Initialize grid and block dimensions
    dim3 DimGrid(input_channel, batch);
    dim3 DimBlock(input_size, input_size);

    //ToTensor and Normalize
    normalize<<<DimGrid, DimBlock>>> (d_image, d_input);
    cudaDeviceSynchronize();

    // Conv2d
    DimGrid.x = conv1_out_channel; DimGrid.y = batch;
    DimBlock.x = input_size-(conv1_kernel_size-1); 
    DimBlock.y = input_size-(conv1_kernel_size-1);
    conv<<<DimGrid , DimBlock>>>(d_input, d_C1_feature_map, d_conv1_weight, 
           d_conv1_bias, batch, input_size, input_size, conv1_in_channel,
           conv1_out_channel, conv1_kernel_size);
    cudaDeviceSynchronize();

    DimGrid.y = batch; DimGrid.x = C1_channel;
    DimBlock.y = C1_size; DimBlock.x = C1_size;
    relu<<<DimGrid, DimBlock>>>(d_C1_feature_map, batch * C1_channel * C1_size * C1_size);
    cudaDeviceSynchronize();

    // MaxPool2d
    DimGrid.y = batch; DimGrid.x = C1_channel;
    DimBlock.y = C1_size/2; DimBlock.x = C1_size/2;
    pool<<<DimGrid, DimBlock>>>(d_C1_feature_map, d_S2_feature_map, batch, C1_channel, C1_size, C1_size);
    cudaDeviceSynchronize();

    // Conv2d
    DimGrid.y = batch; DimGrid.x = conv2_out_channel;
    DimBlock.y = S2_size - (conv2_kernel_size - 1);
    DimBlock.x = S2_size - (conv2_kernel_size - 1);
    conv<<<DimGrid, DimBlock>>>(d_S2_feature_map, d_C3_feature_map, d_conv2_weight, d_conv2_bias, batch,
          S2_size, S2_size, conv2_in_channel, conv2_out_channel, conv2_kernel_size);
    cudaDeviceSynchronize();

    DimGrid.y = batch; DimGrid.x = C3_channel;
    DimBlock.y = C3_size; DimBlock.x = C3_size;
    relu<<<DimGrid, DimBlock>>>(d_C3_feature_map, batch * C3_channel * C3_size * C3_size);
    cudaDeviceSynchronize();

    // MaxPool2d
    DimGrid.y = batch; DimGrid.x = C3_channel;
    DimBlock.y = C3_size/2; DimBlock.x = C3_size/2;
    pool<<<DimGrid, DimBlock>>>(d_C3_feature_map, d_S4_feature_map, batch, 
          C3_channel, C3_size, C3_size);
    cudaDeviceSynchronize();

    // Linear
    //DimGrid.y = 1; DimGrid.x = batch;
    //DimBlock.y = 1; DimBlock.x = fc1_out_channel;
    fc<<<batch, fc1_out_channel>>>(d_S4_feature_map, d_C5_layer, d_fc1_weight, d_fc1_bias,
                              batch, fc1_in_channel, fc1_out_channel);
    cudaDeviceSynchronize();

    DimGrid.y = 1; DimGrid.x = batch;
    DimBlock.y = 1; DimBlock.x = C5_size;
    relu<<<DimGrid, DimBlock>>>(d_C5_layer, batch*C5_size);
    cudaDeviceSynchronize();

    // Linear
    //DimGrid.y = batch; DimGrid.x = 1;
    //DimBlock.y = fc2_out_channel; DimBlock.x = 1;
    fc<<<batch, fc2_out_channel>>>(d_C5_layer, d_F6_layer, d_fc2_weight, d_fc2_bias,
                              batch, fc2_in_channel, fc2_out_channel);
    cudaDeviceSynchronize();

    DimGrid.y = 1; DimGrid.x = batch;
    DimBlock.y = 1; DimBlock.x = F6_size;
    relu<<<DimGrid, DimBlock>>>(d_F6_layer, batch*F6_size);
    cudaDeviceSynchronize();

    // Linear
    //DimGrid.y = batch; DimGrid.x = 1;
    //DimBlock.y = output_size; DimBlock.x = 1;
    fc<<<batch, output_size>>>(d_F6_layer, d_output, d_fc3_weight, d_fc3_bias,
                              batch, fc3_in_channel, fc3_out_channel);
    cudaDeviceSynchronize();
}

void LeNet5_cuda::prepare_device_memory(uint8_t* image) {
  // Alloc Model Parameters
  cudaMalloc((void**)&d_conv1_weight,
             sizeof(double) * conv1_in_channel * conv1_out_channel *
                 conv1_kernel_size * conv1_kernel_size);
  cudaMalloc((void**)&d_conv1_bias, sizeof(double) * conv1_out_channel);
  cudaMalloc((void**)&d_conv2_weight,
             sizeof(double) * conv2_in_channel * conv2_out_channel *
                 conv2_kernel_size * conv2_kernel_size);
  cudaMalloc((void**)&d_conv2_bias, sizeof(double) * conv2_out_channel);
  cudaMalloc((void**)&d_fc1_weight,
             sizeof(double) * fc1_in_channel * fc1_out_channel);
  cudaMalloc((void**)&d_fc1_bias, sizeof(double) * fc1_out_channel);
  cudaMalloc((void**)&d_fc2_weight,
             sizeof(double) * fc2_in_channel * fc2_out_channel);
  cudaMalloc((void**)&d_fc2_bias, sizeof(double) * fc2_out_channel);
  cudaMalloc((void**)&d_fc3_weight,
             sizeof(double) * fc3_in_channel * fc3_out_channel);
  cudaMalloc((void**)&d_fc3_bias, sizeof(double) * fc3_out_channel);

  // Alloc Activations
  cudaMalloc((void**)&d_image,
             sizeof(uint8_t) * batch * input_size * input_size * input_channel);
  cudaMalloc((void**)&d_input,
             sizeof(double) * batch * input_channel * input_size * input_size);
  cudaMalloc((void**)&d_C1_feature_map,
             sizeof(double) * batch * C1_channel * C1_size * C1_size);
  cudaMalloc((void**)&d_S2_feature_map,
             sizeof(double) * batch * S2_channel * S2_size * S2_size);
  cudaMalloc((void**)&d_C3_feature_map,
             sizeof(double) * batch * C3_channel * C3_size * C3_size);
  cudaMalloc((void**)&d_S4_feature_map,
             sizeof(double) * batch * S4_channel * S4_size * S4_size);
  cudaMalloc((void**)&d_C5_layer, sizeof(double) * batch * C5_size);
  cudaMalloc((void**)&d_F6_layer, sizeof(double) * batch * F6_size);
  cudaMalloc((void**)&d_output, sizeof(double) * batch * output_size);

  // Copy Parameters

  cudaMemcpy(d_conv1_weight, conv1_weight,
             sizeof(double) * conv1_in_channel * conv1_out_channel *
                 conv1_kernel_size * conv1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv1_bias, conv1_bias, sizeof(double) * conv1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv2_weight, conv2_weight,
             sizeof(double) * conv2_in_channel * conv2_out_channel *
                 conv2_kernel_size * conv2_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv2_bias, conv2_bias, sizeof(double) * conv2_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc1_weight, fc1_weight,
             sizeof(double) * fc1_in_channel * fc1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc1_bias, fc1_bias, sizeof(double) * fc1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc2_weight, fc2_weight,
             sizeof(double) * fc2_in_channel * fc2_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc2_bias, fc2_bias, sizeof(double) * fc2_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc3_weight, fc3_weight,
             sizeof(double) * fc3_in_channel * fc3_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc3_bias, fc3_bias, sizeof(double) * fc3_out_channel,
             cudaMemcpyHostToDevice);

  // copy input image
  size_t image_size = batch * input_size * input_size * input_channel;
  cudaMemcpy(d_image, image, image_size * sizeof(uint8_t),
             cudaMemcpyHostToDevice);

}

void LeNet5_cuda::classify(int* predict, int batch) {
  // read logits back to cpu
  cudaMemcpy(output, d_output, sizeof(double) * output_size * batch,
             cudaMemcpyDeviceToHost);
  // Softmax
  softmax(output, predict, batch, output_size);
}

LeNet5_cuda::~LeNet5_cuda() {
  cudaFree(d_conv1_weight);   
  cudaFree(d_conv2_weight);   
  cudaFree(d_conv1_bias);     
  cudaFree(d_conv2_bias);     
  cudaFree(d_fc1_weight);     
  cudaFree(d_fc2_weight);     
  cudaFree(d_fc3_weight);     
  cudaFree(d_fc1_bias);       
  cudaFree(d_fc2_bias);       
  cudaFree(d_fc3_bias);       

  cudaFree(d_image);          
  cudaFree(d_input);          
  cudaFree(d_C1_feature_map); 
  cudaFree(d_S2_feature_map); 
  cudaFree(d_C3_feature_map); 
  cudaFree(d_S4_feature_map); 
  cudaFree(d_C5_layer);      
  cudaFree(d_F6_layer);     
  cudaFree(d_output);       
  cudaFree(d_predict_cuda);   
}
