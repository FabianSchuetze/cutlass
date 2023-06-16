/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**


This example shows how to run convolution kernels using functions and data structures
provided by CUTLASS using tensor cores; which we run on a NVIDIA Turing GPU.

Writing a single high performance convolution kernel is hard but do-able. Whereas writing
high performance kernels at scale which works for multiple problem sizes with good abstractions is
really hard. CUTLASS solves this problem by providing simplified abstractions to compose
multiple sections of implicit gemm kernel. When used properly, the kernels can hit peak performance
of GPU easily.

CUTLASS divides a kernel into hierarchical composable sections. Which means, at each thread, warp
and thread-block level, they compute on their own tile-size with higher level of tile sizes being
composed from lower level ones. Multiple thread-tiles (tile size each thread computes) can be used
to form warp-tiles (tile size each warp computes) and multiple warp tiles can be used to compute
threadblock-tile (tile size computed by a threadblock).

In thie example, we split variable initialization into
1. Setting up data properties : describes how tensors are laid out in the memory and how the kernel
can view them (logical to physical mapping)
2. Setting up computation properties : describes how the above set tensors will be used to compute
output of convolution.

First, we setup the data types of the input tensor A, weights' tensor B and output tensor C along
with alpha, beta as the equation for convolution is C = alpha * Conv(A, B) + beta * C. In CUTLASS,
the kernels first compute Conv(A, B) and leave the rest of the computation to end of the kernel as
alpha * X + beta * C is a simple element-wise operation on X (Conv(A, B)) and C. We call this as 
epilogue of kernel. Hence, we setup data types for alpha and beta to be equal to 
ElementComputeEpilogue = float. We want to use MMA instructions on Turing and they support 4-bit
signed integer. But int4b_t is not fully supported by Nvidia software stack, so CUTLASS introduces
cutlass::int4b_t. We use the data type for elements in input tensor A and B as cutlass::int4b_t. We
convey this to CUTLASS kernel by initializing template variables ElementAccumulator (int32_t),
ElementComputeEpilogue (float), ElementInputA (cutlass::int4b_t), ElementInputB (cutlass::int4b_t),
ElementOutput (int32_t). Communicating just the data type is not enough. As the data is laid out 
linearly in memory, we have to convey the layout of tensors. We do that by initializing template
variables LayoutInputA, LayoutInputB and LayoutOutput to TensorNHWC cutlass variable. Next, we setup
rules to comptue alpha * X + beta * C which is called epilogue of the kernel. We initialize template
variable EpilogueOp, which takes the data type of output ElementOutput (int32_t), the number of
elements per vector memory access (32), data type of accumulator (int32_t) and data type of
computation of linear combination (alpha * X + beta * C).

Now that we setup the properties of data, we have to setup properties of computation.

Second, we create template variables of tile sizes for thread-block, warp and mma-op to 128x128x128,
64x64x128, 8x8x32 (MxNxK) respectively. When passed to instantiate CUTLASS Implicit GEMM kernel, it
internally deduces the amount of threads needed per thread-block, amount of shared memory, storing
data in bank-conflict free manner, and ton of other variables required to compose, initialize and
launch a high performance Implicit GEMM kernel. This is the beauty of CUTLASS, it relieves developer
from understanding and coding complicated hardware optimizations which can easily go wrong.

CUTLASS also supports multiple MMA pipelines in a threadblock. What are MMA pipelines? MMA pipelines
constitute the whole process of loading input data from global memory to shared memory, loading data
from shared memory to registers, doing matrix multiplication, store to global memory. The below flow
sequence shows a typical mma pipeline.

tensor in global memory -> registers -> tile in shared memory -> registers -> mma -> registers ->
output to global memory

The problem with single pipeline is, each stage is synchronous which means, each stage has to wait
until the previous finished executing. There are stages in the pipeline which do not have fixed
latency, for example, the loads from global memory and shared memory. Therefore, we can add one more
pipeline with a phase shift in mma kernel to hide latency from global and shared memory loads.
Finally, the pipeline in a kernel looks like

(1) tensor in global memory -> (2) registers -> (3) tile in shared memory -> (4) registers -> (5)
mma -> (6) registers -> (7) output to global memory (1) <null> -> (2) <null> -> (3) tensor in global
memory -> (4) registers -> (5) tile in shared memory -> (6) registers -> (7) mma -> (8) registers ->
(9) output to global memory

This way, you can hide the second global memory load latency by doing computation on already loaded
input data.

There are few more template variables initialized such as, which threadblock tile of output matrix
is done which threadblock launched on an SM, CUDA SM architecture of GPU you want to run on.

These are all put together to create a template variable which describes CUTLASS Implicit GEMM
kernel using cutlass::conv::device::ImplicitGemm template.

The next step is to initialize physical data, instantiate and initialize CUTLASS kernel and run it.
We use CUTLASS utilities to initialize, fill, compare tensors as they are simple and doesn't come
in the way of learning CUTLASS.

Once all the tensors are initialized and filled with data, create arguments tuple to launch CUTLASS
kernel which takes problem size (N = 1, H = 64, W = 64, C = 128), filter size (K = 64,
R = 3, S = 3, C = 128 ), padding, strides, dilation, tensors, alpha, beta and the
important one, split k-dimension factor. Along with that, we query CUTLASS if any scratch-space
memory required by the kernel we instantiated. If yes, we create it and pass it along with other
arguments created to initialize CUTLASS kernel then, the kernel is launched.

In this example, we later on launch a reference convolution kernel (from CUTLASS utilities) to
compare if the output from CUTLASS kernel is same as the reference implicit GEMM kernel.
*/

#include <iostream>
#include <fstream>
#include <sstream>

#include <torch/torch.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>

//#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
//#include "cutlass/util/reference/host/convolution.h"
//#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

// The code section below describes datatype for input, output tensors and computation between
// elements
using ElementAccumulator = float;                 // Data type of accumulator
using ElementComputeEpilogue = float;               // Data type of epilogue computation (alpha, beta)
using ElementInputA = cutlass::half_t;             // Data type of elements in input tensor
using ElementInputB = cutlass::half_t;             // Data type of elements in input tensor
using ElementOutput = cutlass::half_t;             // Data type of elements in output tensor

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
//using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
//using SmArch = cutlass::arch::Sm80;

//// This code section describes the tile size a thread block will compute
//using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 128>;  // Threadblock tile shape

////// This code section describes tile size a warp will compute
//using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;         // Warp tile shape

////// This code section describes the size of MMA op
//using InstructionShape = cutlass::gemm::GemmShape<8, 8, 32>;    // TensorCore instruction shape

// This code section describes how threadblocks are scheduled on GPU
//using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Number of pipelines you want to use
//constexpr int NumStages = 2;
//int const kChannelCount = 8;

// This code section describes the epilogue part of the kernel, we use default value
//using EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<
    //ElementOutput,                                     // Data type of output matrix.
    //8,                                                 // The number of elements per vectorized.
                                                       //// memory access. This becomes the vector width of
                                                       //// math instructions in the epilogue too.
    //ElementAccumulator,                                // Data type of accumulator
    //ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination


  /// Device-level Conv2d instance
  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementComputeEpilogue
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic
    //cutlass::conv::StrideSupport::kStrided,
    //kChannelCount,
    //kChannelCount
  >::Kernel;
  //using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    //ElementA, cutlass::layout::TensorNHWC,
    //ElementB, cutlass::layout::TensorNHWC,
    //ElementC, cutlass::layout::TensorNHWC,
    //ElementAccumulator,
    //cutlass::arch::OpClassTensorOp,
    //cutlass::arch::Sm80,
    //cutlass::gemm::GemmShape<128, 128, 64>,
    //cutlass::gemm::GemmShape<64, 64, 64>,
    //cutlass::gemm::GemmShape<16, 8, 16>,
    //cutlass::epilogue::thread::LinearCombination<
      //ElementC,
      //128 / cutlass::sizeof_bits<ElementC>::value,
      //ElementAccumulator,
      //ElementCompute
    //>,
    //cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    //3,
    //cutlass::arch::OpMultiplyAdd,
    //cutlass::conv::IteratorAlgorithm::kAnalytic
  //>::Kernel;

  //using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;


using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;


/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  cutlass::Tensor4DCoord input_size;
  cutlass::Tensor4DCoord filter_size;
  cutlass::Tensor4DCoord padding;
  cutlass::MatrixCoord conv_stride;
  cutlass::MatrixCoord dilation;
  bool reference_check;
  bool measure_performance;
  int iterations;
  bool save_workspace;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;
  bool benchmark;
  std::string tag;

  Options():
    help(false),
    input_size(1, 32, 32, 32),
    filter_size(32, 3, 3, 32),
    padding(1, 1, 1, 1),
    conv_stride(1, 1),
    dilation(1, 1),
    reference_check(false),
    measure_performance(true),
    iterations(20),
    save_workspace(false),
    alpha(1),
    beta(1),
    benchmark(false) { }

  // Verify the problem size is compatible with the CUTLASS Convolution implementation.
  bool valid() {

    //
    // CUTLASS attempts to load 128b vectors of int4b_t elements. Consequently,
    // all pointers, strides, and tensor extents must be divisible by 32 elements.
    //
    int const kAlignment = 32;

    if ((input_size.c() % kAlignment) ||
      (filter_size.n() % kAlignment)) {

      // misaligned tensors
      return false;
    }

    // Invalid padding
    if ((padding.h() != filter_size.h() / 2) ||
      (padding.w() != filter_size.w() / 2)) {

      return false;
    }

    return true;
  }

  /// Updates input and filter sizes
  void update(
    cutlass::Tensor4DCoord input_size,
    cutlass::Tensor4DCoord filter_size) {

    this->input_size = input_size;
    this->filter_size = filter_size;

    padding.n() = filter_size.h() / 2;
    padding.h() = filter_size.h() / 2;
    padding.w() = filter_size.w() / 2;
    padding.c() = filter_size.w() / 2;
  }

  /// Computes the output tensor size (NPQK)
  cutlass::Tensor4DCoord output_size() const {
    return cutlass::Tensor4DCoord(
      input_size.n(),
      (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row() + 1,
      (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.column() + 1,
      filter_size.n());
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of multiply-adds = NPQK * CRS
    int64_t fmas = output_size().product() * int64_t(filter_size.h() * filter_size.w() * filter_size.c());
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};


torch::Tensor forward_fp16(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    //CHECK_INPUT(input);
    //CHECK_INPUT(weight);


    Options options = Options();

    options.update({input.size(0), input.size(1), input.size(2), input.size(3)},
                  {weight.size(0), weight.size(1), weight.size(2), weight.size(3)});
    if (!options.valid()) {
        throw std::runtime_error("Option is not valid");
    }
    const auto out = options.output_size();


   torch::Device device = torch::kCUDA;
   torch::Tensor output = bias.repeat({out.(), out.h(), out.w(), 1}).to(device).to(torch::kFloat);
   //torch::Tensor output = torch::zeros({out.n(), out.h(), out.w(), out.c()}).to(device).to(torch::kFloat16);
   std::cout << "THe output shape is " << output.sizes() << ", with stride: " << output.strides() << std::endl;
    cutlass::TensorRef<ElementInputA, LayoutInputA> d_src(
            (ElementInputA*)input.data_ptr(),
            LayoutInputA::packed(options.input_size));
    cutlass::TensorRef<ElementInputB, LayoutInputB> d_filter(
            (ElementInputB*)weight.data_ptr(),
            LayoutInputB::packed(options.filter_size));
    cutlass::TensorRef<ElementOutput, LayoutOutput>
            d_dst((ElementOutput*)output.data_ptr(),
                  LayoutOutput::packed(options.output_size()));

    //
    // Define arguments for CUTLASS Convolution
    //

    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    typename ImplicitGemm::Arguments arguments{
            {options.input_size, options.filter_size, options.padding,
             options.conv_stride, options.dilation, options.output_size(), mode,
             split_k_slices},
            d_src,     // tensor_src.device_ref(),
            d_filter,  // tensor_filter.device_ref(),
            d_dst,
            d_dst,     // tensor_dst.device_ref(),
            {ElementComputeEpilogue(options.alpha), ElementComputeEpilogue(options.beta)}};

    //
    // Initialize CUTLASS Convolution
    //

    ImplicitGemm conv_op;

    size_t workspace_size = conv_op.get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    CUTLASS_CHECK(conv_op.can_implement(arguments));

    CUTLASS_CHECK(conv_op.initialize(arguments, workspace.get()));

    //
    // Launch initialized CUTLASS kernel
    //

    CUTLASS_CHECK(conv_op());
    //std::cout << output  << std::endl;

    return output;
}

int main() {
    torch::Device device = torch::kCUDA;
    torch::Tensor a = torch::rand({1, 32, 224, 224});
    torch::save(a, "/tmp/inp.pt");
    a = a.to(device).to(torch::kFloat16);
    torch::nn::ConvOptions<2> options(int64_t(32), int64_t(256), {3,3});
    options.bias(true);
    options.padding(1);
    torch::nn::Conv2d conv{options};
    torch::save(conv, "/tmp/conv.pt");
    conv->to(torch::kCUDA);
    conv->to(torch::kFloat16);
    torch::Tensor y = conv(a);
    torch::save(y, "/tmp/orig.pt");
    a = a.permute({0, 2, 3, 1}).contiguous();
    std::cout << "a sizes and strides: " << a.sizes() << "; " << a.strides() << std::endl;
    torch::Tensor b = conv->parameters()[0];
    b = b.permute({0, 2, 3, 1}).contiguous();
    torch::Tensor bias = conv->parameters()[1];
    std::cout << "b sizes and strides: " << b.sizes() << "; " << b.strides() << std::endl;
    std::cout << "bias sizes and strides: " << bias.sizes() << "; " << bias.strides() << std::endl;
    torch::Tensor y_cutlass = forward_fp16(a, b, bias);
    std::cout << "output sizes and strides: " << y_cutlass.sizes() << "; " << y_cutlass.strides() << std::endl;
    y_cutlass = y_cutlass.permute({0, 3, 1, 2}).contiguous();
    std::cout << "output sizes and strides: " << y_cutlass.sizes() << "; " << y_cutlass.strides() << std::endl;

    torch::Tensor diff = torch::mean(torch::pow( y - y_cutlass, 2));
    torch::Tensor ratio = diff / torch::mean(torch::pow(y, 2));
    std::cout << "The differnce is: " << ratio << std::endl;
    torch::save(y_cutlass, "/tmp/cutlass.pt");
}

