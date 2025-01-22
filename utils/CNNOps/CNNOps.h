#pragma once
#include <vector>

namespace CNNOps {
    // 基础卷积操作
    class Convolution {
    public:
        static std::vector<std::vector<std::vector<double>>> forward3D(
            const std::vector<std::vector<std::vector<double>>>& input,  // [channels][height][width]
            const std::vector<std::vector<std::vector<double>>>& kernels,  // [out_channels][in_channels][kernel_h*kernel_w]
            int stride,
            int padding
        );

        static std::vector<std::vector<std::vector<double>>> backward3D(
            const std::vector<std::vector<std::vector<double>>>& gradient,
            const std::vector<std::vector<std::vector<double>>>& kernels,
            int stride,
            int padding
        );
    };

    // 池化层操作
    class Pooling {
    public:
        static std::vector<std::vector<std::vector<double>>> maxPool3D(
            const std::vector<std::vector<std::vector<double>>>& input,
            int poolSize,
            int stride
        );

        static std::vector<std::vector<std::vector<double>>> maxPoolBackward3D(
            const std::vector<std::vector<std::vector<double>>>& gradient,
            const std::vector<std::vector<std::vector<double>>>& input,
            int poolSize,
            int stride
        );

        static std::vector<std::vector<std::vector<double>>> avgPool3D(
            const std::vector<std::vector<std::vector<double>>>& input,
            int poolSize,
            int stride
        );

        static std::vector<std::vector<std::vector<double>>> avgPoolBackward3D(
            const std::vector<std::vector<std::vector<double>>>& gradient,
            int poolSize,
            int stride
        );
    };

    // 工具函数
    class Utils {
    public:
        static std::vector<std::vector<std::vector<double>>> zeroPadding3D(
            const std::vector<std::vector<std::vector<double>>>& input,
            int padding
        );

        static int calculateOutputSize(
            int inputSize,
            int kernelSize,
            int stride,
            int padding
        );
    };
} 