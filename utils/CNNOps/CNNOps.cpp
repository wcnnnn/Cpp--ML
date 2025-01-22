#include "CNNOps.h"
#include <stdexcept>
#include <iostream>
#include <cmath>

namespace CNNOps {
    // Convolution Implementation
    std::vector<std::vector<std::vector<double>>> Convolution::forward3D(
        const std::vector<std::vector<std::vector<double>>>& input,  // [channels][height][width]
        const std::vector<std::vector<std::vector<double>>>& kernels,  // [out_channels][in_channels][kernel_h*kernel_w]
        int stride,
        int padding
    ) {
        if (input.empty() || input[0].empty() || input[0][0].empty() || 
            kernels.empty() || kernels[0].empty() || kernels[0][0].empty() || 
            stride <= 0 || padding < 0) {
            return std::vector<std::vector<std::vector<double>>>();
        }

        int in_channels = static_cast<int>(input.size());
        int in_height = static_cast<int>(input[0].size());
        int in_width = static_cast<int>(input[0][0].size());
        int out_channels = static_cast<int>(kernels.size());
        int kernel_size = static_cast<int>(std::sqrt(kernels[0][0].size()));

        // 计算输出尺寸
        int out_height = Utils::calculateOutputSize(in_height, kernel_size, stride, padding);
        int out_width = Utils::calculateOutputSize(in_width, kernel_size, stride, padding);

        // 初始化输出
        std::vector<std::vector<std::vector<double>>> output(
            out_channels,
            std::vector<std::vector<double>>(
                out_height,
                std::vector<double>(out_width, 0.0)
            )
        );

        // 对输入进行padding
        std::vector<std::vector<std::vector<double>>> padded_input = Utils::zeroPadding3D(input, padding);

        // 对每个输出通道进行卷积
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    double sum = 0.0;
                    // 对每个输入通道进行卷积并累加
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int h_idx = oh * stride + kh;
                                int w_idx = ow * stride + kw;
                                sum += padded_input[ic][h_idx][w_idx] * 
                                      kernels[oc][ic][kh * kernel_size + kw];
                            }
                        }
                    }
                    output[oc][oh][ow] = sum;
                }
            }
        }
        return output;
    }

    std::vector<std::vector<std::vector<double>>> Convolution::backward3D(
        const std::vector<std::vector<std::vector<double>>>& gradient,
        const std::vector<std::vector<std::vector<double>>>& kernels,
        int stride,
        int padding
    ) {
        if (gradient.empty() || gradient[0].empty() || gradient[0][0].empty() || 
            kernels.empty() || kernels[0].empty() || kernels[0][0].empty() || 
            stride <= 0 || padding < 0) {
            return std::vector<std::vector<std::vector<double>>>();
        }

        int out_channels = static_cast<int>(gradient.size());
        int out_height = static_cast<int>(gradient[0].size());
        int out_width = static_cast<int>(gradient[0][0].size());
        int in_channels = static_cast<int>(kernels[0].size());
        int kernel_size = static_cast<int>(std::sqrt(kernels[0][0].size()));

        // 计算输入尺寸
        int in_height = (out_height - 1) * stride - 2 * padding + kernel_size;
        int in_width = (out_width - 1) * stride - 2 * padding + kernel_size;

        // 初始化输出
        std::vector<std::vector<std::vector<double>>> output(
            in_channels,
            std::vector<std::vector<double>>(
                in_height,
                std::vector<double>(in_width, 0.0)
            )
        );

        // 对梯度进行padding
        std::vector<std::vector<std::vector<double>>> padded_gradient = Utils::zeroPadding3D(gradient, padding);

        // 旋转卷积核
        std::vector<std::vector<std::vector<double>>> rotated_kernels = kernels;
        for (int oc = 0; oc < static_cast<int>(kernels.size()); ++oc) {
            for (int ic = 0; ic < static_cast<int>(kernels[0].size()); ++ic) {
                for (int k = 0; k < kernel_size * kernel_size; ++k) {
                    int i = k / kernel_size;
                    int j = k % kernel_size;
                    rotated_kernels[oc][ic][k] = kernels[oc][ic][(kernel_size - 1 - i) * kernel_size + (kernel_size - 1 - j)];
                }
            }
        }

        // 计算梯度
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int ih = 0; ih < in_height; ++ih) {
                for (int iw = 0; iw < in_width; ++iw) {
                    double sum = 0.0;
                    for (int oc = 0; oc < out_channels; ++oc) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                if (ih + kh < static_cast<int>(padded_gradient[0].size()) && 
                                    iw + kw < static_cast<int>(padded_gradient[0][0].size())) {
                                    sum += padded_gradient[oc][ih + kh][iw + kw] * 
                                          rotated_kernels[oc][ic][kh * kernel_size + kw];
                                }
                            }
                        }
                    }
                    output[ic][ih][iw] = sum;
                }
            }
        }
        return output;
    }

    // Pooling Implementation
    std::vector<std::vector<std::vector<double>>> Pooling::maxPool3D(
        const std::vector<std::vector<std::vector<double>>>& input,
        int poolSize,
        int stride
    ) {
        if (input.empty() || input[0].empty() || input[0][0].empty() || 
            poolSize <= 0 || stride <= 0) {
            return std::vector<std::vector<std::vector<double>>>();
        }

        int channels = static_cast<int>(input.size());
        int in_height = static_cast<int>(input[0].size());
        int in_width = static_cast<int>(input[0][0].size());
        int out_height = (in_height - poolSize) / stride + 1;
        int out_width = (in_width - poolSize) / stride + 1;

        std::vector<std::vector<std::vector<double>>> output(
            channels,
            std::vector<std::vector<double>>(
                out_height,
                std::vector<double>(out_width)
            )
        );

        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    double max_val = input[c][oh * stride][ow * stride];
                    for (int ph = 0; ph < poolSize; ++ph) {
                        for (int pw = 0; pw < poolSize; ++pw) {
                            double val = input[c][oh * stride + ph][ow * stride + pw];
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }
                    output[c][oh][ow] = max_val;
                }
            }
        }
        return output;
    }

    std::vector<std::vector<std::vector<double>>> Pooling::maxPoolBackward3D(
        const std::vector<std::vector<std::vector<double>>>& gradient,
        const std::vector<std::vector<std::vector<double>>>& input,
        int poolSize,
        int stride
    ) {
        if (gradient.empty() || gradient[0].empty() || gradient[0][0].empty() ||
            input.empty() || input[0].empty() || input[0][0].empty() ||
            poolSize <= 0 || stride <= 0) {
            return std::vector<std::vector<std::vector<double>>>();
        }

        int channels = static_cast<int>(input.size());
        int in_height = static_cast<int>(input[0].size());
        int in_width = static_cast<int>(input[0][0].size());

        std::vector<std::vector<std::vector<double>>> output(
            channels,
            std::vector<std::vector<double>>(
                in_height,
                std::vector<double>(in_width, 0.0)
            )
        );

        for (int c = 0; c < channels; ++c) {
            for (int gh = 0; gh < static_cast<int>(gradient[0].size()); ++gh) {
                for (int gw = 0; gw < static_cast<int>(gradient[0][0].size()); ++gw) {
                    // 找到最大值的位置
                    int max_h = gh * stride;
                    int max_w = gw * stride;
                    double max_val = input[c][max_h][max_w];

                    for (int ph = 0; ph < poolSize; ++ph) {
                        for (int pw = 0; pw < poolSize; ++pw) {
                            int curr_h = gh * stride + ph;
                            int curr_w = gw * stride + pw;
                            if (curr_h < in_height && curr_w < in_width) {
                                double curr_val = input[c][curr_h][curr_w];
                                if (curr_val > max_val) {
                                    max_val = curr_val;
                                    max_h = curr_h;
                                    max_w = curr_w;
                                }
                            }
                        }
                    }
                    output[c][max_h][max_w] += gradient[c][gh][gw];
                }
            }
        }
        return output;
    }

    std::vector<std::vector<std::vector<double>>> Pooling::avgPool3D(
        const std::vector<std::vector<std::vector<double>>>& input,
        int poolSize,
        int stride
    ) {
        if (input.empty() || input[0].empty() || input[0][0].empty() || 
            poolSize <= 0 || stride <= 0) {
            return std::vector<std::vector<std::vector<double>>>();
        }

        int channels = static_cast<int>(input.size());
        int in_height = static_cast<int>(input[0].size());
        int in_width = static_cast<int>(input[0][0].size());
        int out_height = (in_height - poolSize) / stride + 1;
        int out_width = (in_width - poolSize) / stride + 1;

        std::vector<std::vector<std::vector<double>>> output(
            channels,
            std::vector<std::vector<double>>(
                out_height,
                std::vector<double>(out_width, 0.0)
            )
        );

        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    double sum = 0.0;
                    for (int ph = 0; ph < poolSize; ++ph) {
                        for (int pw = 0; pw < poolSize; ++pw) {
                            sum += input[c][oh * stride + ph][ow * stride + pw];
                        }
                    }
                    output[c][oh][ow] = sum / (poolSize * poolSize);
                }
            }
        }
        return output;
    }

    std::vector<std::vector<std::vector<double>>> Pooling::avgPoolBackward3D(
        const std::vector<std::vector<std::vector<double>>>& gradient,
        int poolSize,
        int stride
    ) {
        if (gradient.empty() || gradient[0].empty() || gradient[0][0].empty() || 
            poolSize <= 0 || stride <= 0) {
            return std::vector<std::vector<std::vector<double>>>();
        }

        int channels = static_cast<int>(gradient.size());
        int out_height = static_cast<int>(gradient[0].size()) * stride;
        int out_width = static_cast<int>(gradient[0][0].size()) * stride;

        std::vector<std::vector<std::vector<double>>> output(
            channels,
            std::vector<std::vector<double>>(
                out_height,
                std::vector<double>(out_width, 0.0)
            )
        );

        double scale = 1.0 / (poolSize * poolSize);
        for (int c = 0; c < channels; ++c) {
            for (int gh = 0; gh < static_cast<int>(gradient[0].size()); ++gh) {
                for (int gw = 0; gw < static_cast<int>(gradient[0][0].size()); ++gw) {
                    for (int ph = 0; ph < poolSize; ++ph) {
                        for (int pw = 0; pw < poolSize; ++pw) {
                            output[c][gh * stride + ph][gw * stride + pw] += gradient[c][gh][gw] * scale;
                        }
                    }
                }
            }
        }
        return output;
    }

    std::vector<std::vector<std::vector<double>>> Utils::zeroPadding3D(
        const std::vector<std::vector<std::vector<double>>>& input,
        int padding
    ) {
        if (input.empty() || input[0].empty() || input[0][0].empty() || padding < 0) {
            return std::vector<std::vector<std::vector<double>>>();
        }

        int channels = static_cast<int>(input.size());
        int height = static_cast<int>(input[0].size());
        int width = static_cast<int>(input[0][0].size());

        std::vector<std::vector<std::vector<double>>> output(
            channels,
            std::vector<std::vector<double>>(
                height + 2 * padding,
                std::vector<double>(width + 2 * padding, 0.0)
            )
        );

        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    output[c][h + padding][w + padding] = input[c][h][w];
                }
            }
        }
        return output;
    }

    int Utils::calculateOutputSize(
        int inputSize,
        int kernelSize,
        int stride,
        int padding
    ) {
        return (inputSize + 2 * padding - kernelSize) / stride + 1;
    }
} 