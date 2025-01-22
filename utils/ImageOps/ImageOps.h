#pragma once
#include <vector>
#include <string>

namespace ImageOps {
    // 图像数据结构
    struct Image {
        std::vector<std::vector<std::vector<double>>> data; // channels x height x width
        int channels;
        int height;
        int width;
    };

    class ImageLoader {
    public:
        static Image loadImage(const std::string& filepath);
        static void saveImage(const std::string& filepath, const Image& image);
    };

    class ImageProcessor {
    public:
        // 图像预处理
        static Image normalize(const Image& image);
        static Image resize(const Image& image, int newHeight, int newWidth);
        
        // 数据增强
        static Image rotate(const Image& image, double angle);
        static Image flip(const Image& image, bool horizontal);
        static Image adjustBrightness(const Image& image, double factor);
        static Image adjustContrast(const Image& image, double factor);
    };

    class BatchProcessor {
    public:
        // 批处理操作
        static std::vector<Image> createBatch(
            const std::vector<std::string>& filepaths,
            int batchSize
        );
        
        static std::vector<Image> augmentBatch(
            const std::vector<Image>& batch
        );
    };
} 