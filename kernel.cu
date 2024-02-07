#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <chrono>
#define BLOCK_SIZE 16

using namespace std;
using namespace cv;

typedef struct
{
    int width;
    int height;
    float* dataArray;
    int dataArraySize;
} Data;

__constant__ float sobelMaskX[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
__constant__ float sobelMaskY[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
__constant__ float piramidalnyMask[25] = { 1, 2, 3, 2, 1,
                                           2, 4, 6, 4, 2,
                                           3, 6, 9, 6, 3,
                                           2, 4, 6, 4, 2,
                                           1, 2, 3, 2, 1 };

__global__ void SobelFilterCUDA(Data dim_gpu, float* values_gpu) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < dim_gpu.width - 1 && j > 0 && j < dim_gpu.height - 1) {
        const int offset = j * dim_gpu.width + i;

        float gx = 0;
        float gy = 0;

        for (int k = -1; k <= 1; ++k) {
            for (int l = -1; l <= 1; ++l) {
                int idx = (j + k) * dim_gpu.width + (i + l);
                float neighborValue = dim_gpu.dataArray[idx];
                gx += neighborValue * sobelMaskX[(k + 1) * 3 + (l + 1)];
                gy += neighborValue * sobelMaskY[(l + 1) * 3 + (k + 1)];
            }
        }

        values_gpu[offset] = sqrtf(gx * gx + gy * gy);
    }
}

void SobelFilter(const Data& dim, float* values) {
    for (int j = 1; j < dim.height - 1; ++j) {
        for (int i = 1; i < dim.width - 1; ++i) {
            const int offset = j * dim.width + i;

            float gx = 0;
            float gy = 0;

            for (int k = -1; k <= 1; ++k) {
                for (int l = -1; l <= 1; ++l) {
                    int idx = (j + k) * dim.width + (i + l);
                    float neighborValue = dim.dataArray[idx];
                    gx += neighborValue * sobelMaskX[(k + 1) * 3 + (l + 1)];
                    gy += neighborValue * sobelMaskY[(l + 1) * 3 + (k + 1)];
                }
            }

            values[offset] = sqrtf(gx * gx + gy * gy);
        }
    }
}

__global__ void LaplaceFilterCUDA(Data dim_gpu, float* values_gpu) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < dim_gpu.width - 1 && j > 0 && j < dim_gpu.height - 1) {
        const int offset = j * dim_gpu.width + i;

        float laplaceValue = 0.0f;

        for (int k = -1; k <= 1; ++k) {
            for (int l = -1; l <= 1; ++l) {
                int idx = ((j + k) * dim_gpu.width) + (i + l);
                laplaceValue += dim_gpu.dataArray[idx];
            }
        }

        laplaceValue -= 8 * dim_gpu.dataArray[offset];

        const float threshold = 255.0f;

        values_gpu[offset] = (fabs(laplaceValue) > threshold) ? 255.0f : 0.0f;
    }
}

void LaplaceFilter(const Data& dim, float* values) {
    for (int j = 1; j < dim.height - 1; ++j) {
        for (int i = 1; i < dim.width - 1; ++i) {
            const int offset = j * dim.width + i;

            float laplaceValue = 0.0f;

            for (int k = -1; k <= 1; ++k) {
                for (int l = -1; l <= 1; ++l) {
                    int idx = ((j + k) * dim.width) + (i + l);
                    laplaceValue += dim.dataArray[idx];
                }
            }

            laplaceValue -= 8 * dim.dataArray[offset];

            const float threshold = 255.0f;

            values[offset] = (fabs(laplaceValue) > threshold) ? 255.0f : 0.0f;
        }
    }
}

void drawBarChart(Mat& image, const vector<float>& data, int width, int height, int cpu, int gpu) {
    int numBars = data.size();
    int barWidth = width / numBars;
    int maxValue = *max_element(data.begin(), data.end());

    for (int i = 0; i < numBars; ++i) {
        Scalar color;
        if (data[i] == cpu)
            color = Scalar(0, 0, 255);
        else if (data[i] == gpu)
            color = Scalar(255, 0, 0);
        else
            color = Scalar(0, 0, 0);

        int barHeight = static_cast<int>(static_cast<float>(data[i]) / maxValue * height);
        Point pt1(i * barWidth, height - barHeight);
        Point pt2((i + 1) * barWidth, height);
        rectangle(image, pt1, pt2, color, -1);
    }
}

cv::Mat convertToGrayscale(const cv::Mat& inputImage) {
    cv::Mat grayscaleImage;
    cv::cvtColor(inputImage, grayscaleImage, cv::COLOR_BGR2GRAY);
    return grayscaleImage;
}

__global__ void KuwaharaFilterColorCUDA(Data dim_gpu, float* values_gpu) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 1 && i < dim_gpu.width - 2 && j > 1 && j < dim_gpu.height - 2) {
        const int offset = j * dim_gpu.width + i;

        float neighborhoods[4][4][3];

        for (int m = 0; m < 4; ++m) {
            for (int n = 0; n < 4; ++n) {
                for (int c = 0; c < 3; ++c) {
                    int idx = ((j + m - 1) * dim_gpu.width + (i + n - 1)) * 3 + c;
                    neighborhoods[m][n][c] = dim_gpu.dataArray[idx];
                }
            }
        }

        float averages[4][3];
        for (int m = 0; m < 4; ++m) {
            for (int c = 0; c < 3; ++c) {
                averages[m][c] = 0.0f;
                for (int n = 0; n < 4; ++n) {
                    averages[m][c] += neighborhoods[m][n][c];
                }
                averages[m][c] /= 4.0f;
            }
        }

        float variances[4][3];
        for (int m = 0; m < 4; ++m) {
            for (int c = 0; c < 3; ++c) {
                variances[m][c] = 0.0f;
                for (int n = 0; n < 4; ++n) {
                    variances[m][c] += (neighborhoods[m][n][c] - averages[m][c]) * (neighborhoods[m][n][c] - averages[m][c]);
                }
                variances[m][c] /= 4.0f;
            }
        }

        int minVarianceIdx = 0;
        for (int m = 1; m < 4; ++m) {
            for (int c = 0; c < 3; ++c) {
                if (variances[m][c] < variances[minVarianceIdx][c]) {
                    minVarianceIdx = m;
                }
            }
        }

        for (int c = 0; c < 3; ++c) {
            values_gpu[offset * 3 + c] = averages[minVarianceIdx][c];
        }
    }
}

void KuwaharaFilterColorCPU(Data dim, float* values) {
    for (int i = 2; i < dim.width - 2; ++i) {
        for (int j = 2; j < dim.height - 2; ++j) {
            const int offset = j * dim.width + i;

            float neighborhoods[4][4][3];

            for (int m = 0; m < 4; ++m) {
                for (int n = 0; n < 4; ++n) {
                    for (int c = 0; c < 3; ++c) {
                        int idx = ((j + m - 1) * dim.width + (i + n - 1)) * 3 + c;
                        neighborhoods[m][n][c] = dim.dataArray[idx];
                    }
                }
            }

            float averages[4][3];
            for (int m = 0; m < 4; ++m) {
                for (int c = 0; c < 3; ++c) {
                    averages[m][c] = 0.0f;
                    for (int n = 0; n < 4; ++n) {
                        averages[m][c] += neighborhoods[m][n][c];
                    }
                    averages[m][c] /= 4.0f;
                }
            }

            float variances[4][3];
            for (int m = 0; m < 4; ++m) {
                for (int c = 0; c < 3; ++c) {
                    variances[m][c] = 0.0f;
                    for (int n = 0; n < 4; ++n) {
                        variances[m][c] += (neighborhoods[m][n][c] - averages[m][c]) * (neighborhoods[m][n][c] - averages[m][c]);
                    }
                    variances[m][c] /= 4.0f;
                }
            }

            int minVarianceIdx = 0;
            for (int m = 1; m < 4; ++m) {
                for (int c = 0; c < 3; ++c) {
                    if (variances[m][c] < variances[minVarianceIdx][c]) {
                        minVarianceIdx = m;
                    }
                }
            }

            for (int c = 0; c < 3; ++c) {
                values[offset * 3 + c] = averages[minVarianceIdx][c];
            }
        }
    }
}

__global__ void PiramidalnyFilterColorCUDA(Data dim_gpu, float* values_gpu) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 1 && i < dim_gpu.width - 2 && j > 1 && j < dim_gpu.height - 2) {
        const int offset = j * dim_gpu.width + i;

        float piramidalnyValue[3] = { 0.0f, 0.0f, 0.0f };

        for (int k = -2; k <= 2; ++k) {
            for (int l = -2; l <= 2; ++l) {
                int idx = ((j + k) * dim_gpu.width) + (i + l);
                for (int c = 0; c < 3; ++c) {
                    piramidalnyValue[c] += dim_gpu.dataArray[idx * 3 + c] * piramidalnyMask[(k + 2) * 5 + (l + 2)];
                }
            }
        }

        for (int c = 0; c < 3; ++c) {
            piramidalnyValue[c] /= 81.0f;
            values_gpu[offset * 3 + c] = piramidalnyValue[c];
        }
    }
}

void PiramidalnyFilterColorCPU(Data dim, float* values) {
    for (int i = 2; i < dim.width - 2; ++i) {
        for (int j = 2; j < dim.height - 2; ++j) {
            const int offset = j * dim.width + i;

            float piramidalnyValue[3] = { 0.0f, 0.0f, 0.0f };

            for (int k = -2; k <= 2; ++k) {
                for (int l = -2; l <= 2; ++l) {
                    int idx = ((j + k) * dim.width) + (i + l);
                    for (int c = 0; c < 3; ++c) {
                        piramidalnyValue[c] += dim.dataArray[idx * 3 + c] * piramidalnyMask[(k + 2) * 5 + (l + 2)];
                    }
                }
            }

            for (int c = 0; c < 3; ++c) {
                piramidalnyValue[c] /= 81.0f;
                values[offset * 3 + c] = piramidalnyValue[c];
            }
        }
    }
}

__global__ void MozaikowyFilterColorCUDA(Data dim_gpu, float* values_gpu, int tileSize) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < dim_gpu.width && j < dim_gpu.height) {
        const int offset = j * dim_gpu.width + i;

        int start_x = (i / tileSize) * tileSize;
        int start_y = (j / tileSize) * tileSize;

        float mozaikowyValue[3] = { 0.0f, 0.0f, 0.0f };
        int pixelCount = 0;

        for (int m = 0; m < tileSize; ++m) {
            for (int n = 0; n < tileSize; ++n) {
                int idx = (start_y + m) * dim_gpu.width + start_x + n;
                if (idx < dim_gpu.width * dim_gpu.height) {
                    for (int c = 0; c < 3; ++c) {
                        mozaikowyValue[c] += dim_gpu.dataArray[idx * 3 + c];
                    }
                    pixelCount++;
                }
            }
        }

        for (int c = 0; c < 3; ++c) {
            mozaikowyValue[c] /= static_cast<float>(pixelCount);
            values_gpu[offset * 3 + c] = mozaikowyValue[c];
        }
    }
}

void MozaikowyFilterColorCPU(Data dim, float* values, int tileSize) {
    for (int i = 0; i < dim.width; ++i) {
        for (int j = 0; j < dim.height; ++j) {
            const int offset = j * dim.width + i;

            int start_x = (i / tileSize) * tileSize;
            int start_y = (j / tileSize) * tileSize;

            float mozaikowyValue[3] = { 0.0f, 0.0f, 0.0f };
            int pixelCount = 0;

            for (int m = 0; m < tileSize; ++m) {
                for (int n = 0; n < tileSize; ++n) {
                    int idx = (start_y + m) * dim.width + start_x + n;
                    if (idx < dim.width * dim.height) {
                        for (int c = 0; c < 3; ++c) {
                            mozaikowyValue[c] += dim.dataArray[idx * 3 + c];
                        }
                        pixelCount++;
                    }
                }
            }

            for (int c = 0; c < 3; ++c) {
                mozaikowyValue[c] /= static_cast<float>(pixelCount);
                values[offset * 3 + c] = mozaikowyValue[c];
            }
        }
    }
}

__global__ void EmbossFilterCUDA(Data dim_gpu, float* values_gpu) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < dim_gpu.width - 1 && j > 0 && j < dim_gpu.height - 1) {
        const int offset = (j * dim_gpu.width + i) * 3;

        float embossKernel[3][3] = {
            { -2, -1, 0 },
            { -1, 1, 1 },
            { 0, 1, 2 }
        };

        float embossValueR = 0.0f;
        float embossValueG = 0.0f;
        float embossValueB = 0.0f;

        for (int m = -1; m <= 1; ++m) {
            for (int n = -1; n <= 1; ++n) {
                int idx = ((j + m) * dim_gpu.width + (i + n)) * 3;
                embossValueR += dim_gpu.dataArray[idx] * embossKernel[m + 1][n + 1];
                embossValueG += dim_gpu.dataArray[idx + 1] * embossKernel[m + 1][n + 1];
                embossValueB += dim_gpu.dataArray[idx + 2] * embossKernel[m + 1][n + 1];
            }
        }

        embossValueR = max(0.0f, min(255.0f, embossValueR));
        embossValueG = max(0.0f, min(255.0f, embossValueG));
        embossValueB = max(0.0f, min(255.0f, embossValueB));

        values_gpu[offset] = embossValueR;
        values_gpu[offset + 1] = embossValueG;
        values_gpu[offset + 2] = embossValueB;
    }
}

void EmbossFilterCPU(Data dim, float* values) {
    for (int i = 1; i < dim.width - 1; ++i) {
        for (int j = 1; j < dim.height - 1; ++j) {
            const int offset = (j * dim.width + i) * 3;

            float embossKernel[3][3] = {
                { -2, -1, 0 },
                { -1, 1, 1 },
                { 0, 1, 2 }
            };
  
            float embossValueR = 0.0f;
            float embossValueG = 0.0f;
            float embossValueB = 0.0f;

            for (int m = -1; m <= 1; ++m) {
                for (int n = -1; n <= 1; ++n) {
                    int idx = ((j + m) * dim.width + (i + n)) * 3;
                    embossValueR += dim.dataArray[idx] * embossKernel[m + 1][n + 1];
                    embossValueG += dim.dataArray[idx + 1] * embossKernel[m + 1][n + 1];
                    embossValueB += dim.dataArray[idx + 2] * embossKernel[m + 1][n + 1];
                }
            }

            embossValueR = max(0.0f, min(255.0f, embossValueR));
            embossValueG = max(0.0f, min(255.0f, embossValueG));
            embossValueB = max(0.0f, min(255.0f, embossValueB));

            values[offset] = embossValueR;
            values[offset + 1] = embossValueG;
            values[offset + 2] = embossValueB;
        }
    }
}

__global__ void SepiaFilterCUDA(Data dim_gpu, float* values_gpu, int W) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < dim_gpu.width && j < dim_gpu.height) {
        const int offset = j * dim_gpu.width + i;

        float sepiaValue[3] = { 0.0f, 0.0f, 0.0f };

        float grayValue = 0.299f * dim_gpu.dataArray[offset * 3 + 2] + 0.587f * dim_gpu.dataArray[offset * 3 + 1] + 0.114f * dim_gpu.dataArray[offset * 3];

        sepiaValue[0] = fminf(255.0f, grayValue);
        sepiaValue[1] = fminf(255.0f, grayValue + W);
        sepiaValue[2] = fminf(255.0f, grayValue + 2 * W);

        for (int c = 0; c < 3; ++c) {
            values_gpu[offset * 3 + c] = fminf(255.0f, fmaxf(0.0f, sepiaValue[c]));
        }
    }
}

void SepiaFilterCPU(Data dim, float* values, int W) {
    for (int i = 0; i < dim.width; ++i) {
        for (int j = 0; j < dim.height; ++j) {
            const int offset = j * dim.width + i;

            float sepiaValue[3] = { 0.0f, 0.0f, 0.0f };

            float grayValue = 0.299f * dim.dataArray[offset * 3 + 2] + 0.587f * dim.dataArray[offset * 3 + 1] + 0.114f * dim.dataArray[offset * 3];

            sepiaValue[0] = fminf(255.0f, grayValue);
            sepiaValue[1] = fminf(255.0f, grayValue + W);
            sepiaValue[2] = fminf(255.0f, grayValue + 2 * W);

            for (int c = 0; c < 3; ++c) {
                values[offset * 3 + c] = fminf(255.0f, fmaxf(0.0f, sepiaValue[c]));
            }
        }
    }
}

int main() {
    string imageName = "";
    bool rightimg = false;
    cv::Mat image;
    while(rightimg == false) {
        std::cout << "Insert Image name (with extension np. .jpg): ";
        std::cin >> imageName;
        system("CLS");
        image = imread(imageName);
        if (image.empty()) {
            std::cerr << "Error: Couldn't read the image." << std::endl;
            return -1;
        }
        else {
            rightimg = true;
        }
    }
 
    Data dim{};
    dim.width = image.cols;
    dim.height = image.rows;
    dim.dataArraySize = dim.width * dim.height * image.channels();
    dim.dataArray = new float[dim.dataArraySize];

    bool endProgram = false;
    float durationGPU = 0;
    float CPU_dur = 0;
    int filter = 0;
    while (endProgram == false) {
        cv::Mat copy = image.clone();
        filter = 0;
        std::cout << "Czas wykonania na CPU: " << CPU_dur << " mikrosekundy" << std::endl;
        std::cout << "Czas wykonania na GPU: " << durationGPU * 1000 << " mikrosekundy" << std::endl;
        durationGPU = 0;
        CPU_dur = 0;
        std::cout << "1. Sobel Filter" << std::endl;
        std::cout << "2. Laplace Filter" << std::endl;
        std::cout << "3. Kuwahara Filter" << std::endl;
        std::cout << "4. Pyramid Filter" << std::endl;
        std::cout << "5. Mosaic Filter" << std::endl;
        std::cout << "6. EmbossFilter3D" << std::endl;
        std::cout << "7. Sepia Filter" << std::endl;
        std::cout << "8. End Program" << std::endl;
        std::cout << "Which filter do you want to use?: ";
        std::cin >> filter;

        bool transformToGray = false;
        if (filter == 1 || filter == 2) {
            transformToGray = true;
        }

        if (transformToGray) {
            copy = convertToGrayscale(copy);
            for (int j = 0; j < dim.height; ++j) {
                for (int i = 0; i < dim.width; ++i) {
                    dim.dataArray[j * dim.width + i] = static_cast<float>(copy.at<uchar>(j, i));
                }
            }
        }
        else {
            for (int j = 0; j < dim.height; ++j) {
                for (int i = 0; i < dim.width; ++i) {
                    cv::Vec3b pixel = image.at<cv::Vec3b>(j, i);
                    for (int c = 0; c < image.channels(); ++c) {
                        dim.dataArray[j * dim.width * image.channels() + i * image.channels() + c] = static_cast<float>(pixel[c]);
                    }
                }
            }
        }

        float* values_cpu = new float[dim.dataArraySize];
        float* values = new float[dim.dataArraySize];
        float* values_gpu;
        cudaMalloc(&values_gpu, dim.dataArraySize * sizeof(float));

        Data dim_gpu{};
        dim_gpu.width = dim.width;
        dim_gpu.height = dim.height;
        dim_gpu.dataArraySize = dim.dataArraySize;
        cudaMalloc(&dim_gpu.dataArray, dim.dataArraySize * sizeof(float));
        cudaMemcpy(dim_gpu.dataArray, dim.dataArray, dim.dataArraySize * sizeof(float), cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blocksPerGrid(dim.width / threadsPerBlock.x, dim.height / threadsPerBlock.y);

        switch (filter) {
        case 1: {
            auto startCPU = std::chrono::high_resolution_clock::now();
            SobelFilter(dim, values);
            auto endCPU = std::chrono::high_resolution_clock::now();
            auto durationCPU = std::chrono::duration_cast<std::chrono::microseconds>(endCPU - startCPU);
            CPU_dur = durationCPU.count();
            cudaEvent_t startGPU, endGPU;
            cudaEventCreate(&startGPU);
            cudaEventCreate(&endGPU);
            cudaEventRecord(startGPU);
            SobelFilterCUDA << <blocksPerGrid, threadsPerBlock >> > (dim_gpu, values_gpu);
            cudaEventRecord(endGPU);
            cudaEventSynchronize(endGPU);
            cudaEventElapsedTime(&durationGPU, startGPU, endGPU);          
            cudaEventDestroy(startGPU);
            cudaEventDestroy(endGPU);
            break;
        }
        case 2: {
            auto startCPU = std::chrono::high_resolution_clock::now();
            LaplaceFilter(dim, values);
            auto endCPU = std::chrono::high_resolution_clock::now();
            auto durationCPU = std::chrono::duration_cast<std::chrono::microseconds>(endCPU - startCPU);
            CPU_dur = durationCPU.count();
            cudaEvent_t startGPU, endGPU;
            cudaEventCreate(&startGPU);
            cudaEventCreate(&endGPU);
            cudaEventRecord(startGPU);
            LaplaceFilterCUDA << <blocksPerGrid, threadsPerBlock >> > (dim_gpu, values_gpu);
            cudaEventRecord(endGPU);
            cudaEventSynchronize(endGPU);
            cudaEventElapsedTime(&durationGPU, startGPU, endGPU);
            cudaEventDestroy(startGPU);
            cudaEventDestroy(endGPU);
            break;
        }
        case 3: {
            auto startCPU = std::chrono::high_resolution_clock::now();
            KuwaharaFilterColorCPU(dim, values);
            auto endCPU = std::chrono::high_resolution_clock::now();
            auto durationCPU = std::chrono::duration_cast<std::chrono::microseconds>(endCPU - startCPU);
            CPU_dur = durationCPU.count();
            cudaEvent_t startGPU, endGPU;
            cudaEventCreate(&startGPU);
            cudaEventCreate(&endGPU);
            cudaEventRecord(startGPU);
            KuwaharaFilterColorCUDA << <blocksPerGrid, threadsPerBlock >> > (dim_gpu, values_gpu);
            cudaEventRecord(endGPU);
            cudaEventSynchronize(endGPU);
            cudaEventElapsedTime(&durationGPU, startGPU, endGPU);
            cudaEventDestroy(startGPU);
            cudaEventDestroy(endGPU);
            break;
        }
        case 4: {
            auto startCPU = std::chrono::high_resolution_clock::now();
            PiramidalnyFilterColorCPU(dim, values);
            auto endCPU = std::chrono::high_resolution_clock::now();
            auto durationCPU = std::chrono::duration_cast<std::chrono::microseconds>(endCPU - startCPU);
            CPU_dur = durationCPU.count();
            cudaEvent_t startGPU, endGPU;
            cudaEventCreate(&startGPU);
            cudaEventCreate(&endGPU);
            cudaEventRecord(startGPU);
            PiramidalnyFilterColorCUDA << <blocksPerGrid, threadsPerBlock >> > (dim_gpu, values_gpu);
            cudaEventRecord(endGPU);
            cudaEventSynchronize(endGPU);
            cudaEventElapsedTime(&durationGPU, startGPU, endGPU);
            cudaEventDestroy(startGPU);
            cudaEventDestroy(endGPU);
            break;
        }
        case 5:
        {
            string tile = "1";
            std::cout << "Type 'q' to exit";
            std::cout << "How many tiles would you like?: ";
            std::cin >> tile;
            if (tile == "q") {
                system("cls");
                continue;
            }
            int tile_size = std::stoi(tile);
            auto startCPU = std::chrono::high_resolution_clock::now();
            MozaikowyFilterColorCPU(dim, values,tile_size);
            auto endCPU = std::chrono::high_resolution_clock::now();
            auto durationCPU = std::chrono::duration_cast<std::chrono::microseconds>(endCPU - startCPU);
            CPU_dur = durationCPU.count();
            cudaEvent_t startGPU, endGPU;
            cudaEventCreate(&startGPU);
            cudaEventCreate(&endGPU);
            cudaEventRecord(startGPU);
            MozaikowyFilterColorCUDA << <blocksPerGrid, threadsPerBlock >> > (dim_gpu, values_gpu, tile_size);  
            cudaEventRecord(endGPU);
            cudaEventSynchronize(endGPU);
            cudaEventElapsedTime(&durationGPU, startGPU, endGPU);
            cudaEventDestroy(startGPU);
            cudaEventDestroy(endGPU);
            break;
        }
        case 6:
        {
            auto startCPU = std::chrono::high_resolution_clock::now();
            EmbossFilterCPU(dim, values);
            auto endCPU = std::chrono::high_resolution_clock::now();
            auto durationCPU = std::chrono::duration_cast<std::chrono::microseconds>(endCPU - startCPU);
            CPU_dur = durationCPU.count();
            cudaEvent_t startGPU, endGPU;
            cudaEventCreate(&startGPU);
            cudaEventCreate(&endGPU);
            cudaEventRecord(startGPU);
            EmbossFilterCUDA << <blocksPerGrid, threadsPerBlock >> > (dim_gpu, values_gpu);
            cudaEventRecord(endGPU);
            cudaEventSynchronize(endGPU);
            cudaEventElapsedTime(&durationGPU, startGPU, endGPU);
            cudaEventDestroy(startGPU);
            cudaEventDestroy(endGPU);
            break;
        }
        case 7: {
            string W = "0";
            std::cout << "Type 'q' to exit";
            std::cout << "What fill factor would you like?: ";
            std::cin >> W;
            int fillFactor = std::stoi(W);
            if (W == "q") {
                system("cls");
                continue;
            }
            auto startCPU = std::chrono::high_resolution_clock::now();
            SepiaFilterCPU(dim, values,fillFactor);
            auto endCPU = std::chrono::high_resolution_clock::now();
            auto durationCPU = std::chrono::duration_cast<std::chrono::microseconds>(endCPU - startCPU);
            CPU_dur = durationCPU.count();
            cudaEvent_t startGPU, endGPU;
            cudaEventCreate(&startGPU);
            cudaEventCreate(&endGPU);
            cudaEventRecord(startGPU);
            SepiaFilterCUDA << <blocksPerGrid, threadsPerBlock >> > (dim_gpu, values_gpu, fillFactor);
            cudaEventRecord(endGPU);
            cudaEventSynchronize(endGPU);
            cudaEventElapsedTime(&durationGPU, startGPU, endGPU);
            cudaEventDestroy(startGPU);
            cudaEventDestroy(endGPU);
            break;
        }
        case 8:
        {
            endProgram = true;
        }
        default:
            std::cerr << "Invalid choice of filter/command." << std::endl;
            return -1;
        }

        cudaMemcpy(values_cpu, values_gpu, dim.dataArraySize * sizeof(float), cudaMemcpyDeviceToHost);


        cv::Mat resultImage(dim.height, dim.width, transformToGray ? CV_8UC1 : CV_8UC3);
        for (int j = 0; j < dim.height; ++j) {
            for (int i = 0; i < dim.width; ++i) {
                if (transformToGray) {
                    resultImage.at<uchar>(j, i) = static_cast<uchar>(values_cpu[j * dim.width + i]);
                }
                else {
                    cv::Vec3b& pixel = resultImage.at<cv::Vec3b>(j, i);
                    for (int c = 0; c < image.channels(); ++c) {
                        pixel[c] = static_cast<uchar>(values_cpu[j * dim.width * image.channels() + i * image.channels() + c]);
                    }
                }
            }
        }
        
        int width = 800;
        int height = 600;
        Mat chart(height, width, CV_8UC3, Scalar(255, 255, 255));

        vector<float> data = { 0 ,0 ,0 ,CPU_dur ,0 ,durationGPU * 1000, 0 ,0 ,0 };

        drawBarChart(chart, data, width, height, CPU_dur, durationGPU*1000);

        cv::imshow("Bar Chart", chart);
        cv::imshow("Original Image", image);
        cv::imshow("Filtered Image", resultImage);
        cv::waitKey(0);

        delete[] values_cpu;
        delete[] values;
        cudaFree(dim_gpu.dataArray);
        cudaFree(values_gpu);
        system("CLS");
    }
    delete[] dim.dataArray;
    return 0;
}
