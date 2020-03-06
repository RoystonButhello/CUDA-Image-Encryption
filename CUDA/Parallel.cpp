#include "Common.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;
using namespace thrust;
using namespace chrono;

int main()
{
    kernel_WarmUp();

    // Read the file and confirm it's been opened
    Mat img = cv::imread(path.fn_img_in, 1);
    if (!img.data)
    {
        cout << "Image not found!\n";
        return -1;
    }

    // Read image dimensions
    const int dim[3] = { img.rows, img.cols, img.channels() };

    // Generate permutation and diffusion vectors
    vector<double> P1(dim[0] * dim[1] * 2);
    vector<double> P2(dim[0] * dim[1] * 2);
    const auto U = genRelocVec(dim[0], dim[1], P1);
    const auto V = genRelocVec(dim[1], dim[0], P2);

    // Upload image to device
    uint8_t *d_img, *d_imgtmp;
    size_t data_size = img.rows * img.cols * img.channels();
    cudaMalloc<uint8_t>(&d_img, data_size);
    cudaMalloc<uint8_t>(&d_imgtmp, data_size);
    cudaMemcpy(d_img, img.data, data_size, cudaMemcpyHostToDevice);

    auto cudaStatus = CudaPermute(d_img, d_imgtmp, dim, U, V);
     if (cudaStatus != cudaSuccess)
    {
        cerr << "CudaPermute Failed!\n";
        return 1;
    }

    cudaMemcpy(img.data, d_img, data_size, cudaMemcpyDeviceToHost);
    cv::imshow("Encrypted", img);

    /*
    // Convert image to (effectively) an array
    auto imgVec = img.reshape(1, dim[0]*dim[1]);

    // Allocate memory for diffusion vectors
    int len = dim[0] * dim[1];
    Mat3b fDiff(len, 1);
    Mat3b rDiff(len, 1);

    // Initiliaze diffusion parameters
    random_device randev;
    mt19937 seeder(randev());
    uniform_int_distribution<uint8_t> intGen(0, 8);
    auto alpha = intGen(seeder);
    auto beta = intGen(seeder);
    auto f = intGen(seeder);
    auto r = intGen(seeder);

    img = imgVec.reshape(dim[1],dim[0]);
    */

    //cv::imshow("Decrypted", img);

    cudaDeviceReset();
    waitKey(0);
    return 0;
}