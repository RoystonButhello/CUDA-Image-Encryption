#include "Common.hpp"
#include "SerialFunctions.hpp"

using namespace cv;
using namespace std;
using namespace thrust;
using namespace chrono;

int main()
{
    // Read the file and confirm it's been opened
    Mat imgin = cv::imread(path.fn_img_in, 1);
    if (!imgin.data)
    {
        cout << "Image not found!\n";
        return -1;
    }

    // Read image dimensions
    const int M = imgin.rows, N = imgin.cols;

    // Generate permutation and diffusion vectors
    vector<double> P1(M * N * 2);
    vector<double> P2(M * N * 2);
    const auto U = genRelocVec(M, N, P1);
    const auto V = genRelocVec(N, M, P2);

    /*
    Serial::Permute(imgin, U, V, 4);
    cv::imshow("Encrypted", imgin);
    cv::imwrite(path.fn_img_out, imgin);
    Serial::UnPermute(imgin, U, V, 4);
    */

    auto cudaStatus = CudaPermute(imgin, U, V);
    if (cudaStatus != cudaSuccess)
    {
        cerr << "CudaPermute Failed!\n";
        return 1;
    }
    cv::imshow("Encrypted", imgin);
    
    /*
    // Convert image to (effectively) an array
    auto imgVec = imgin.reshape(1, M*N);

    // Allocate memory for diffusion vectors
    int len = M * N;
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

    imgin = imgVec.reshape(N,M);
    */

    //cv::imshow("Decrypted", imgin);

    cudaDeviceReset();
    waitKey(0);
    return 0;
}