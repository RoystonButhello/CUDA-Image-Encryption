#include <iostream>
#include <stdio.h>
#include <string>
#include <random>
#include <chrono>
#include <fstream>
#include <stdint.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <typeinfo>

using namespace cv;
using namespace std;
using namespace thrust;
using namespace chrono;

thrust::host_vector<int> genRelocVec(int, int, vector<double>&);
void columnRotator(Mat3b, Mat3b, int, int, int);
void rowRotator(Mat3b, Mat3b, int, int, int);

int main()
{
    // Initiliaze file-related strings
    string src = "images/";
    string temp = "temp/";
    string target = "cat.png";
    string fn_img_in = src + target;
    string fn_img_out = fn_img_in + "_ENC";
    string fn_vars = temp + "vars.txt";

    int rounds = 4;

    // Read the file and confirm it's been opened
    Mat imgin = cv::imread(fn_img_in, 1);
    if (!imgin.data)
    {
        cout << "Image not found!\n";
        return -1;
    }

    int M = imgin.rows, N = imgin.cols; // Read image dimensions

    // Generate permutation and diffusion vectors
    vector<double> P1(M * N * 2);
    vector<double> P2(M * N * 2);
    auto U = genRelocVec(M, N, P1);
    auto V = genRelocVec(N, M, P2);

    Mat3b imgout(M, N); // Intermediary Matrix

    //auto start = steady_clock::now();
    //cout << "genRelocVec: " << duration_cast<milliseconds>(steady_clock::now() - start).count() << "ms\n";

    // Permutation
    for (int i = 0; i < rounds; i++)
    {
        for (int j = 0; j < N; j++) // For each column
        {
            columnRotator(imgout, imgin.col(j), j, U[j], M);
        }

        for (int j = 0; j < M; j++) // For each row
        {
            rowRotator(imgin, imgout.row(j), j, V[j], N);
        }
    }
    
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

    cv::imshow("Encrypted", imgin);
    cv::imwrite(fn_img_out, imgin);
    
    // Unpermutation
    for (int i = 0; i < rounds; i++)
    {
        for (int j = 0; j < M; j++)
        {
            rowRotator(imgout, imgin.row(j), j, -V[j], N);
        }

        for (int j = 0; j < N; j++)
        {
            columnRotator(imgin, imgout.col(j), j, -U[j], M);
        }
    }
    
    cv::imshow("Decrypted", imgin);
    waitKey(0);
    return 0;
}

thrust::host_vector<int> genRelocVec(int M, int N, vector<double> &randomReal)
{
    //Initiliaze Generators
    double unzero = 0.0000000001;
    random_device randev;
    mt19937 seeder(randev());
    uniform_int_distribution<int> intGen(1, 32);
    uniform_real_distribution<double> realGen(unzero, 1);

    //Initiliaze parameters
    auto a = intGen(seeder);
    auto b = intGen(seeder);
    auto c = a * b + 1;
    auto x = realGen(seeder);
    auto y = realGen(seeder);
    auto offset = intGen(seeder);

    //Skip first few values in sequence
    for (int i = 0; i < offset; i++)
    {
        x = fmod(x + a * y, 1) + unzero;
        y = fmod(b * x + c * y, 1) + unzero;
    }

    //Generate vector of real numbers in the interval (0,1)
    int limit = M * N;
    for (int i = 0; i < limit; i++)
    {
        x = fmod(x + a * y, 1) + unzero;
        y = fmod(b * x + c * y, 1) + unzero;
        randomReal[i * 2] = x;
        randomReal[i * 2 + 1] = y;
    }

    thrust::host_vector<int> relocVec(N);
    uniform_int_distribution<int> offsetGen(1, N * (M - 1));
    auto vec_offset = offsetGen(seeder);
    int exp = (int)pow(10, 8);
    for (int i = 0; i < N; i++)
    {
        relocVec[i] = (int)(randomReal[vec_offset + i] * exp) % M;
    }
    return relocVec;
}

void columnRotator(Mat3b img, Mat3b col, int index, int offset, int M)
{
    // M elements per column
    if (offset > 0)
    {
        for (int k = 0; k < M; k++)
        {
            img.at<Vec3b>(k, index) = col.at<Vec3b>((k + offset) % M, 0);
        }
    }
    else if (offset < 0)
    {
        for (int k = 0; k < M; k++)
        {
            img.at<Vec3b>(k, index) = col.at<Vec3b>((k + offset + M) % M, 0);
        }
    }
    else
    {
        for (int k = 0; k < M; k++)
        {
            img.at<Vec3b>(k, index) = col.at<Vec3b>(k, 0);
        }
    }
}

void rowRotator(Mat3b img, Mat3b row, int index, int offset, int N)
{
    // N elements per row
    if (offset > 0)
    {
        for (int k = 0; k < N; k++)
        {
            img.at<Vec3b>(index, k) = row.at<Vec3b>(0, (k + offset) % N);
        }
    }
    else if (offset < 0)
    {
        for (int k = 0; k < N; k++)
        {
            img.at<Vec3b>(index, k) = row.at<Vec3b>(0, (k + offset + N) % N);
        }
    }
    else
    {
        for (int k = 0; k < N; k++)
        {
            img.at<Vec3b>(index, k) = row.at<Vec3b>(0, k);
        }
    }
}