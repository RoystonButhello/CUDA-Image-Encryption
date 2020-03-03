#pragma once
#include "Common.hpp"

namespace Serial
{
    // Called by Serial(Un)Permute
    void columnRotator(Mat3b img, Mat3b col, const int index, const int offset, const int M)
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

    // Called by Serial(Un)Permute
    void rowRotator(Mat3b img, Mat3b row, const int index, const int offset, const int N)
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

    // Permute <imgin> <rounds> times using column-rotation vector U and row-rotation vector V
    void Permute(Mat3b imgin, const thrust::host_vector<int> U, const thrust::host_vector<int> V, const int rounds)
    {
        int M = imgin.rows, N = imgin.cols;
        Mat3b imgout(M, N);
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
    }

    // Unpermute <imgin> <rounds> times using column-rotation vector U and row-rotation vector V
    void UnPermute(Mat3b imgin, const thrust::host_vector<int> U, const thrust::host_vector<int> V, const int rounds)
    {
        int M = imgin.rows, N = imgin.cols;
        Mat3b imgout(M, N);
        for (int i = 0; i < rounds; i++)
        {
            for (int j = 0; j < M; j++) // For each row
            {
                rowRotator(imgout, imgin.row(j), j, -V[j], N);
            }
            for (int j = 0; j < N; j++) // For each column
            {
                columnRotator(imgin, imgout.col(j), j, -U[j], M);
            }
        }
    }
}