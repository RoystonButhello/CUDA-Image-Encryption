#!/bin/bash
nvcc -std=c++11 kernels.cu Main.cpp -o Main  `pkg-config opencv --cflags --libs`
./Main
