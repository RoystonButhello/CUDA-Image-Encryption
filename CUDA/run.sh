#!/bin/bash
nvcc -std=c++11 kernels.cu -x cu Main.cpp -o Main `pkg-config opencv --cflags --libs` -I/opt/ssl/include/ -L/opt/ssl/lib/ -lcrypto
./Main
