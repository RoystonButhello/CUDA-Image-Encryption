#!/bin/bash
nvcc -O3 -std=c++11 kernel.cu main.cpp -o main `pkg-config opencv --cflags --libs` -I/opt/ssl/include/ -L/opt/ssl/lib/ -lcrypto
nvcc -O3 -std=c++11 kernel.cu decrypt.cpp -o decrypt `pkg-config opencv --cflags --libs` -I/opt/ssl/include/ -L/opt/ssl/lib/ -lcrypto
echo "Encrypting and decrypting $1"
./main $1
./decrypt $1

