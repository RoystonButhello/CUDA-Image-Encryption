#!/bin/bash
g++ -std=c++11 readstructure.cpp -Wno-sizeof-array-argument -o readstructure `pkg-config opencv --cflags --libs`
g++ -std=c++11 writestructure.cpp -Wno-sizeof-array-argument -o writestructure `pkg-config opencv --cflags --libs`
./writestructure
printf "\n"
./readstructure 
