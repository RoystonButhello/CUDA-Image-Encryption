g++ -std=c++11 mae.cpp -o mae `pkg-config opencv --cflags --libs`
g++ -std=c++11 npcr_uaci.cpp -o mae `pkg-config opencv --cflags --libs`
g++ -std=c++11 mse.cpp -o mse `pkg-config opencv --cflags --libs`
g++ -std=c++11 resize.cpp -o resize `pkg-config opencv --cflags --libs`
g++ -std=c++11 pixel_replace.cpp -o pixel_replace `pkg-config opencv --cflags --libs`
