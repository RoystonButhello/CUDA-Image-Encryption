serial: serial.o 
	g++ -std=c++11 serial.o -o serial `pkg-config opencv --cflags --libs`

serial.o: serial.cpp
	g++ -std=c++11 serial.cpp -c 

clean:
	rm *.o
	
