serialdecrypt: serialdecrypt.o 
	g++ -std=c++11 serialdecrypt.o -o serialdecrypt `pkg-config opencv --cflags --libs`

serialdecrypt.o: serialdecrypt.cpp
	g++ -std=c++11 serialdecrypt.cpp -c 

clean:
	rm *.o
	
