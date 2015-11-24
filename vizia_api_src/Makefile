all: doom_controller vizia_main python_api clean

doom_controller: ViziaDoomController.cpp ViziaDoomController.h ViziaDoomControllerExample.cpp boost/process.hpp
	g++ -Wall -I . ViziaDoomController.cpp ViziaDoomControllerExample.cpp -o controllerExample -lrt -lpthread -L/usr/local/lib -lboost_system -lboost_thread -lSDL2

vizia_main: ViziaMain.cpp ViziaMain.h ViziaDoomController.h boost/process.hpp
	g++ -Wall -I . ViziaDoomController.cpp ViziaMain.cpp ViziaMainExample.cpp -o mainExample -lrt -lpthread -L/usr/local/lib -lboost_system -lboost_thread

PythonVizia.o: PythonVizia.cpp ViziaPythonApi.cpp ViziaMain.h ViziaDoomController.h  ViziaMain.cpp 
	g++  -c -fPIC PythonVizia.cpp -o PythonVizia.o -I . -lrt -lpthread  -L/usr/local/lib -lboost_system -lboost_thread

python_api: PythonVizia.o
	g++ -shared -Wl,-soname,vizia.so -o vizia.so PythonVizia.o -lpython2.7 -lboost_python -I . -lrt -lpthread  -L/usr/local/lib -lboost_system -lboost_thread

clean:
	rm -rf *.o