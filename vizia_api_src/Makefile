all: python_api mainExample controllerExample

python_api: vizia.so
vizia.so: ViziaPythonWrapper.cpp ViziaMainPython.o ViziaMain.o ViziaDoomController.o ViziaMainPython.h
	g++ -fPIC -shared -Wl,-soname,vizia.so -o vizia.so ViziaPythonWrapper.cpp ViziaMainPython.o ViziaMain.o ViziaDoomController.o -lpython2.7 -lboost_python -I . -I /usr/include/python2.7 -lrt -lpthread  -L/usr/local/lib -lboost_system -lboost_thread 

ViziaMainPython.o:  ViziaMainPython.cpp ViziaMainPython.h ViziaMain.h ViziaDefines.h
	g++ -Wall -fPIC -I . -I /usr/include/python2.7 -c -fPIC ViziaMainPython.cpp  -lrt -lpthread  -L/usr/local/lib -lboost_system -lboost_thread -lboost_python -lpython2.7

mainExample: ViziaMainExample.cpp ViziaMain.o ViziaDoomController.o ViziaMain.h   
	g++ -Wall -I . ViziaMainExample.cpp ViziaMain.o  ViziaDoomController.o -o mainExample -lrt -lpthread -L/usr/local/lib -lboost_system -lboost_thread
ViziaMain.o: ViziaMain.cpp ViziaMain.h ViziaDoomController.h  ViziaDefines.h
	g++ -Wall -fPIC -I . -c  ViziaMain.cpp -lrt -lpthread -L/usr/local/lib -lboost_system -lboost_thread

controllerExample:  ViziaDoomControllerExample.cpp  ViziaDoomController.h  ViziaDefines.h ViziaDoomController.o
	g++ -Wall -I . -o controllerExample ViziaDoomController.o ViziaDoomControllerExample.cpp  -lrt -lpthread -L/usr/local/lib -lboost_system -lboost_thread -lSDL2
ViziaDoomController.o: ViziaDoomController.cpp ViziaDoomController.h ViziaDefines.h boost/process.hpp
	g++ -Wall -fPIC -I . -c ViziaDoomController.cpp -lrt -lpthread -L/usr/local/lib -lboost_system -lboost_thread -lSDL2


clean:
	rm -rf *.o vizia.so
