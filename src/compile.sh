#!/bin/bash	
g++ -I. ViziaDoomController.cpp ViziaDoomControllerExample.cpp -o controllerExample -lrt -lpthread -L/usr/local/lib -lboost_system -lboost_thread -lSDL2
g++ -I. ViziaDoomController.cpp ViziaMain.cpp ViziaDoomControllerExample.cpp -o mainExample -lrt -lpthread -L/usr/local/lib -lboost_system -lboost_thread
