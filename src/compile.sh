#!/bin/bash	
g++ -I. ViziaDoomController.cpp ViziaDoomControllerExample.cpp -o example -lrt -lpthread -L/usr/local/lib -lboost_system -lboost_thread -lSDL2
