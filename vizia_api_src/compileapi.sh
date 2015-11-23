#!/bin/bash
export CPLUS_INCLUDE_PATH=/usr/include/python2.7 	
g++ -c -fPIC api.cpp -o api.o -I. -lrt -lpthread  -L/usr/local/lib -lboost_system -lboost_thread
g++ -shared -Wl,-soname,api.so -o api.so  api.o -lpython2.7 -lboost_python -I. -lrt -lpthread  -L/usr/local/lib -lboost_system -lboost_thread
