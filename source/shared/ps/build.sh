#!/bin/bash

g++ -fPIC --shared -std=c++11 -Wall *.cpp \
-I/opt/python/include \
-I/opt/python/include/python3.6m \
-I${HOME}/.local/include/python3.6m \
-L/opt/python/lib/ -lpython3.6m \
-o ps.so
