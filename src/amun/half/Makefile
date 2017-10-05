#!/bin/bash

.secondary:

all: normal half half2

half: half.cu
	nvcc $< -arch sm_61 -lcublas -lcurand -o $@ --std=c++11

half2: half2.cu
	nvcc $< -arch sm_61 -lcublas -lcurand -o $@ --std=c++11

normal: normal.cu
	nvcc $< -arch sm_61 -lcublas -lcurand -o $@ --std=c++11
