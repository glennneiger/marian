#!/bin/bash

.secondary:

all: gemm_normal gemm_half gemm_half2

gemm_half: gemm_half.cu
	nvcc $< -arch sm_61 -lcublas -lcurand -o $@ --std=c++11

gemm_half2: gemm_half2.cu
	nvcc $< -arch sm_61 -lcublas -lcurand -o $@ --std=c++11

gemm_normal: gemm.cu
	nvcc $< -arch sm_61 -lcublas -lcurand -o $@ --std=c++11
