#!/usr/bin/env bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hpcheaf1/hieu/workspace/cuda-9.0/lib64

./gemm_normal
./gemm_fp16
