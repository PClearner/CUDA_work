
CUDA_PATH := /usr/local/corex

.PHONY: all
all:
	clang++ -std=c++11 --cuda-path=${CUDA_PATH} -I${CUDA_PATH}/include --cuda-gpu-arch=ivcore11 --cuda-gpu-arch=ivcore10 -L${CUDA_PATH}/lib64 -lcudart -lcublas -o ./gemm/result ./gemm/m_gemm.cu
	clang++ -std=c++11 --cuda-path=${CUDA_PATH} -I${CUDA_PATH}/include --cuda-gpu-arch=ivcore11 --cuda-gpu-arch=ivcore10 -L${CUDA_PATH}/lib64 -lcudart -lcublas -o ./reduce/result ./reduce/m_reduce.cu