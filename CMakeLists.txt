cmake_minimum_required(VERSION 3.27)

set(this Network)

project(${this} C CXX CUDA)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

add_subdirectory(googletest)

set(HEADER_SET
    mnist.h
    Mnist_helper.h
    mathkernel.cuh
    network.cuh
)

set(SOURCE_SET
    mathkernel.cu
    Mnist_helper.cpp
    Network.cu
)

enable_testing()

add_library(${this} STATIC ${SOURCE_SET} ${HEADER_SET})

target_link_libraries(
    ${this} PRIVATE
    curand
)

add_subdirectory(test)