cmake_minimum_required(VERSION 3.27)

set(this Network)

project(${this} C CXX CUDA)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

set(HEADER_SET
    Network.h
    mnist.h
    Mnist_helper.h
    kernel.cuh
    )
    
set(SOURCE_SET
    Network.cpp
    Mnist_helper.cpp
    train.cu
    kernel.cu 
)

add_executable(${this} ${SOURCE_SET} ${HEADER_SET})

set_property(TARGET ${this} PROPERTY CUDA_ARCHITECTURES 86)

set_target_properties(
    ${this} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)