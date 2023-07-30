cmake_minimum_required(VERSION 3.26)

set(this Network)

project(${this} C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED)

enable_testing()

add_subdirectory(googletest)

set(HEADER_SET
        layer.h
        dense.h
        activation.h
        MLP.h
        mnist_loader.h
)

set(SOURCE_SET
        convolution.cu
        dense.cpp
        MLP.cpp
        Net.h
        # main.cpp
)

add_library(${this} STATIC  ${HEADER_SET} ${SOURCE_SET})

add_subdirectory(test)

target_link_libraries(
    ${this} PUBLIC
    ${OpenCV_LIBS}
)