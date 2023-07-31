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
        layers/layer.h
        layers/dense.h
        utils/activation.h
        MLP.h
        utils/mnist_loader.h
        layers/conv.h
        layers/pooling.h
        utils/convolution.h
        CNN.h
        layers/layer_type.h
)

set(SOURCE_SET
        layers/dense.cpp
        MLP.cpp
        Net.h
        layers/conv.cpp
        layers/conv.h
        layers/pooling.cpp
        utils/convolution.cpp
        CNN.cpp
        CNN.h
)

add_library(${this} STATIC  ${HEADER_SET} ${SOURCE_SET})

add_subdirectory(test)

target_link_libraries(
    ${this} PUBLIC
    ${OpenCV_LIBS}
)