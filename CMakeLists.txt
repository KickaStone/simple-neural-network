cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

set(this Network)

project(${this} C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
# find_package(kissfft CONFIG REQUIRED)

enable_testing()

add_compile_options(
       -Wall -Werror -Wno-error=maybe-uninitialized
       $<$<CONFIG:RELEASE>:-Ofast>
       $<$<CONFIG:DEBUG>:-O0>
       $<$<CONFIG:DEBUG>:-ggdb3>
)
add_compile_definitions(
        $<$<CONFIG:RELEASE>:NDEBUG>
        $<$<CONFIG:RELEASE>:BOOST_DISABLE_ASSERTS>
)
add_subdirectory(googletest)

set(HEADER_SET
        layers/layer.h
        layers/dense.h
        layers/pooling.h
        layers/conv.h
        layers/layer_type.h
        utils/eigen_helper.h
        utils/activation.h
        utils/mnist_loader.h
        CNN.h
        MLP.h
)

set(SOURCE_SET
        layers/dense.cpp
        MLP.cpp
        Net.h
        layers/conv.cpp
        layers/conv.h
        layers/pooling.cpp
        utils/eigen_helper.cpp
        CNN.cpp
        CNN.h
)

add_library(${this} STATIC  ${HEADER_SET} ${SOURCE_SET})

add_subdirectory(test)

target_link_libraries(
    ${this} PUBLIC
Eigen3::Eigen
)