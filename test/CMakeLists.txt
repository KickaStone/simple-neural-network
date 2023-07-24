cmake_minimum_required(VERSION 3.27)

set(this Test)

set(SOURCES 
    kernel_test.cu
    net_test.cu
    load_test.cu 
)

find_package(spdlog CONFIG REQUIRED)

add_executable(${this} ${SOURCES})

target_link_libraries(${this} PUBLIC
    gtest_main
    gmock_main
    gmock
    gtest
    spdlog::spdlog
    ${PROJECT_NAME}
)

set_target_properties(
    ${this} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 86
)

add_test(
    NAME ${this}
    COMMAND ${this}
)




