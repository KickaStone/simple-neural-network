cmake_minimum_required(VERSION 3.26)

set(this Test)

set(Sources
    # eigen_test.cpp
    conv_test.cpp
    # mlp_test.cpp
    cnn_test.cpp
    # pooling_test.cpp
)

set(Headers
)

add_executable(${this} ${Sources})

target_link_libraries(${this} PUBLIC
    gtest_main
    Network
)

add_test(
    NAME ${this}
    COMMAND ${this}
)


