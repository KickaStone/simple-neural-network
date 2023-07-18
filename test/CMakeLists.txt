cmake_minimum_required(VERSION 3.27)

set(this Test)

set(Sources
    NetworkTest.cpp
)

set(Headers
    NetworkTest.h
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

