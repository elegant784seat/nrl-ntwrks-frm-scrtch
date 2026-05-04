cmake_minimum_required(VERSION 3.20)
project(nrl_ntwrks_frm_scrtch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(APP_TARGET nrl_ntwrks_frm_scrtch)

add_executable(${APP_TARGET}
        app/main.cpp
        nn/layers/linear_layer.cpp
        nn/layers/nonlinear_layer.cpp
        nn/activation/relu_func.cpp
        nn/activation/sigmoid_func.cpp
        nn/activation/tanh_func.cpp
        nn/except.cpp
        nn/layers/random.cpp
        app/run_all_tests.cpp
        nn/layers/network.cpp
)

target_include_directories(${APP_TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
        ${CMAKE_CURRENT_SOURCE_DIR}/external/eigenrand
        ${CMAKE_CURRENT_SOURCE_DIR}/nn
)
