cmake_minimum_required(VERSION 3.20)
project(nrl_ntwrks_frm_scrtch)

set(CMAKE_CXX_STANDARD 20)

set(APP_TARGET nrl_ntwrks_frm_scrtch)

add_executable(${APP_TARGET}
        app/main.cpp
        nn/include/any_func.hpp
        nn/include/loss.hpp
        nn/include/layers/linear_layer.hpp
        nn/include/tensor.hpp
        nn/include/verify/verify.hpp
        nn/include/verify/location.hpp
        nn/source/tensor.cpp
        nn/source/layers/linear_layer.cpp
        nn/include/layers/nonlinear_layer.hpp
        nn/source/layers/nonlinear_layer.cpp
)

target_include_directories(${APP_TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
        ${CMAKE_CURRENT_SOURCE_DIR}/external/eigenrand
        ${CMAKE_CURRENT_SOURCE_DIR}/nn/include
)