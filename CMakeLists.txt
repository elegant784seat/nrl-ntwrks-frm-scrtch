cmake_minimum_required(VERSION 3.20)
project(nrl_ntwrks_frm_scrtch)

set(CMAKE_CXX_STANDARD 20)

set(APP_TARGET nrl_ntwrks_frm_scrtch)

add_executable(${APP_TARGET}
        app/main.cpp
        nn/include/activation.hpp
        nn/include/loss.hpp
        nn/include/layer.hpp
        nn/include/verify.hpp
)

target_include_directories(${APP_TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
        ${CMAKE_CURRENT_SOURCE_DIR}/external/eigenrand
        ${CMAKE_CURRENT_SOURCE_DIR}/nn/include
)