cmake_minimum_required(VERSION 3.20)
project(nrl_ntwrks_frm_scrtch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -flto -march=native")

set(NN_SOURCES
        nn/layers/linear_layer.cpp
        nn/layers/nonlinear_layer.cpp
        nn/activation/relu_func.cpp
        nn/activation/sigmoid_func.cpp
        nn/activation/tanh_func.cpp
        nn/except.cpp
        nn/layers/random.cpp
        nn/layers/network.cpp
        nn/dataloader.cpp
        nn/train/train_loop.cpp
        nn/datasets/mnist_loader.cpp
        nn/datasets/idx_loader.cpp
        nn/datasets/fashion_mnist_loader.cpp
)

add_executable(nrl_ntwrks_frm_scrtch
        app/main.cpp
        app/app.cpp
        ${NN_SOURCES}
        nn/datasets/idx_loader.cpp
        nn/datasets/fashion_mnist_loader.cpp
)

add_executable(run_all_tests
        app/test_main.cpp
        app/run_all_tests.cpp
        ${NN_SOURCES}
)

target_include_directories(nrl_ntwrks_frm_scrtch PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
        ${CMAKE_CURRENT_SOURCE_DIR}/external/eigenrand
        ${CMAKE_CURRENT_SOURCE_DIR}/nn
)

target_include_directories(run_all_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
        ${CMAKE_CURRENT_SOURCE_DIR}/external/eigenrand
        ${CMAKE_CURRENT_SOURCE_DIR}/nn
)