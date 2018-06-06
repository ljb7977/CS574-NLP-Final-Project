project(nmtrnng)

set(CMAKE_CXX_STANDARD 11)

cmake_minimum_required(VERSION 3.10)

include_directories ("${PROJECT_SOURCE_DIR}/Eigen")

add_executable(NMTRNNG AttentionBiEncDec.cpp BlackOut.cpp
        LSTM.cpp
        NMTRNNG.cpp
        SoftMax.cpp
        TreeLSTM.cpp
        Vocabulary.cpp
        N3LP)