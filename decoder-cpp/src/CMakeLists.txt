cmake_minimum_required(VERSION 2.6)

project(decoder-cpp)


include_directories(../include)
include_directories(../lm)
include_directories(../util)

aux_source_directory(. SRC_SOURCE)
add_library(src_lib ${SRC_SOURCE})


target_link_libraries(src_lib kenlm kenlm_util)
