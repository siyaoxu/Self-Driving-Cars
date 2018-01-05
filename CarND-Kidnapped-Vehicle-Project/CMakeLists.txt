cmake_minimum_required(VERSION 3.8)
project(CarND_Kidnapped_Vehicle_Project)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        src/helper_functions.h
        src/json.hpp
        src/main.cpp
        src/map.h
        src/particle_filter.cpp
        src/particle_filter.h)

add_executable(CarND_Kidnapped_Vehicle_Project ${SOURCE_FILES})