cmake_minimum_required(VERSION 3.10...3.18)

project(${DD_LIBRARY} LANGUAGES CXX)

set(CMAKE_CXX_STANDART 17)
set(CMAKE_CXX_STANDART_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SOURCE_EXE main.cpp)

add_library(${DD_LIBRARY} STATIC
	src/dpcpp_dijkstra.cpp
	include/dpcpp_dijkstra.h
	)

target_include_directories(${DD_LIBRARY} PUBLIC include)

add_executable(DD_main ${SOURCE_EXE})

target_link_libraries( ${DD_LIBRARY} PUBLIC ${IO_LIBRARY})

target_link_libraries(DD_main PUBLIC ${DD_LIBRARY})