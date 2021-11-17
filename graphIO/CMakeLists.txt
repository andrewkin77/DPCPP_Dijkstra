cmake_minimum_required(VERSION 3.10...3.18)

add_subdirectory(mmio)

project(${IO_LIBRARY} LANGUAGES CXX)

set(CMAKE_CXX_STANDART 17)
set(CMAKE_CXX_STANDART_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SOURCE_EXE main.cpp)

add_library(${IO_LIBRARY} STATIC
	src/graphio.cpp
	include/graphio.h)

target_include_directories(${IO_LIBRARY} PUBLIC include)

target_link_libraries(${IO_LIBRARY} PUBLIC mmio)

add_executable(IOmain ${SOURCE_EXE})

target_link_libraries(IOmain PUBLIC ${IO_LIBRARY})