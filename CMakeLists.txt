cmake_minimum_required(VERSION 3.10...3.18)

set(PROJECT_NAME DPCPP_dijkstra)

project(${PROJECT_NAME})

set(IO_LIBRARY graphIO)
set(DD_LIBRARY dijkstra)

#subprojects
add_subdirectory(graphIO)
add_subdirectory(dijkstra)