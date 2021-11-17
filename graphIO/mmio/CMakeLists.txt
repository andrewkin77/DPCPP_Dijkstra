cmake_minimum_required(VERSION 3.10...3.18)

project(mmio)

add_library(mmio STATIC
	src/mmio.c
	include/mmio.h)

target_include_directories(mmio PUBLIC include)