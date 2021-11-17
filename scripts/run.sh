#!/bin/bash

# Expects two parameters:
# 	graph name, dafault COL
# 	configuratoin type, default Debug

# Constants
PRPL='\033[0;35m' # Purple
GRN='\033[1;32m' # Green
CN='\033[0;36m' # Cyan
NC='\033[0m' # No Color

if [[ $# -eq 0 ]]; then
	echo -e "${PRPL}No parameters were given"
	echo -e "Using default parameter - ${CN}CAL${NC}"
	gr_name="CAL"
	input_gr="$(pwd)/../files/Input/CAL.gr"
	corr_bin="$(pwd)/../files/Output/bin_res/corSSSP_CAL.bin"
	output_file="$(pwd)/../files/Output/mySSSP_CAL.txt"
else
	input_gr="$(pwd)/../files/Input/$1.gr"
	corr_bin="$(pwd)/../files/Output/bin_res/corSSSP_$1.bin"
	output_file="$(pwd)/../files/Output/mySSSP_$1.txt"
fi

main_params= "$input_gr $output_file $corr_bin $gr_name"
cmake_params= "--build ./build --target DD_main"


# if [[ $# < 2 ]]; then
# 	cd ../graphAlgorithms/Debug
# else
# 	cd ../graphAlgorithms/$2
# fi


# Building
CXX=dpcpp cmake $cmake_params

# Checking if build was successful
if [[ $? -eq 0 ]]; then
	echo -e "${GRN}Starting DD_main...${NC}"
	./build/dijkstra/DD_main $main_params
	echo -e "${GRN}Finished DD_main${NC}"
fi