#include "dpcpp_dijkstra.h"
#include <chrono>

using namespace sycl;

// First pararmeter - patth to input file
// Second parameter - path to output file
// Third paremeter - path to bin file with correct result
// Fourth parameter - graph name
int main(int argc, char* argv[]) {
    queue q(gpu_selector{});
    std::cout << "Device: " << q.get_device().get_info<info::device::name>() << std::endl;

    crsGraph gr;
    const char* input = "/home/u89269/andrew/files/Input/test.mtx"; //argv[1];
    const char* output = "/home/u89269/andrew/files/Output/mySSSP_test.txt"; //argv[2];
    const char* corr_bin = "/home/u89269/andrew/files/Output/bin_res/corSSSP_CAL.bin"; //argv[3];

    /* Creating graph */
    init_graph(&gr);
    if (read_mtx_to_crs(&gr, input, q)) {
        std::cout << "Read failed" << std::endl;
        return 1;
    }
    // Creating distance array
    gr.d = malloc_shared<float>(gr.V, q);

    std::cout << "Starting dijkstra" << std::endl;
    /* Calculating SSSP array */
    auto start = std::chrono::high_resolution_clock::now();
    int err = dpcpp_dijkstra(gr, 0, 10, q);
    auto stop = std::chrono::high_resolution_clock::now();

    /* Calculating time */
    std::chrono::duration<double> duration = stop - start;
    std::cout << "Time taken by dpcpp_dijkstra: "
         << duration.count() << " seconds" << std::endl;

    /* Writing result to a txt file */
    write_arr_to_txt(gr.d, gr.V, output);

    /* Getting correct array to compare */ 
    // float* corr_res = new float[gr.V];
    // if (read_arr_from_bin(corr_res, gr.V, corr_bin)) {
    //     return 1;
    // }
    // for (int i = 0; i < gr.V; i++) {
    //     if (corr_res[i] != gr.d[i]) {
    //         std::cout << "Wrong dostance to vertex " << i+1 << std::endl;
    //         return 1;
    //     }
    // }
    std::cout << "Success!" << std::endl;
    free(gr.d, q);
    return 0;
}