//
//  graphio.h
//  GraphIO
//
//  Created by Andrew Lebedev on 29.09.2020.
//

#ifndef graphio_h
#define graphio_h
extern "C" {
    #include "mmio.h"
}
#include <CL/sycl.hpp>


typedef struct  {
    int vertex;
    float val;
} edge;

typedef struct  {
    int* Adjncy;
    int* Xadj;
    float* Eweights;
    float* d;
    MM_typecode matcode;
    int V;
    int nz;
} crsGraph;


int init_graph(crsGraph* gr);
int free_graph_pointers(crsGraph* gr, sycl::queue &q);
int read_mtx_to_crs(crsGraph* gr, const char* filename, sycl::queue &q);
int read_gr_to_crs(crsGraph* gr, const char* filename, sycl::queue &q);
int write_crs_to_mtx(crsGraph* gr, const char* filename);
int read_arr_from_bin(float* arr, int size, const char* filename);
int write_arr_to_bin(float* arr, int size, const char* filename);
int write_arr_to_txt(float* arr, int size, const char* filename);

#endif /* graphio_h */
