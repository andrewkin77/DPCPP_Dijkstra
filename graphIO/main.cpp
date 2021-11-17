//
//  main.c
//  GraphIO
//
//  Created by Andrew Lebedev on 29.09.2020.
//

#include "graphio.h"
#include <time.h>

int main() {
    float duration;
    int i, size = 5;
    double* arr = (double*)malloc(sizeof(double) * size);
    double* b = (double*)malloc(sizeof(double) * size);
    for(i = 0; i < size; i++){
        arr[i] = i;
    }
    clock_t start, stop;
    crsGraph gr;
    init_graph(&gr);
    start = clock();
    read_mtx_to_crs(&gr, "mycielskian3.mtx");
    stop = clock();
    duration = (float)(stop - start) / CLOCKS_PER_SEC;
    printf("Duration: %f\n", duration);
    start = clock();
    write_crs_to_mtx(&gr, "write.mtx");
    stop = clock();
    printf("Duration: %f\n", duration);
    
    write_arr_to_bin(arr, size, "write.bin");
    read_arr_from_bin(b, size, "write.bin");
    for(i = 0; i < size; i++) {
        printf("%lg\n", b[i]);
    }
    write_arr_to_txt(arr, size, "write.txt");
}
