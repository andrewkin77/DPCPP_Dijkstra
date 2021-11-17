//
//  graphio.c
//  GraphIO
//
//  Created by Andrew Lebedew on 29.09.2020.
//

#include "graphio.h"

using namespace sycl;

int init_graph(crsGraph* gr) {
    gr -> Adjncy = nullptr;
    gr -> Xadj = nullptr;
    gr -> Eweights = nullptr;
    gr -> d = nullptr;
    return 0;
}

int free_graph_pointers(crsGraph* gr, queue &q) {
    if (!(gr -> Adjncy) || !(gr -> Xadj) || !(gr -> Eweights)) {
        printf("Graph is empty\n");
        return 1;
    }
    free(gr -> Adjncy, q);
    free(gr -> Xadj, q);
    free(gr -> Eweights, q);
    free(gr -> d, q);
    gr -> Adjncy = nullptr;
    gr -> Xadj = nullptr;
    gr -> Eweights = nullptr;
    gr -> d = nullptr;
    return 0;
}

int read_mtx_to_crs(crsGraph* gr, const char* filename, queue &q) {
    
    /* variables */
    int N, i, row, col, nz_size;
    int *edge_num, *last_el;
    float val;
    fpos_t position;
    FILE *file;
    
    /* mtx correctness check */
    if ((file = fopen(filename, "r")) == NULL) {
        printf("Cannot open file\n");
        return 1;
    }
    if (mm_read_banner(file, &(gr -> matcode))) {
        return 1;
    }
    if (mm_read_mtx_crd_size(file, &(gr -> V), &N, &(gr -> nz))) {
        return 1;
    }
    if (mm_is_complex(gr -> matcode) || mm_is_array(gr -> matcode)) {
        printf("Thsis application doesn't support %s", mm_typecode_to_str(gr -> matcode));
        return 1;
    }
    if (N != (gr -> V)) {
        printf("Is not a square matrix\n");
        return 1;
    }
    
    /* Allocating memmory to store adjacency list */
    last_el = (int*)malloc(sizeof(int) * gr -> V);
    edge_num = (int*)malloc(sizeof(int) * gr -> V);
    
    for (i = 0; i < (gr -> V); i++) {
        edge_num[i] = 0;
    }
    
    /* Saving value of nz so we can change it */
    nz_size = gr -> nz;
    
    /* Saving position in file to start reading from it later */
    fgetpos(file, &position);

    /* Reading file to count degrees of each vertex */
    for(i = 0; i < nz_size; i++) {
       fscanf(file, "%d %d %f", &row, &col, &val);
       row--;
       col--;
       if (row == col) {
           gr -> nz --;
           continue; //we don't need loops
       }
       edge_num[row]++;
       if (mm_is_symmetric(gr -> matcode)) {
           edge_num[col]++;
           gr -> nz ++;
       }
    }

    /* Checking if graph already has arrays */
    if ((gr -> Adjncy != NULL) || (gr -> Xadj != NULL) || (gr -> Eweights != NULL)) {
       free_graph_pointers(gr, q);
    }

    /* Creating CRS arrays */
    gr -> Adjncy = malloc_shared<int>(gr -> nz, q);
    gr -> Xadj = malloc_shared<int>((gr -> V) + 1, q);
    gr -> Eweights = malloc_shared<float>(gr -> nz, q);

    /* Writing data in Xadj and last_el */
    gr -> Xadj[0] = 0;
    for(i = 0; i < gr -> V; i++) {
       gr -> Xadj[i+1] = gr -> Xadj[i] + edge_num[i];
       last_el[i] = gr -> Xadj[i];
    }

    /* Setting right position */
    fsetpos(file, &position);

    /* Reading file to write it's content in crs */
    for(i = 0; i < nz_size; i++) {
       fscanf(file, "%d %d %f", &row, &col, &val);
       row--;
       col--;
       if (row == col) {
           continue; //we don't need loops
       }
       gr -> Adjncy[last_el[row]] = col;
       gr -> Eweights[last_el[row]] = val;
       last_el[row]++;
       if (mm_is_symmetric(gr -> matcode)) {
           gr -> Adjncy[last_el[col]] = row;
           gr -> Eweights[last_el[col]] = val;
           last_el[col]++;
       }
    }
    free(edge_num);
    free(last_el);
    fclose(file);
    return 0;
}

int read_gr_to_crs(crsGraph* gr, const char* filename, queue &q) {
    int i, row, col;
    int *edge_num, *last_el;
    float val;
    char sym = 'c';
    char str[101];
    fpos_t position;
    FILE *file;
    
    /* checking if we can read file */
    if ((file = fopen(filename, "r")) == NULL) {
        printf("Cannot open file\n");
        return 1;
    }

    while (sym == 'c') {
        sym = fgetc(file);
        if (sym == 'p') {
            fscanf(file, "%100s %d %d", str, &gr -> V, &gr -> nz);
            fgets(str, sizeof(str), file);
            fgetpos(file, &position);
        } else {
            fgets(str, sizeof(str), file);
        }
    }

    /* Allocating memmory to store adjacency list */
    last_el = (int*)malloc(sizeof(int) * gr -> V);
    edge_num = (int*)malloc(sizeof(int) * gr -> V);
    
    for (i = 0; i < (gr -> V); i++) {
        edge_num[i] = 0;
    }

    while ((sym = fgetc(file)) != EOF) {
        if (sym == 'a') {
            fscanf(file, "%d %d %f", &row, &col, &val);
            row--;
            col--;
            if (row == col) {
                gr -> nz --; // We don't need loops
            } else {
                edge_num[row]++;
            }
        }
        fgets(str, sizeof(str), file); // Moving to a new line
    }

    /* Checking if graph already has arrays */
    if ((gr -> Adjncy != NULL) || (gr -> Xadj != NULL) || (gr -> Eweights != NULL)) {
       free_graph_pointers(gr, q);
    }

    /* Creating CRS arrays */
    gr -> Adjncy = malloc_shared<int>(gr -> nz, q);
    gr -> Xadj = malloc_shared<int>((gr -> V) + 1, q);
    gr -> Eweights = malloc_shared<float>(gr -> nz, q);

    /* Writing data in Xadj and last_el */
    gr -> Xadj[0] = 0;
    for(i = 0; i < gr -> V; i++) {
       gr -> Xadj[i+1] = gr -> Xadj[i] + edge_num[i];
       last_el[i] = gr -> Xadj[i];
    }

    /* Setting right position */
    fsetpos(file, &position);

    /* Reading file to write it's content in crs */
    while ((sym = fgetc(file)) != EOF) {
        if (sym == 'a'){
            fscanf(file, "%d %d %f", &row, &col, &val);
            row--;
            col--;
            if (row == col) {
                fgets(str, sizeof(str), file);
                continue; //we don't need loops
            }
            gr -> Adjncy[last_el[row]] = col;
            gr -> Eweights[last_el[row]] = val;
            last_el[row]++;
            fgets(str, sizeof(str), file);
        } else {
            fgets(str, sizeof(str), file);
        }
    }

    free(edge_num);
    free(last_el);
    fclose(file);
    return 0;
}

int write_crs_to_mtx(crsGraph* gr, const char* filename) {
    int i,j;
    FILE* f;
    if ((f = fopen(filename, "w")) == NULL) {
        printf("Can't open file\n");
        return 1;
    }
    
    /* Writing banner and size in mtx */
    mm_write_banner(f, gr -> matcode);
    if(mm_is_symmetric(gr -> matcode)) {
        mm_write_mtx_crd_size(f, gr -> V, gr -> V, gr -> nz/2);
    } else {
        mm_write_mtx_crd_size(f, gr -> V, gr -> V, gr -> nz);
    }
    
    for(i = 0; i < gr -> V; i++) {
        for(j = gr -> Xadj[i]; j < gr -> Xadj[i+1]; j++) {
            if (i > gr -> Adjncy[j] || !mm_is_symmetric(gr -> matcode)) {
                fprintf(f, "%d %d %f\n", i + 1, gr -> Adjncy[j] + 1, gr -> Eweights[j]);
            }
        }
    }
    fclose(f);
    return 0;
}

int read_arr_from_bin(float* arr, int size, const char* filename) {
    int result;
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Couldn't opem file\n");
        return 1;
    }
    result = fread(arr, sizeof(float), size, file);
    fclose(file);
    if (result == size) {
        return 0;
    } else {
        printf("Reading error\n");
        return 1;
    }
}

int write_arr_to_bin(float* arr, int size, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file){
        printf("Couldn't opem file\n");
        return 1;
    }
    fwrite(arr, sizeof(float), size, file);
    fclose(file);
    return 0;
}

int write_arr_to_txt(float* arr, int size, const char* filename) {
    int i;
    FILE* file = fopen(filename, "w");
    if (!file){
        printf("Couldn't opem file\n");
        return 1;
    }
    for(i = 0; i < size; i++) {
        fprintf(file, "%f\n", arr[i]);
    }
    fclose(file);
    return 0;
}
