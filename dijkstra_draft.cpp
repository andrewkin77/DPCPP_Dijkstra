#include <CL/sycl.hpp>
#include <limits>
#include <algorithm>
using namespace sycl;
static const int N = 5;
static const int M = 6;
static const int thr = 4;
static const int s = 0;

//#define DEBUG

int main(){
    // define queue which has default device associated for offload
    queue q;
    std::cout << "Device: " << q.get_device().get_info<info::device::name>() << std::endl;

    int _Adj[2*M] = {1, 2, 4, 0, 3, 0, 3, 1, 2, 4, 0, 3};
    int _Xadj[N+1] = {0, 3, 5, 7, 10, 12};
    double _Weights[2*M] = {5, 7, 3, 5, 2, 7, 2, 2, 3, 4, 3, 4};

    // Unified Shared Memory Allocation enables data access on host and device
    int *Adj = malloc_shared<int>(2*M, q);
    int *Xadj = malloc_shared<int>(N+1, q);
    double *Weights = malloc_shared<double>(2*M, q);
    double *d = malloc_shared<double>(N, q);
    int *Visited = malloc_shared<int>(N, q);
    int *inQueue = malloc_shared<int>(N, q);
    int *qsize = malloc_shared<int>(1, q);
    int *delta = malloc_shared<int>(1, q);
    int *rem = malloc_shared<int>(1, q);
    int *u = malloc_shared<int>(1, q);
    double min;

    // aux
    double *aux_min = malloc_shared<double>(thr, q);
    int *aux_vert = malloc_shared<int>(thr, q);
    int *aux_qsize = malloc_shared<int>(thr, q);

    // Initialization
    q.parallel_for(range<1>(N), [=] (id<1> ind){
        d[ind] = std::numeric_limits<double>::infinity();
        Visited[ind] = 0;
        inQueue[ind] = 0;
    }).wait();

    for (int i = 0; i < 2*M; i++) {
        Adj[i] = _Adj[i];
        Weights[i] = _Weights[i];
    }

    for (int i = 0; i < N+1; i++) {
        Xadj[i] = _Xadj[i];
    }

    inQueue[s] = 1;
    qsize[s] = 1;
    d[s] = 0;

    // main loop
    while(qsize[0]) {

        #ifdef DEBUG
            std::cout << "Queue: ";
            for(int i = 0; i < N; i++) {
                if(inQueue[i] == 1) {
                    std::cout << i << " ";
                }
            }
            std::cout << std::endl;

            std::cout << "d: ";
            for(int i = 0; i < N; i++) {
                std::cout << d[i] << " ";
            }
            std::cout << std::endl;        
        #endif // DEBUG

        
        *delta = N / thr;
        *rem = N % thr;

        // Searching for vertex in queue with minimum distance
        q.parallel_for(range<1>(thr), [=] (id<1> ind) {
            double loc_min = std::numeric_limits<double>::infinity();
            int loc_vert = -1;
            int start, fin;
            if (ind < (*rem)) {
                start = ind * (*delta + 1);
                fin = start + (*delta) + 1;
            } else {
                start = ind * (*delta) + (*rem);
                fin = start + (*delta);
            }
            for(int i = start; i < fin; i++) {
                if(inQueue[i] == 1 && d[i] < loc_min) {
                    loc_min = d[i];
                    loc_vert = i;
                }
            }
                
            aux_min[ind] = loc_min;
            aux_vert[ind] = loc_vert;
        }).wait();

        // reduction
        *u = aux_vert[0];
        min = aux_min[0];
        for (int i = 1; i < thr; i++) {
            if (aux_min[i] < min) {
                min = aux_min[i];
                *u = aux_vert[i];
            }
        }

        #ifdef DEBUG
            std::cout<< "Min vert: " << *u << std::endl;
            std::cout << std::endl;
        #endif // DEBUG

        qsize[0]--;
        inQueue[*u] = 0;
        Visited[*u] = 1;
        *delta = (Xadj[(*u) + 1] - Xadj[*u]) / thr;
        *rem = (Xadj[(*u) + 1] - Xadj[*u]) % thr;

        // relaxation
        q.parallel_for(range<1>(thr), [=] (id<1> ind) {
            int loc_qsize = 0;
            int start, fin;
            if (ind < (*rem)) {
                start = Xadj[*u] + ind * (*delta + 1);
                fin = start + (*delta) + 1;
            } else {
                start = Xadj[*u] + ind * (*delta) + (*rem);
                fin = start + (*delta);
            }
            for (int i = start; i < fin; i++) {
                int z = Adj[i];
                if (d[z] > d[*u] + Weights[i]) {
                    d[z] = d[*u] + Weights[i];
                }
                if(!inQueue[z] && !Visited[z]) {
                    inQueue[z] = 1;
                    loc_qsize++;
                }
            }
            aux_qsize[ind] = loc_qsize;
        }).wait();

        // reduction
        for (int i = 0; i < thr; i++) {
            qsize[0] += aux_qsize[i];
        }
    }

    // Print Output
    for(int i = 0; i < N; i++) {
        std::cout << d[i] << std::endl;
    }

    free(Adj, q);
    free(Xadj, q);
    free(Weights, q);
    free(d, q);
    free(Visited, q);
    free(inQueue, q);
    free(qsize, q);
    free(delta, q);
    free(rem, q);
    free(u, q);
    free(aux_min, q);
    free(aux_qsize, q);
    free(aux_vert, q);
    return 0;
}