#include <limits>
#include <algorithm>
#include "dpcpp_dijkstra.h"

static const int thr = 10;
using namespace sycl;

//#define DEBUG

using minloc = sycl::ext::oneapi::minimum<pair>;

int dpcpp_dijkstra(crsGraph &gr, int s, queue &q) {

    // Unified Shared Memory Allocation enables data access on host and device
    int *Visited = malloc_shared<int>(gr.V, q);
    int *inQueue = malloc_shared<int>(gr.V, q);
    int *qsize = malloc_shared<int>(1, q);
    int *delta = malloc_shared<int>(1, q);
    int *rem = malloc_shared<int>(1, q);
    int *u = malloc_shared<int>(1, q);
    pair* glob_min = malloc_shared<pair>(1, q);
    float min;
    
    #ifdef DEBUG
        std::cout << "Number of threads: " << thr << std::endl;
    #endif // DEBUG

    // Initialization
    q.parallel_for(range<1>(gr.V), [=] (id<1> ind){
        gr.d[ind] = std::numeric_limits<float>::infinity();
        Visited[ind] = 0;
        inQueue[ind] = 0;
    }).wait();

    pair identity = {
      std::numeric_limits<float>::infinity(), -1 
    };
    *glob_min = identity;

    // reductions
    auto min_red = sycl::reduction(glob_min, identity, minloc());
    auto sum_red = sycl::reduction(qsize, sycl::ext::oneapi::plus<int>());

    inQueue[s] = 1;
    qsize[0] = 1;
    gr.d[s] = 0;

    #ifdef DEBUG
        std::cout << "Starting main loop" << std::endl;
    #endif // DEBUG

    // main loop
    while(qsize[0]) {

        #ifdef DEBUG
        if (gr.V <= 20) {
            std::cout << "Queue: ";
            for(int i = 0; i < gr.V; i++) {
                if(inQueue[i] == 1) {
                    std::cout << i << " ";
                }
            }
            std::cout << "Queue size: " << qsize[0] << std::endl;
            std::cout << std::endl;

            std::cout << "d: ";
            for(int i = 0; i < gr.V; i++) {
                std::cout << gr.d[i] << " ";
            }
            std::cout << std::endl;
        }
        #endif // DEBUG

        
        *delta = gr.V / thr;
        *rem = gr.V % thr;

        // Searching for vertex in queue with minimum distance
        q.submit([&](handler& h) {
            h.parallel_for(range<1>(thr), min_red, [=] (id<1> ind, auto& glob_min) {
                pair loc_min = {std::numeric_limits<float>::infinity(), -1};
                int start, fin;
                if (ind < (*rem)) {
                    start = ind * (*delta + 1);
                    fin = start + (*delta) + 1;
                } else {
                    start = ind * (*delta) + (*rem);
                    fin = start + (*delta);
                }
                for(int i = start; i < fin; i++) {
                    if(inQueue[i] == 1 && gr.d[i] < loc_min.val) {
                        loc_min.val  = gr.d[i];
                        loc_min.vertex = i;
                    }
                }
                glob_min.combine(loc_min);
            });
        }).wait();

        *u = glob_min -> vertex;
        *glob_min = identity;

        #ifdef DEBUG
            std::cout<< "Min vert: " << *u << std::endl;
            std::cout << std::endl;
        #endif // DEBUG

        qsize[0]--;
        inQueue[*u] = 0;
        Visited[*u] = 1;

        *delta = (gr.Xadj[(*u) + 1] - gr.Xadj[*u]) / thr;
        *rem = (gr.Xadj[(*u) + 1] - gr.Xadj[*u]) % thr;

        // relaxation
        q.submit([&](handler& h) {
            h.parallel_for(range<1>(thr), sum_red, [=] (id<1> ind, auto& qsize) {
                int loc_qsize = 0;
                int start, fin;
                if (ind < (*rem)) {
                    start = gr.Xadj[*u] + ind * (*delta + 1);
                    fin = start + (*delta) + 1;
                } else {
                    start = gr.Xadj[*u] + ind * (*delta) + (*rem);
                    fin = start + (*delta);
                }
                for (int i = start; i < fin; i++) {
                    int z = gr.Adjncy[i];
                    if (gr.d[z] > gr.d[*u] + gr.Eweights[i]) {
                        gr.d[z] = gr.d[*u] + gr.Eweights[i];
                    }
                    if(!inQueue[z] && !Visited[z]) {
                        inQueue[z] = 1;
                        loc_qsize++;
                    }
                }
                qsize.combine(loc_qsize);
            });
        }).wait();
    }

    // Print Output
    if (gr.V < 15) {
        for(int i = 0; i < gr.V; i++) {
            std::cout << gr.d[i] << std::endl;
        }
    }

    free(Visited, q);
    free(inQueue, q);
    free(qsize, q);
    free(delta, q);
    free(rem, q);
    free(u, q);
    free(glob_min, q);
    return 0;
}