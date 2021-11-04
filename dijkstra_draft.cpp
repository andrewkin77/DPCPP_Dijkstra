#include <CL/sycl.hpp>
#include <limits>
#include <algorithm>
using namespace sycl;
static const int N = 5;
static const int M = 6;
static const int thr = 1000;
static const int s = 0;

// #define DEBUG

struct pair {
    bool operator<(const pair& o) const {
    return val < o.val;
    }
    double val;
    int vertex;
};

using minloc = sycl::ext::oneapi::minimum<pair>;

int main(){
    // define queue which has default device associated for offload
    queue q(gpu_selector{});
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
    pair* glob_min = malloc_shared<pair>(1, q);
    double min;


    // Initialization
    q.parallel_for(range<1>(N), [=] (id<1> ind){
        d[ind] = std::numeric_limits<double>::infinity();
        Visited[ind] = 0;
        inQueue[ind] = 0;
    }).wait();

    pair identity = {
      std::numeric_limits<double>::infinity(), -1 
    };
    *glob_min = identity;

    // reductions
    auto min_red = sycl::reduction(glob_min, identity, minloc());
    auto sum_red = sycl::reduction(qsize, sycl::ext::oneapi::plus<int>());

    for (int i = 0; i < 2*M; i++) {
        Adj[i] = _Adj[i];
        Weights[i] = _Weights[i];
    }

    for (int i = 0; i < N+1; i++) {
        Xadj[i] = _Xadj[i];
    }

    inQueue[s] = 1;
    qsize[0] = 1;
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
        q.submit([&](handler& h) {
            h.parallel_for(range<1>(thr), min_red, [=] (id<1> ind, auto& glob_min) {
                pair loc_min = {std::numeric_limits<double>::infinity(), -1};
                int start, fin;
                if (ind < (*rem)) {
                    start = ind * (*delta + 1);
                    fin = start + (*delta) + 1;
                } else {
                    start = ind * (*delta) + (*rem);
                    fin = start + (*delta);
                }
                for(int i = start; i < fin; i++) {
                    if(inQueue[i] == 1 && d[i] < loc_min.val) {
                        loc_min.val  = d[i];
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
        *delta = (Xadj[(*u) + 1] - Xadj[*u]) / thr;
        *rem = (Xadj[(*u) + 1] - Xadj[*u]) % thr;

        // relaxation
        q.submit([&](handler& h) {
            h.parallel_for(range<1>(thr), sum_red, [=] (id<1> ind, auto& qsize) {
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
                qsize.combine(loc_qsize);
            });
        }).wait();
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
    free(glob_min, q);
    return 0;
}
