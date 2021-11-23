#ifndef dpcpp_dijkstra_h
#define dpcpp_dijkstra_h

#include "graphio.h"

struct pair {
    bool operator<(const pair& o) const {
    return val < o.val;
    }
    float val;
    int vertex;
};

int dpcpp_dijkstra(crsGraph &gr, int s, const int thr, sycl::queue &q);

#endif /* dpcpp_dijkstra.h */
