#ifndef GRAPH_H
#define GRAPH_H
#include "sfmt/SFMT.h"
#include <iostream>
#include <set>
#include <list>
#include <sstream>
#include <cmath>
#include <queue>
#include <fstream>
#include <string>
#include <cstdio>
#include <functional>
#include <algorithm>
#include <climits>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <map>
#include <deque>

typedef char int8;
typedef unsigned char uint8;
typedef long long int64;
typedef unsigned long long uint64;
typedef double (*pf)(int,int);

using namespace std;

class Graph{
    public:
        // Attributes
        int n, m, k;
        vector<int> inDeg;
        vector<vector<int> > gT;
        vector<vector<double> > probT;
        vector<bool> visit;
        vector<int> visit_mark;
        enum InfluModel {IC, LT};
        InfluModel influModel;
        vector<bool> hasnode;

        Graph(string graph_file, int node_cnt, int edge_cnt, int seed_size, string model);
        void readGraph(string graph_file);
        void add_edge(int a, int b, double p);

};

#endif
