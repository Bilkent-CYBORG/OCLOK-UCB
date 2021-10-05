#ifndef INFGRAPH_H
#define INFGRAPH_H

#include "Graph.h"
class InfGraph:public Graph{
    public:
        vector<vector<int> > hyperG;
        vector<vector<int> > hyperGT;
        int64 hyperId = 0;
        deque<int> q;
        sfmt_t sfmtSeed;
        set<int> seedSet;
        enum ProbModel {TR, WC, TR001};
        ProbModel probModel;

        InfGraph(string graph_file, int node_cnt, int edge_cnt, int seed_size, string model);

        void BuildHypergraphR(int64 R);
        void BuildSeedSet();
        double InfluenceHyperGraph();
        int BuildHypergraphNode(int uStart, int hyperiiid, bool addHyperEdge);
};


#endif
