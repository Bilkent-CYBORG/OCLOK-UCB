#include "Graph.h"

Graph::Graph(string graph_file, int node_cnt, int edge_cnt, int seed_size, string model){

    k = seed_size;
    n = node_cnt;
    m = edge_cnt;

    if(model == "IC"){
        influModel = Graph::IC;
    }
    else if(model == "LT"){
        influModel = Graph::LT;
    }

    visit_mark=vector<int>(n);
    visit=vector<bool>(n);

    //init vector
    for(int i=0; i<((int)n); i++){
        gT.push_back(vector<int>());
        hasnode.push_back(false);
        probT.push_back(vector<double>());
        //hyperGT.push_back(vector<int>());
        inDeg.push_back(0);
    }

    readGraph(graph_file);
}

void Graph::add_edge(int a, int b, double p){
    probT[b].push_back(p);
    gT[b].push_back(a);
    inDeg[b]++;
}

void Graph::readGraph(string graph_file){
    FILE * fin= fopen((graph_file).c_str(), "r");
    int readCnt=0;
    for(int i=0; i<m; i++){
        readCnt ++;
        int a, b;
        double p;
        fscanf(fin, "%d%d%lf", &a, &b, &p);
        hasnode[a]=true;
        hasnode[b]=true;
        add_edge(a, b, p);
    }
    fclose(fin);
}
