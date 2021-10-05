#include "InfGraph.h"

InfGraph::InfGraph(string graph_file, int node_cnt, int edge_cnt, int seed_size, string model)
:Graph(graph_file, node_cnt, edge_cnt, seed_size, model){
    hyperG.clear();
    for(int i=0; i<n; i++)
        hyperG.push_back(vector<int>());
    for(int i=0; i<12; i++)
        sfmt_init_gen_rand(&sfmtSeed, i+1234);
}

void InfGraph::BuildHypergraphR(int64 R){
    hyperId=R;
    //for(int i=0; i<n; i++)
        //hyperG[i].clear();
    hyperG.clear();
    for(int i=0; i<n; i++)
        hyperG.push_back(vector<int>());
    hyperGT.clear();
    while((int)hyperGT.size() <= R)
        hyperGT.push_back( vector<int>() );

    for(int i=0; i<R; i++){
        BuildHypergraphNode(sfmt_genrand_uint32(&sfmtSeed)%n, i, true);
    }
    int totAddedElement=0;
    for(int i=0; i<R; i++){
        for(int t:hyperGT[i])
        {
            hyperG[t].push_back(i);
            //hyperG.addElement(t, i);
            totAddedElement++;
        }
    }
}

int InfGraph::BuildHypergraphNode(int uStart, int hyperiiid, bool addHyperEdge){
    int n_visit_edge=1;
    if(addHyperEdge){
        hyperGT[hyperiiid].push_back(uStart);
    }

    int n_visit_mark=0;

    //hyperiiid ++;
    q.clear();
    q.push_back(uStart);
    visit_mark[n_visit_mark++]=uStart;
    visit[uStart]=true;
    while(!q.empty()) {
        int expand=q.front();
        q.pop_front();
        if(influModel==IC){
            int i=expand;
            for(int j=0; j<(int)gT[i].size(); j++){
                //int u=expand;
                int v=gT[i][j];
                n_visit_edge++;
                double randDouble=double(sfmt_genrand_uint32(&sfmtSeed))/double(RAND_MAX)/2;
                if(randDouble > probT[i][j])
                    continue;
                if(visit[v])
                    continue;
                if(!visit[v])
                {
                    visit_mark[n_visit_mark++]=v;
                    visit[v]=true;
                }
                q.push_back(v);
                //#pragma omp  critical
                //if(0)
                if(addHyperEdge)
                {
                    //hyperG[v].push_back(hyperiiid);
                    hyperGT[hyperiiid].push_back(v);
                }
            }
        }
        else if(influModel==LT){
            if(gT[expand].size()==0)
                continue;
            n_visit_edge+=gT[expand].size();
            double randDouble=double(sfmt_genrand_uint32(&sfmtSeed))/double(RAND_MAX)/2;
            for(int i=0; i<(int)gT[expand].size(); i++){
                randDouble -= probT[expand][i];
                if(randDouble>0)
                    continue;
                //int u=expand;
                int v=gT[expand][i];

                if(visit[v])
                    break;
                if(!visit[v]){
                    visit_mark[n_visit_mark++]=v;
                    visit[v]=true;
                }
                q.push_back(v);
                if(addHyperEdge){
                    hyperGT[hyperiiid].push_back(v);
                }
                break;
            }
        }
    }
    for(int i=0; i<n_visit_mark; i++)
        visit[visit_mark[i]]=false;
    return n_visit_edge;
}

void InfGraph::BuildSeedSet() {
    vector< int > degree;
    vector< int> visit_local(hyperGT.size());
    //sort(ALL(degree));
    //reverse(ALL(degree));
    seedSet.clear();
    for(int i=0; i<n; i++)
    {
        degree.push_back( hyperG[i].size() );
        //degree.push_back( hyperG.size(i) );
    }
    for(int i=0; i<k; i++){
        int id=max_element(degree.begin(), degree.end())-degree.begin();
        seedSet.insert(id);
        degree[id]=0;
        for(int t:hyperG[id]){
            if(!visit_local[t]){
                visit_local[t]=true;
                for(int item:hyperGT[t]){
                    degree[item]--;
                }
            }
        }
    }
}

double InfGraph::InfluenceHyperGraph(){
    set<int> s;
    for(auto t:seedSet){
        for(auto tt:hyperG[t]){
            s.insert(tt);
        }
    }
    double inf=(double)n*s.size()/hyperId;
    return inf;
}
