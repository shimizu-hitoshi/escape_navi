import sys
from cffi import FFI
import numpy as np
sys.path.append('../')
from edges import Edge

if __name__ == '__main__':
    libsimfn = "./libsim.so"
    datadir  = "../mkUserlist/data/N80000r0i0/"
    agentfn  = "%s/agentlist.txt"%datadir
    graphfn  = "%s/graph.twd"%datadir
    goalfn   = "%s/goallist.txt"%datadir
    sim_time = 20000
    interval = 600
    max_step = int( np.ceil( sim_time / interval ))
    num_agents = 80000
    ffi = FFI()
    lib = ffi.dlopen(libsimfn)
    ffi.cdef("""
    void init(int argc, char** argv);
    int setStop(int t);
    void iterate();
    void setBombDirect( char *text);
    void setBomb( char *fn);
    int cntOnEdge(int fr, int to);
    void restart();
    void init_restart(int argc, char** argv);
    """) 

    # edges = Edge(12)
    # print("num_edges")
    # print(edges.num_edges)
    edges = Edge()
    argv = [sys.argv[0]]
    argv.extend([
        agentfn,
        graphfn,
        goalfn,
        "-o",
        "result2",
        "-l",
        "99999",
        "-e",
        str(sim_time)
        ])
    print(argv)
    tmp = []
    for a in argv:
        tmp.append(ffi.new("char []", a.encode('ascii')))
    argv = ffi.new("char *[]", tmp)
    # call simulator
    lib.init(len(argv), argv)
    # lib.setBomb("./event.txt".encode('ascii'))
    input()
    res = []
    time = 0
    traveltime = 0
    for step in range(max_step):
        print("TIME", time)
        lib.setStop(time)
        lib.iterate()
        ret = np.zeros(len(edges.observed_edge))
        for e, idx in edges.observed_edge.items():
            fr, to     = e
            ret[idx]   = lib.cntOnEdge(fr-1, to-1)
        # traveltime += np.sum(ret) * interval / num_agents
        res.append(np.sum(ret) * interval / num_agents)
        time += interval
    print(traveltime)
    print(res)
    print(np.sum(res))
    # with open("result/travel_free_time.json", "w") as f:
    #     json.dump(res, f)
