from cffi import FFI
import sys
import numpy as np

ffi = FFI()
lib = ffi.dlopen("libsim.so")
ffi.cdef("""
void    init(int argc, char** argv);
int	    setStop(int t);
void	iterate();
int		cntDest(int node, double radius);
int		cntSrc(int node, double radius);
void	setBomb( char *fn);
int		cntOnEdge(int fr, int to);
void    setBombDirect(char *text);
void    restart();
void    save_ulog(char *fn);
void    init_restart(int argc, char** argv);
int	    goalAgentCnt(int stime, int etime, int cnt);
int		goalAgent(int stime, int etime,int n,  int result[][3]);
""")
argv    = [sys.argv[0]]
print(argv)
argv.extend([
"../userlist_new_add_via2.txt",
"../graph_new_add.twd",
"../stationlist_new_add2.txt",
"-o",
"result",
"-l",
"300"
])

argv2    = [sys.argv[0]]
print(argv)
argv.extend([
"../userlist_new_add_via2.txt",
"../graph_new_add.twd",
"../stationlist_new_add2.txt",
"-o",
"result",
"-l",
"300"
])




tmp    = []
for a in argv:
    tmp.append(ffi.new("char[]", a))
argv    = ffi.new("char *[]", tmp)




for i in range(5):
    
    if i ==0:
        lib.init(len(argv), argv)
    #elif i % 2 == 1:
    #    argv    = [sys.argv[0]]
    #    print(argv)
    #    argv.extend([
    #    "userlist_new_add_via2.txt",
    #    "../graph_new_add.twd",
    #    "../stationlist_new_add2.txt",
    #    "-o",
    #    "result",
    #    "-l",
    #    "300"
    #    ])
    #    
    #    tmp    = []
    #    for a in argv:
    #        tmp.append(ffi.new("char[]", a))
    #    print(" ".join(argv))
    #    argv    = ffi.new("char *[]", tmp)


    #    lib.init_restart(len(argv), argv)
    #else:
    #    argv    = [sys.argv[0]]
    #    print(argv)
    #    argv.extend([
    #    "../userlist_new_add_via2.txt",
    #    "../graph_new_add.twd",
    #    "../stationlist_new_add2.txt",
    #    "-o",
    #    "result",
    #    "-l",
    #    "300"
    #    ])
    #    
    #    tmp    = []
    #    for a in argv:
    #        tmp.append(ffi.new("char[]", a))
    #    print(" ".join(argv))
    #    argv    = ffi.new("char *[]", tmp)


    #    lib.init_restart(len(argv), argv)
    
    else:
        lib.restart()


####

    if(lib.setStop(200) !=0):
        print("err")
        sys.exit()
    lib.iterate()
    print("node58 radius 300", lib.cntDest(58-1, 300.))
    print("772->150",lib.cntOnEdge(772-1, 150-1))
    lib.setBombDirect("400 set_gate 150 328 10 10");
    raw_input()
    lib.setStop(800)
    lib.iterate()
    if True:
        ##total goal number#########################
        c = lib.goalAgentCnt(1-1, 800-1, 762-1)     #startTime EndTime targetNode
        print(c)
        ###################################################
        c   = 80000 #large N
        c = lib.goalAgentCnt(1-1, 800-1, -1)
        res = ffi.new("int[%d][3]" % c)
        l = lib.goalAgent(1-1, 800-1, c+1, res)
        print(l)    #result number
        for i in range(l):
            print res[i][0]+1, res[i][1], res[i][2]+1   #agentid movingtime lastNode 
        sys.exit()
    print lib.cntDest(58-1, 300.)
    print lib.cntSrc(58, 300.)
    lib.setBomb("bomb.txt")
    raw_input()
    lib.setStop(1000)
    lib.iterate()
    raw_input()
    lib.save_ulog("result/ulog"+str(i)+".txt")
