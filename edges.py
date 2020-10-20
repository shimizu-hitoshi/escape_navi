import numpy as np
import os, re
import networkx as nx

class Edge():
    # def __init__(self, degree=12):
    def __init__(self, datadir=None):
        if datadir is None:
            # datadir = os.path.dirname(__file__) + "/data"
            datadir = os.path.dirname(__file__) + "/../mkUserlist/data/kawaramachi"
        self.dict_edge, self.dist, self.width, self.G = self.read_graph("%s/graph.twd"%datadir)
        # self.observed_edge  = self._read_graph("%s/observe_edge_%d.txt"%(datadir,degree))
        self.observed_edge  = self._read_graph("%s/observe_edge.txt"%datadir)
        self.dict_edge_swap = {v: k for k, v in self.dict_edge.items()}
        # self.dict_graph = self.read_graph("data/graph.twd")
        # print(self.dict_graph)
        self.num_obsv_edge = len(self.observed_edge)
        # self.observed_goal  = {767:0, 768:1, 769:2, 770:3, 771:4, 772:5} # extract from goallist.txt (future work)
        self.observed_goal  = self.read_goalids("%s/goalids.txt"%datadir) 
        self.num_obsv_goal = len(self.observed_goal)
        self.goal_capa = self.read_goallist("%s/goallist.txt"%datadir)
        self.DistanceMatrix = self.mk_DistanceMatrix()
        # print("goal_capa",self.goal_capa)
        if os.path.exists( "%s/points.txt"%datadir ):
            self.POINT = self.read_point("%s/points.txt"%datadir)
        if os.path.exists( "%s/curve.txt"%datadir ):
            self.CURVE = self.read_curve("%s/curve.txt"%datadir)
        # sys.exit()

        # 観測対象道路の長さと幅を使いたければ，ここをアンコメント
        # self.obsv_dist  = np.zeros(self.num_edges) # 道路長
        # self.obsv_width = np.zeros(self.num_edges) # 道路幅
        # # for e, idx in self.edges.dict_edge.items():
        # for e, idx in self.observed_edge.items():
        #     fr, to = e
        #     self.obsv_dist[idx]  = self.edges.dist[fr, to]
        #     self.obsv_width[idx] = self.edges.width[fr, to]

    def nodeid2goalid(self, nodeid):
        goalid = self.observed_goal[nodeid]
        return goalid

    def mk_DistanceMatrix(self):
        ret = np.zeros((len(self.observed_goal), len(self.observed_goal)))
        for m1, fr in enumerate(sorted( self.observed_goal)):
            for m2, to in enumerate(sorted(self.observed_goal)):
                tmp = nx.shortest_path_length(self.G, fr, to, weight='weight')
                ret[m1,m2] = tmp
                ret[m2,m1] = tmp
        # print(ret)
        return ret

    def get_edge_idx(fr, to):
        return self.dict_edge[(fr, to)]

    def get_edge(idx):
        return self.dict_edge_swap[idx]

    def read_edge_log(self, ifn):
        # ret = np.zeros(len(self.dict_graph))
        ret = np.zeros(len(self.dict_edge))
        with open(ifn) as fp:
            for line in fp:
                line = line.strip()
                line = line.split()
                fr, to, val = map(int, line)
                # idx = self.dict_graph[(fr, to)]
                idx = self.dict_edge[(fr, to)]
                # ret[idx] = val / (self.width[(fr, to)] * self.dist[(fr, to)])
                ret[idx] = val
            return ret
    def _read_graph(self, ifn):
        ret = {}
        with open(ifn) as f:
            lines = f.readlines()
            for idx, l in enumerate(lines):
                fr, to = list(map(int, l.split(" ")))
                ret[(fr, to)] = idx
        return ret
    def read_graph(self, ifn):
        G = nx.DiGraph()
        ret    = {}
        dists  = {}
        widths = {}
        with open(ifn) as fp:
            idx = 0
            val = 0
            next(fp)
            for line in fp:
                idx += 1
                line = line.strip()
                line = line.split()
                line = line[1:]
                tmps = [int( l.split(":")[0]) for l in line if ":" in l ]
                attrs = [(float(l.split(":")[1]), float(l.split(":")[2])) for l in line if ":" in l ]
                for tmp, attr in zip(tmps, attrs):
                    width = attr[0]
                    dist = attr[1]
                    ret[idx, tmp]    = val
                    dists[idx, tmp]  = dist
                    widths[idx, tmp] = width
                    # print(val,dist,width)
                    G.add_edge(idx, tmp, idx=val, weight=dist, width=width)
                    val += 1
        return ret, dists, widths, G

    def read_goallist(self, ifn):
        # assume ノードID = ゴールID
        # 同じゴールIDに複数ノードを含めるなら，要修正
        # 1stepでの流入制限throughputは，とりあえず無視
        goals = list( self.observed_goal.keys() )
        # ret    = {}
        ret    = []
        with open(ifn) as fp:
            for line in fp:
                line = line.strip().split("\t")
                id = int( line[0] )
                if id not in self.observed_goal:
                    continue
                capa = int( line[1] )
                ret.append(capa)
                # ret[id] = capa
                # 1stepでの流入制限throughputを考慮するなら，ここを使う（未：単体試験）
                # nodeids = [int( l.split(":")[0]) for l in line[2].split(" ") if ":" in l ]
                # throughputs = [int( l.split(":")[1]) for l in line[2].split(" ") if ":" in l ]
                # ret[id] = []
                # for nodeid, throughput in zip(nodeids, throughputs):
                #     ret[id].append((nodeid, capa))
        return np.array( ret )

    def read_goalids(self, ifn):
        ret = {}
        with open(ifn) as f:
            lines = f.readlines()
            for idx, l in enumerate(lines):
                goalid = int(l)
                ret[goalid] = idx
        return ret

    def read_curve(self, fnCurve):
        CURVE    = {}
        fp = open(fnCurve)
        for line in fp:
            line    = line.strip("\r\n")
            dat    = list( map(int, re.split("\s+", line)[:2]) )
            pt    = re.findall("\[([\d\.]+), ([\d\.]+)\]", line) # たぶんpoint
            for i in range(len(pt)):
                pt[i] = list( map(float, pt[i]) )
            if(not dat[0] in CURVE.keys()):
                CURVE[dat[0]] ={}
            CURVE[dat[0]][dat[1]]=pt
        fp.close()
        return CURVE

    def read_point(self, fn):
        ret    = {}
        fp = open(fn)
        for line in fp:
            line    = line.strip("\r\n")
            line = line.split("\t")
            nodeid    = int( line[0])
            pt    = list( map(float, line[1:] ) ) # たぶんpoint
            ret[nodeid]=pt[::-1] # なんでCURVEとPOINTで緯度経度が逆順やねん
        fp.close()
        return ret