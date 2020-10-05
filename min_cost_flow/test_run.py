import subprocess
datadir = "/home/shimizu/project/2020/escape_navi/navi_curriculum/data"
bindir = "/home/shimizu/project/2020/escape_navi/simulator"
binfn = "%s/simulator"%bindir
# outdir = "/home/shimizu/project/2020/escape_navi/navi_curriculum/test_result"
# outdir = "./result0"
outdir = "./result1"
agentfn = "%s/agentlist.txt"%datadir
# agentfn = "%s/agentlist_new.txt"%datadir
agentfn = "./agentlist_new.txt"
goalfn = "%s/goallist.txt"%datadir
graphfn = "%s/graph.twd"%datadir
eventfn = "./event.txt"
# subprocess.call(
#     "%s %s %s %s -o %s -l 9999999 -e 9000"%(binfn, agentfn, graphfn, goalfn, outdir),
#     shell=True
# )
# 

subprocess.call(
    "%s %s %s %s -o %s -l 10 -e 9000 -B %s -S"%(binfn, agentfn, graphfn, goalfn, outdir, eventfn),
    shell=True
)
