
fn = "/home/shimizu/project/2019/RD2017revisit/data/user_enter5400_split6.txt"
ofn = "/home/shimizu/project/2019/RD2017revisit/data/agentlist.txt"
f = open(fn)
ofp = open(ofn, "w")
for line in f:
    line = line.strip()
    line = line.split()
    line.append("1") # direction
    line.append("0") # via
    out = " ".join(line)
    out += "\n"
    print(out)
    ofp.write(out)

f.close()
ofp.close()
