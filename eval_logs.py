import numpy as np

def read_userlogs(fn):
    ret = []
    ret2 = []
    with open(fn) as f:
        for line in f:
            line = line.strip().split("\t")
            userid = int(line[0])
            history = line[3].split(" ")[1:]
            start = int( history[0].split(":")[1] )
            goal = int( history[-1].split(":")[1] )
            time = goal - start
            ret.append([userid, history, start, goal, time])
            ret2.append(time)
    return ret, ret2

if __name__ == '__main__':
    userlogs, time1 = read_userlogs("tmp_result/userlogs.txt")
    userlogs, time2 = read_userlogs("tmp_result2/userlogs.txt")
    # print(userlogs)
    print(np.mean(time1))
    print(np.mean(time2))
    print(np.max(time1), np.max(time1) // 60, ":", np.max(time1) % 60)
    print(np.max(time2), np.max(time2) // 60, ":", np.max(time2) % 60)
