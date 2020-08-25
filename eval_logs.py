#!/usr/bin/env python

import numpy as np
import glob

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

def read_log(fn):
    with open(fn) as f:
        ret = len(f.readlines())
    return ret

def sum_logs(logdir):
    ret = 0
    # fns = glob.glob("%s/log*.txt"%logdir)
    # for fn in fns:
    for i in range(30):
        fn = "{:}/log{:0=6}.txt".format(logdir, (i+1)*300)
        print(fn)
        if "station" in fn:
            continue
        ret += read_log(fn)
    return ret

def show_stats(fn):
    userlogs, time1 = read_userlogs(fn)
    print(np.mean(time1))
    print(np.max(time1), np.max(time1) // 60, ":", np.max(time1) % 60)

if __name__ == '__main__':
    # show_stats("tmp_result2/userlogs.txt")
    print(sum_logs("tmp_result2") * 300 / 5000)
    # n = read_log("tmp_result2/log000010.txt")
    # print(n)

    # userlogs, time1 = read_userlogs("tmp_result/userlogs.txt")
    # userlogs, time2 = read_userlogs("tmp_result2/userlogs.txt")
    # # print(userlogs)
    # print(np.mean(time1))
    # print(np.mean(time2))
    # print(np.max(time1), np.max(time1) // 60, ":", np.max(time1) % 60)
    # print(np.max(time2), np.max(time2) // 60, ":", np.max(time2) % 60)
