#!/usr/bin/env python3
"""
# Directive:
# Provide methods to graph telemetry data in order to observe the effectiveness of certain parameters.
# Notes:
# -
"""
#
# Name:Shane Hazeqluist
# Email:shazelquist@csu.fullerton.edu
#
# Date: Monday, 5/9/2022 Time: 20:57.57
# Imports:
from matplotlib import pyplot as plt
import numpy as np
from json import loads
from os import listdir


def enumfiles(path):
    files = listdir(path)
    savs = {}
    print("{} files found".format(len(files)))
    for f in files:
        with open("{}/{}".format(path, f), encoding="utf-8") as sav:
            savs[f[:-5]] = loads(sav.read())
    # print('file and keys')
    # for k in savs:
    #    if 'matched' not in savs[k]:
    #        print(k)
    # input()

    return savs


def show_pop_hist(savs):
    fig, ax = plt.subplots()
    alph = 0.2
    for s in savs:
        his = plt.hist(savs[s]["fit"][0], bins=100, alpha=alph)  # ,alpha=0.5
        # print(his,dir(his[-1]))
        for p in savs[s]["fit"][1:]:
            plt.hist(p, bins=100, alpha=alph)  # color=his[0].get_color()
    plt.show()


def graph_fits(savs, title=None, labels=None):
    """graphs savs, (savs, title)
    general purpose graph, will eventually be fitness avg+error and max
    """
    if not title:
        title = "Avg and Max fitness with std error"
    fig, ax = plt.subplots()
    max_x = [0]
    labeli = 0
    for s in savs:
        # print(s, list(savs[s].keys()))
        x = [i for i in range(0, len(savs[s]["avgfit"]))]

        if max_x[-1] < x[-1]:
            max_x = x
        y_low = [
            h - np.std(j) / len(j) ** 0.5
            for h, j in zip(savs[s]["avgfit"], savs[s]["fit"])
        ]
        y_high = [
            h + np.std(j) / len(j) ** 0.5
            for h, j in zip(savs[s]["avgfit"], savs[s]["fit"])
        ]

        # instead of using df from mean, calc std
        if not labels:
            label = "{}".format(s)
        else:
            label = labels[labeli]
            labeli += 1
        av_p = plt.plot(x, savs[s]["avgfit"], label="avg " + label)
        # av_m = plt.plot(x, [sum(i)/len(i) for i in savs[s]["matched"]], '.', color=av_p[0].get_color(), label="avg matches " + label)
        plt.fill_between(x, y_low, y_high, color=av_p[0].get_color(), alpha=0.3)
        plt.plot(
            x,
            savs[s]["maxfit"],
            linestyle="-.",
            label="max " + label,
            color=av_p[0].get_color(),
        )
    plt.legend()
    # plt.plot(max_x,[0.0001]*len(max_x),'.',color='black')
    plt.grid()
    plt.title(title)
    plt.xlabel("Generation Number")
    plt.ylabel("Fitness")

    plt.show()


def combine_s(savs, label="new"):
    ret = {"minfit": [], "maxfit": [], "avgfit": [], "fit": []}  # , "matched": []
    funq = {
        "minfit": lambda x: min(x),
        "maxfit": lambda x: max(x),
        # "matched": lambda x: sum(x) / len(x),
        "avgfit": lambda x: sum(x) / len(x),
    }
    print(
        "combining savs: {}\n^ Under Label {}".format(
            ",\n".join(list(savs.keys())), label
        )
    )
    # combine all values
    for k in savs:
        for k2 in ret:
            if len(k2) == 6:  # save as set for ???fit (that way we can update them)
                while len(ret[k2]) < len(savs[k][k2]):  # append to length
                    ret[k2].append([])
                for i in range(0, len(savs[k][k2])):
                    ret[k2][i].append(savs[k][k2][i])
                # ret[k2]+=savs[k][k2] should iterate these the other way i->sample
            else:
                while len(ret[k2]) < len(savs[k][k2]):  # append to length
                    ret[k2].append([])
                for i in range(0, len(savs[k][k2])):
                    ret[k2][i].append(savs[k][k2][i])
                ret[k2] += [savs[k][k2]]

    # merge sampled values
    for k2 in ret:
        if len(k2) == 6:
            # print('gen number? ',len(ret[k2]))
            for i in range(0, len(ret[k2])):  # do function for each
                if k2 in funq:
                    pass
                    ret[k2][i] = funq[k2](ret[k2][i])

    # for k in ret:
    #    i=ret[k]
    #    print(k)
    #    deep=1
    #    while '__iter__' in dir(i):
    #        print('{}shape size {}'.format('\t'*deep,len(i)))
    #        if not len(i):
    #            break
    #        i=i[0]
    #        deep+=1
    return {label: ret}


def filter_s(savs, conditions):
    """
    Filter results to a new dict of useful comparisons parameters: (savs, [conditions])
    Conditions are a list of functions to test
    """
    # Target and source should be the same
    ret = {}
    for k in savs:
        clear = True
        for cond in conditions:
            if not cond(savs[k]):
                clear = False
                break
        if clear:
            ret[k] = savs[k]

    return ret


def main():
    """Main in graph_results.py"""
    path = "telemetry"
    savs = enumfiles(path)
    labels = [savs[s]["coding_master"] for s in savs]
    slabel = list(set(labels))

    graph_fits(combine_s(savs, "All data combined"))
    graph_fits(savs, title="All data")

    coded_set = {}

    for l in slabel:  # seperate by coding_master label
        coded_set.update(
            combine_s(
                filter_s(
                    savs,
                    [lambda x: x["coding_master"] == l, lambda x: "0" in l],
                ),
                l,
            )
        )
        if not coded_set[l]["fit"]:
            del coded_set[l]
    print("GRAPHING 0")
    graph_fits(coded_set, "Fitness by operations w/standard error")

    for l in slabel:  # seperate by coding_master label
        coded_set.update(
            combine_s(
                filter_s(
                    savs,
                    [lambda x: x["coding_master"] == l, lambda x: "0" not in l],
                ),
                l,
            )
        )
        if not coded_set[l]["fit"]:
            del coded_set[l]
    print("GRAPHING !0")
    graph_fits(coded_set, "Fitness by operations w/standard error")

    # graph_fits(savs,labels=labels)# graph all samples individually
    # savs = filter_s(
    #    savs,
    #    [
    #        lambda x: np.array(
    #            [i in x["coding_master"] for i in "0"]
    #        ).all(),  # validate the operations avaliable
    #        #lambda x: x["fresh_rate"] !=0 ,  # check freshrate
    #        #lambda x: x["popcount"] > 2000,  # validate the length of the example
    #        #lambda x: len(x["fit"]) > 50,  # validate the length of the example
    #    ],
    # )
    # compare non-coding segments
    use_0 = filter_s(  # select all the samples using 0
        savs, [lambda x: np.array([i in x["coding_master"] for i in "0"]).all()]
    )
    not_0 = filter_s(  # select all the samples not using 0
        savs, [lambda x: np.array([i not in x["coding_master"] for i in "0"]).all()]
    )
    print("filtered to {} samples".format(len(list(savs.keys()))))
    com_0 = combine_s(use_0, '{} samples using "0"'.format(len(list(use_0.keys()))))
    com_n0 = combine_s(
        not_0, '{} samples not using "0"'.format(len(list(not_0.keys())))
    )
    # print(com)
    com_0.update(com_n0)
    graph_fits(com_0, title="Comparing non-coding results")
    # show_pop_hist(savs)
    # graph_fits(savs)


if __name__ == "__main__":
    main()
