#!/usr/bin/env python3
"""
# Directive:
#   Implement Game class for Nim, run game with ab prune alg with access for human players
# Notes:
# - Genetic Algorthm:
#   - Generate initial population
#   - Genetic Crossover
#   - Mutation
#   - Fitness Testing
# Genetic sim at 0.2 OR noncoding/fatal gene
# Fitness!=Functionality of genomic success...Even if somewhat analagous
# defining fitness:
#   1 if perfect match for query, and no match for anything else
#   Closest starting point reg:(start, end) sol(min,max) if end<(min) before category, if start>(max) end category
#   if 
#
#   Annealing plan, if temperature was present then can bundle ancestry & fitness with population
#
#   Unimplemented:
#   - Gene ripping
#   - Population telemetry
#
#
#   SA could be done with populations as well, instead of on an individual level
"""
#
# Name:Shane Hazeqluist
# Email:shazelquist@csu.fullerton.edu
#
# Date: Thursday, 3/17/2022 Time: 17:10.30
# Imports:
from sys import path
from sys import argv
from json import dumps
from datetime import datetime
import numpy as np
import random
import re


class gen_reg:
    """Contains states and operations required to generate regular expressions"""

    def __init__(
        self,
        popsize=2000,
        coding_master="0_+|^",
        temp=0,
        fresh_rate=0.2,
        regplace=None,
        sourcetext=None,
        target=None,
    ):
        """
        Initializes gen_reg(
        Arguments:              Type:           Desc:
        popsize=2000,           even integer    Size of populations
        coding_master="0_+|^",  string          Regular Expression operations
        temp=0,                 integer         Temperature for simulated annealing
        fresh_rate=0.2,         float           Proportion of population to introduce as untrained genetic samples
        regplace=None           dictionary      Replacement for characters that require escapes
        sourcetext,             string          Text to search through
        target,                 string          Text to attempt to generate
        )
        """
        if sourcetext and type(sourcetext) == type(""):  # use string source
            self.sourcetext = sourcetext
            self.sourcesize = len(sourcetext)
            self.base_set = list(set(self.__clean_source__()))  # set cleaned base_set
        elif sourcetext and type(sourcetext) == type(open()):  # use file source
            self.sourcetext = sourcetext
            self.sourcesize = len(sourcetext)
            self.base_set = list(set(self.__clean_source__()))  # set cleaned base_set
        else:
            self.sourcetext = ""
            self.sourcesize = 0
            self.base_set = []
        if target:
            self.target = target  # TODO: Remove target here in case target is not found in source text
            self.targetsize = len(target)
        else:
            self.target = target
            self.targetsize = 0
        self.popsize = popsize  # must be even currently
        self.pop_count = 0
        self.coding_master = coding_master
        self.maxtemp = temp
        self.temp = temp
        self.fresh_rate = fresh_rate
        self.reg_err = []  # container for regular expression errors
        self.time = datetime.now()  # Time to use as an identifier
        self.running = False  # Flag for telemetry collection

        # sort base character set to maintain order in debug situations
        self.base_set.sort()
        self.base_set = self.base_set

        self.target_reference = self.test_search(target)
        print("Building from the following set ", self.base_set)

        self.telem = {
            "ctemp": [],
            "fit": [],  # list of fitness
            "maxfit": [],
            "minfit": [],
            "avgfit": [],
            "dfmean": [],
            "matched": [[]],  # Number of matches of a result
            # Incremental items
            "pmutation": [0],  # point mutation
            "smutation": [0],  # swap mutation
            "infanticide": [0],  #
        }

    def __repop_telem__(self):
        """Handles some grunt telemetry work for each new population"""
        for k in ["pmutation", "smutation", "infanticide"]:  # set incremental values
            self.telem[k].append(0)
        self.telem["matched"].append([])

    def dump_telemetry(self, sourcefile):
        """Dumps telemetry to file requires name of source file"""
        sav = {
            "sourcefile": sourcefile,
            "mtemp": self.maxtemp,
            "popcount": self.popsize,
            "base_set": self.base_set,
            "coding_master": self.coding_master,
            "fresh_rate": self.fresh_rate,
            "target": self.target,
        }
        sav.update(self.telem)
        sav["matched"].pop(-1)  # remove extrenous match list
        cont = True
        while cont:
            ans = input("Do you want to save these results? y/n: ")
            if ans == "y":
                cont = False
            elif ans == "n":
                cont = False
                print("Results not saved")
                return
            else:
                print('Incorrect input "{}"'.format(ans))
        with open(
            "telemetry/telem_sav_{}_.json".format(self.time)
            .replace(" ", "+")
            .replace(":", "~"),
            "w",
            encoding="utf-8",
        ) as jsonfile:
            jsonfile.write(dumps(sav))
            print(
                'Wrote save to "{}"'.format(
                    "telemetry/telem_sav_{}_.json".format(self.time)
                    .replace(" ", "+")
                    .replace(":", "~")
                )
            )

    def dump_pop(self, pop):
        """Given population, dump to target save_file"""
        with open(
            "pop/population_{}_.json".format(self.time)
            .replace(" ", "+")
            .replace(":", "~"),
            "a",
            encoding="utf-8",
        ) as savfile:
            savfile.write("{}\n".format(dumps(pop)))
        print(
            'Wrote population datat to "pop/population_{}_.json"'.format(self.time)
            .replace(" ", "+")
            .replace(":", "~")
        )

    def __clean_source__(self, regplace=None):
        """Cleans text of characters requiring escapes (regplace:dict) of characters to escaped characters"""
        # convert items to a regex target friendly characterset IE: ' '-> '\s' and generally inserts excape characters when needed
        text = list(set(self.sourcetext))
        if not regplace:
            regplace = {
                " ": "\s",
                "<": "\<",
                ">": "\>",
                "[": "\[",
                "]": "\]",
                "(": "\(",
                ")": "\)",
                "/": "\/",
                ":": "\:",
                "◦": "\◦",
                "•": "\•",
                "-": "\-",
                "\t": "\\t",
                "\n": "\\n",
                "^": "\^",
                "“": "\“",
                "”": "\”",
                "'": "\\'",
                "’": "\’",
                '"': '\\"',
                "…": "\…",
                "▪": "\▪",
            }
        for s, r in zip(regplace.keys(), regplace.values()):
            if s in text:
                text[text.index(s)] = r

        return set(text)

    def __child_mixing__(self, split_count, dif, gleft, cleft, gright, cright):
        """Recursively performs genetic mixing for children (count, diference, chance, gleft, cleft, gright, cright)"""
        if not split_count:  # no splits left
            return gleft, cleft, gright, cright

        split = int(random.random() * (min(len(cleft), len(cright)) - 1))
        gleft = gleft.split(" ")
        gright = gright.split(" ")
        # if rip:
        #   pass
        # else:
        #   continue

        ngleft = gleft[:split] + gright[split:]
        ngright = gright[:split] + gleft[split:]

        return self.__child_mixing__(
            split_count - 1,
            dif,
            " ".join(ngleft),
            cleft[:split] + cright[split:],
            " ".join(ngright),
            cright[:split] + cleft[split:],
        )

    def __mutate__(self, coding):
        """A sub function to mutate the coding sement of an individual"""
        # mutations will only occur in the coding segment
        mtype = int(random.random() * 2)
        if mtype == 0:  # point mutation
            mindex = int(random.random() * len(coding))
            repchr = int(random.random() * len(self.coding_master))  # random.choice
            coding = (
                coding[:mindex] + self.coding_master[repchr] + coding[mindex + 1 :]
            )  # perform mutation
            self.telem["pmutation"][-1] += 1
        elif mtype == 1:  # Swp mutation
            mindex_a = int(random.random() * len(coding))
            mindex_b = int(random.random() * (len(coding) - 1))  # reduce space
            mindex_b += int(
                mindex_b < mindex_a
            )  # increment (if eq or greater) to avoid collisions
            if mindex_b < mindex_a:  # swp so a<b
                mindex_a, mindex_b = mindex_b, mindex_a
            # perform swap of indexes
            coding = (
                coding[:mindex_a]
                + coding[mindex_b]
                + coding[mindex_a + 1 : mindex_b]
                + coding[mindex_a]
                + coding[mindex_b + 1 :]
            )  # perform mutation
            # pick two different coding int
            self.telem["smutation"][-1] += 1
        else:
            input("unhandled mutation", mtype)
        return coding

    def new_target(self, target):
        """Sets new target string, returns status"""
        if self.test_search(target):
            self.target = target
            self.targetsize = len(target)
            return True
        else:
            print('Error, new target "{}" is not in sourcetext'.format(target))
            return False

    def new_source(self, sourcetext):
        """Sets a new source and returns status"""
        result = [i.span() for i in re.finditer(self.target, sourcetext)]
        found = False
        if result:
            found = True
        if found and sourcetext and type(sourcetext) == type(""):  # use string source
            self.sourcetext = sourcetext
            self.sourcesize = len(sourcetext)
            b_set = list(set(self.__clean_source__()))  # set cleaned base_set
        elif (
            found and sourcetext and type(sourcetext) == type(open())
        ):  # use file source
            self.sourcetext = sourcetext
            self.sourcesize = len(sourcetext)
            b_set = set(self.__clean_source__())  # set cleaned base_set
            b_set = list(bset + self.base_set)  # inclusivly add new set of characters
            b_set.sort()
            self.base_set = set(b_set)
        else:
            print(
                'Error, new sourcetext does not contain target "{}"'.format(self.target)
            )
            return False
        return True

    def generate_gene(self, targetsize=None):  # target length
        """Generate initial population genetic material"""
        if not targetsize:
            targetsize = self.targetsize
        gene = ""
        coding = ""
        minlen = 3
        maxlen = targetsize + (targetsize - minlen)
        gcount = [0.5, 0.7, 0.95]
        count = 3
        genelen = minlen + int(
            random.random() * (maxlen - minlen)
        )  # randomly choose gene length between 3,15

        for i in range(0, genelen):
            rate = random.random()
            count = sum(
                [rate >= gc for gc in gcount]
            )  # obtain gene count IE bracket size
            bracket = ""
            for i in range(0, 1 + count):
                bracket += self.base_set[int(random.random() * (len(self.base_set)))]

            gene += " {}".format(bracket)
            coding += self.coding_master[
                int(random.random() * (len(self.coding_master)))
            ]

        # remove leading space
        return gene[1:], coding

    def generate_population(self, size=None, target=None):  # target length
        """Generates a population (size, target) for size and target length"""
        if not size:
            return []
        if not target:
            target = self.targetsize
        print("Generating population s=", size)
        population = []
        for i in range(0, size):
            a, b = self.generate_gene()
            population.append([a, b])
        return population

    def generate_children(self, left, right, number=2, temp=10, chance=0.2):
        """Generates children from parents (left,right,number,temp (unused),chance)"""
        gleft, cleft = left
        gright, cright = right
        dif = len(cleft) - len(cright)
        children = []
        while len(children) < number:
            gleft, cleft = left
            gright, cright = right
            # number of splits
            split_count = 1 + (random.random() > 0.75) * 1
            ngleft, ncleft, ngright, ncright = self.__child_mixing__(
                split_count, dif, gleft, cleft, gright, cright
            )
            if random.random() < chance:  # mutation occured
                ncleft = self.__mutate__(ncleft)
            if random.random() < chance:  # mutation occured
                ncright = self.__mutate__(ncright)

            childl = [ngleft, ncleft]
            childr = [ngright, ncright]
            children += [childl, childr]
        return children

    def log_fit(self, fit, matches, adjusted, maxfit):
        """dump_results to a file"""
        filename = "fitness_scores_{}_.json".format(datetime.now())
        data = {
            "param": [],
            "fit": [],
            "adjusted": [],
            "avg_fit": [],
            "match_count": [],
            "max_fit": [],
        }

    def fitness(self, sample, fullrange):
        """Given a sample match, determine fitness as a float (0,1)"""
        # sample,reference are the
        # comparing [(index0, index1)]
        # reference:{---------[++++++]-----------}
        matchpoint = 0.0001  # base value for having a match (also provides room to eliminate large matches of bad sectors)
        if not sample and False:  # no sample given
            print("no sample")
            return 0.0
        # print([i[1]-i[0] for i in reference])
        full = fullrange  # (0,30)    # full range of text
        ref_size = sum([i[1] - i[0] for i in self.target_reference])
        score = 0  # matchpoint-matchpoint*(not sample)
        history = []
        for targ in self.target_reference:
            for samp in sample:
                rpoint = 0.0  # round point
                rpenal = 0.0
                past = False
                if samp[0] <= targ[0] and samp[1] >= targ[0]:  #    (--[++)?
                    past += 1
                    rpoint += samp[1] - targ[0]
                    rpenal += targ[0] - samp[0]
                if samp[1] >= targ[1] and samp[0] <= targ[1]:  #    (++]--)?
                    past += 1
                    rpoint += targ[1] - samp[0]
                    rpenal += samp[1] - targ[1]
                if past == 2 and samp[0] <= targ[1]:  # [ (++) ]?
                    rpoint = targ[1] - targ[0]  # /=2
                if past == 1:  # []
                    pass
                if not past and samp[0] >= targ[0] and samp[1] <= targ[1]:  # ( [] )
                    rpoint += samp[1] - samp[0]
                elif not rpenal and not past:  #
                    rpenal += samp[1] - samp[0]
                else:
                    pass
                history.append(
                    (rpoint) / (ref_size) - rpenal / (full[1] - ref_size)
                )  # (Bonus/size)/match penalty
        history.sort(reverse=True)
        history = history[: len(self.target_reference)]  # top N items

        score = sum(history)  # /len(history)
        score = matchpoint + (1 - matchpoint) * score / (
            1 + abs(len(sample) - len(self.target_reference))
        )
        if score < 0:
            score = matchpoint - abs(score) * matchpoint
        return score

    def convert_to_reg(self, gene, coding):
        """Given gene and coding, converts and returns regular expression"""
        # 'abc deef eaefg efgssgi sfwrfwe wcecww'
        # '+0|+^|+'
        # [^...] negated set
        # ([][]) added set
        # ([])|([]) or set
        # 0, non-coding set

        # gene='a bc de fg hi jk lm no pq rst uv w x yz'
        # coding='_^|+00+|^_'
        # avgwordlen=5,13

        gene = gene.split(" ")
        genelen = len(gene)

        def idn(index):
            return index % genelen

        gi = 0
        result = "("
        for c in coding:
            if c == "0":
                pass  # skip gene
            elif c == "+":
                # end parenthesis
                result += "[{}])(".format(gene[idn(gi)])
            elif c == "|":
                # or parenthesis
                result += "[{}])|(".format(gene[idn(gi)])
            elif c == "^":
                # not set
                result += "[^{}]".format(gene[idn(gi)])
            elif c == "_":
                # continue word
                result += "[{}]".format(gene[idn(gi)])
            gi += 1
        result += ")"
        if result[-3:] == "|()":
            return result[:-3]
        if result[-2:] == "()":
            return result[:-2]
        return result

    def test_search(self, reg):
        """Uses given regular expression to search for results, returns ranges"""
        try:
            result = [i.span() for i in re.finditer(reg, self.sourcetext)]
        except Exception as e:
            if e not in self.reg_err:
                self.reg_err.append(e)
            print("ERROR FOUND", e, reg)
            if self.running:
                self.telem["matched"][-1].append(0)
            return
        if self.running:
            # input('matched: {}'.format(self.telem['matched']))
            self.telem["matched"][-1].append(len(result))
        return result

    def run_population(self, pop, modifier=1.0001):
        """Given a population, runs parameters to produce new population"""
        if not self.target or not self.sourcetext:
            print(
                'Invalid start status, target = "{}" len(source) = {}'.format(
                    self.target, len(self.sourcetext)
                )
            )
            exit(1)
        # sample reference, fullname
        print("Running population")
        self.running = True
        # modifier = 1.0001
        if type(pop) != type(
            dict()
        ):  # first run will need to calculate population as well
            # TODO:Need to modify to extract number of matches here or piecemeal in test_search w/run flag
            fit = [
                self.fitness(
                    self.test_search(self.convert_to_reg(sample[0], sample[1])),
                    (0, self.sourcesize),
                )
                for sample in pop
            ]
            self.telem["matched"].append([])
        else:
            fit = pop["fitness"]
            pop = pop["population"]
        # newfit=[]

        mean = sum(fit) / len(fit)
        df_mean = [abs(i - mean) for i in fit]

        # adjust fitness for cumsum
        adjusted = [(i + abs(i - mean) / modifier) for i in fit]
        # normalize the adjusted fitness

        adjusted = [i / sum(adjusted) for i in adjusted]
        fresh_pop = int(len(pop) * self.fresh_rate)

        # generate a new sub-poplation to add diversity (and prevent bottleneck)
        newpop = self.generate_population(
            fresh_pop, self.target_reference[-1][1] - self.target_reference[0][0]
        )
        newfit = [
            self.fitness(
                self.test_search(self.convert_to_reg(sample[0], sample[1])),
                (0, self.sourcesize),
            )
            for sample in newpop
        ]
        refnum = list(range(0, len(pop)))
        maxfit_t = max(fit)
        avgfit_t = sum(fit) / len(fit)
        dfmean_t = sum(df_mean) / len(fit)

        self.telem["fit"].append(fit)
        self.telem["ctemp"].append(self.temp)

        winner_i = fit.index(maxfit_t)
        winner = pop[winner_i]
        print(
            'population avg fitness {}, Max Fitness {} AVG df from Mean {} avg pull to leader {}\n\tWinner {} r"{}"\n'.format(
                avgfit_t,
                maxfit_t,
                dfmean_t,
                maxfit_t - avgfit_t,
                winner,
                self.convert_to_reg(*winner),
            )
        )

        self.telem["maxfit"].append(maxfit_t)
        self.telem["minfit"].append(min(fit))
        self.telem["avgfit"].append(avgfit_t)
        self.telem["dfmean"].append(dfmean_t)

        print(
            "\t\tNumber of matches by Winner:{}".format(
                self.telem["matched"][-2][winner_i]
            )
        )
        while len(newpop) < len(pop):
            # Randomly choose two parents, based on the adjusted fitness
            parent_ai, parent_bi = np.random.choice(refnum, 2, p=adjusted)  # ,None
            parent_a = pop[parent_ai]
            parent_b = pop[parent_bi]

            newchildren = self.generate_children(parent_a, parent_b, 1)

            # calculate child fitness
            ca_fit = self.fitness(
                self.test_search(
                    self.convert_to_reg(newchildren[0][0], newchildren[0][1])
                ),
                (0, self.sourcesize),
            )
            cb_fit = self.fitness(
                self.test_search(
                    self.convert_to_reg(newchildren[1][0], newchildren[1][1])
                ),
                (0, self.sourcesize),
            )
            if not self.temp:  # no temp or temp is zero
                newfit += [ca_fit, cb_fit]
                newpop += newchildren
            else:  # have temperature, perform simulated annealing
                # Simulated annealing
                delta = ca_fit - fit[parent_ai]  # parent-child
                if not delta > 0:  # if temp, perform self annealing
                    pass
                    delta = 0.0
                    if random.random() < np.exp(delta / self.temp):
                        # use less fit child
                        newfit.append(ca_fit)
                        newpop.append(newchildren[0])
                    else:
                        # use fitter parent
                        newfit.append(fit[parent_ai])
                        newpop.append(parent_a)
                        self.telem["infanticide"][-1] += 1
                        # save parent match number instead
                        self.telem["matched"][-1][-2] = self.telem["matched"][-2][
                            parent_ai
                        ]
                else:  # use child as basecase
                    newfit.append(ca_fit)
                    newpop.append(newchildren[0])
                delta = ca_fit - fit[parent_bi]  # parent-child
                if not delta > 0:  # if temp, perform self annealing
                    pass
                    delta = 0.0
                    if random.random() < np.exp(delta / self.temp):
                        # use less fit child
                        newfit.append(cb_fit)
                        newpop.append(newchildren[1])
                    else:
                        # use fitter parent
                        newfit.append(fit[parent_bi])
                        newpop.append(parent_b)
                        # save parent match number instead
                        self.telem["matched"][-1][-1] = self.telem["matched"][-2][
                            parent_bi
                        ]
                        self.telem["infanticide"][-1] += 1
                else:  # use child as basecase
                    newfit.append(ca_fit)
                    newpop.append(newchildren[0])
            # eo Simulated annealing

        self.temp -= 1
        self.pop_count += 1
        self.__repop_telem__()
        self.running = False
        return {"population": newpop, "fitness": newfit}, maxfit_t

    def get_random(self, index=None):  # Depreciated
        """
        Generate a rendom number from a file as a replacement for random.random()
        Useful for debugging to generate the same results for each run.
        NOTICE: Depreciated
        """
        global rand_index
        global sourcefile
        if not sourcefile:
            sourcefile = open("rand.csv", "r")
        if RANDOM:
            return random.random()
        if index:
            rand_index = index
        num = -1
        if rand_index >= maxind:
            rand_index = 0
            print("\n\n\nEOF\n\n\n")

        sourcefile.seek(22 * rand_index)
        num = sourcefile.read(18)
        rand_index += 1

        return float(num)


def generate_num():  # Depreciated
    """Generate file with random numbers 0-1 seperated by ", " DEPRECIATED"""
    f = open("rand.csv", "w")
    i = 0
    while i < maxind:
        f.write("{:.18f}, ".format(random.random()))
        print("{} entries {}".format("\x1b[K1", i), end="\r")
        i += 1
        # input()


def obtain_sample(fname="sample.txt"):
    """split file into set of characters"""
    sample = ""
    with open(fname, encoding="utf-8") as sfile:
        sample = sfile.read()
    if not sample:
        print("no sample obtained")  # stderr
        exit(1)
    return set(sample)


def main():
    """main in gen_reg"""
    if len(argv) == 5:
        print(argv)
        print("Source text", argv[-4])
        print("Target", argv[-3])
        filename = argv[-4]
        target = argv[-3]
        temp = int(argv[-2])
        operations = argv[-1]
        repeats = (temp > 0) * temp + (temp == 0) * 100
    else:
        target = "influence"
        filename = "README.md"
        temp = 10
        repeats = 100
        operations = "0_+^"
        print('No arguments given, "sourcetext" "target" "Temp" "operations"')

    text = open(filename, "r", encoding="utf-8").read()
    prop_doc = gen_reg(
        sourcetext=text,
        target=target,
        popsize=2000,
        coding_master=operations,
        fresh_rate=0.3,
        temp=temp,
    )

    population_n = 0
    cont = True
    pop = prop_doc.generate_population(size=prop_doc.popsize)
    SA = prop_doc.temp != 0
    mf = 0.0
    while cont:
        for i in range(0, repeats):
            pop, mf = prop_doc.run_population(pop)
            population_n += 1
            if mf == 1:
                print("\n\nSOLUTION REACHED after {} populations".format(population_n))
                cont = False
                break
            if prop_doc.maxtemp and prop_doc.temp == 0 and SA:
                print("\n\nSimulated Annealing done\n\n")
                SA = False
        if "y" in input("{} populations, quit? y/n\t\a".format(population_n)):
            cont = False
    prop_doc.dump_telemetry(filename)


if __name__ == "__main__":
    main()
