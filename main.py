import time

import collections
import operator
from itertools import zip_longest, starmap, count, chain, islice, takewhile, accumulate, tee, dropwhile, repeat, \
    combinations, permutations, cycle, product
from functools import reduce, partial, cmp_to_key
from collections import Counter
import re
import numpy as np


def tails(xs):
    while xs:
        yield xs
        xs = xs[1:]


def day1(filename: str):
    def tonumber(s: str):
        lookup = dict(one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9)
        return int(s[0]) if s[0].isnumeric() else next((v for k, v in lookup.items() if s.startswith(k)), None)

    matches = [re.findall(r'\d', line) for line in open(filename).readlines()]
    answer1 = sum(int(f'{match[0]}{match[-1]}') for match in matches)

    lines = open(filename).readlines()
    digits_found = (filter(None, (tonumber(tail) for tail in tails(line))) for line in lines)
    numbers = (int(f'{xs[0]}{xs[-1]}') for xs in map(list, digits_found))
    answer2 = sum(numbers)

    return answer1, answer2


def day2(filename: str):
    def collect(s):
        return np.array([int(match.group(1))
                         if (match := re.search(f'(\\d+) {k}', s)) else 0
                         for k in {'red', 'green', 'blue'}])

    lines = [re.split(r':|;', line.strip()) for line in open(filename).readlines()]
    gameids = (int(header[5:]) for header, *_ in lines)
    maxs = [reduce(np.maximum, (collect(draw) for draw in draws)) for _, *draws in lines]
    part1 = sum(id for id, m in zip(gameids, maxs) if all(m <= np.array([12, 13, 14])))
    part2 = sum(reduce(operator.mul, m) for m in maxs)
    return part1, part2


if __name__ == '__main__':
    solvers = [(key, value) for key, value in globals().items() if key.startswith("day") and callable(value)]
    solvers = sorted(((int(key.split('day')[-1]), value) for key, value in solvers), reverse=True)

    for idx, solver in reversed(solvers):
        ns1 = time.process_time_ns()
        p1, p2 = solver(f"input/day{idx}.txt")
        ns2 = time.process_time_ns()
        print(f"day {idx} - part 1: {p1}, part 2: {p2}. time: {(ns2 - ns1) * 1e-9} seconds")
        # break
