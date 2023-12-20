import functools
import itertools
import time
from math import sqrt, floor, ceil, lcm
import collections
import operator
from itertools import zip_longest, starmap, count, chain, islice, takewhile, accumulate, tee, dropwhile, repeat, \
    combinations, permutations, cycle
from functools import reduce, partial, cmp_to_key
from collections import Counter
import re
from typing import Iterable

import numpy as np


def tails(xs):
    while xs:
        yield xs
        xs = xs[1:]


def sliding_window(iterable, n):
    iterables = itertools.tee(iterable, n)

    for iterable, num_skipped in zip(iterables, itertools.count()):
        for _ in range(num_skipped):
            next(iterable, None)

    return zip(*iterables)


def day11(filename: str):
    galaxies = {(row, col)
                   for row, line in enumerate(open(filename).readlines())
                   for col, ch in enumerate(line)
                   if ch == '#'}

    rows = { row for row, _ in galaxies }
    cols = { col for _, col in galaxies }

    def num_gaps(r1, r2, values):
        return sum(r not in values for r in range(min(r1, r2), max(r1, r2)))

    def distance(r1, c1, r2, c2):
        return abs(r1 - r2) + abs(c1 - c2), num_gaps(r1, r2, rows) + num_gaps(c1, c2, cols)

    distances = [distance(*pt1, *pt2) for pt1, pt2 in itertools.combinations(galaxies, 2)]
    part1 = sum(d + gap for d, gap in distances)
    part2 = sum(d + gap * 999999 for d, gap in distances)

    return part1, part2


def day10(filename: str):
    grid = {(r * 2, c * 2): ch
            for r, row in enumerate(open(filename).readlines())
            for c, ch in enumerate(row.strip())}
    maxrow, maxcol = max(grid)

    valid = dict(up={'|': "7F|", '-': "", '7': "", 'J': "7F|", 'L': "7F|", 'F': "", 'S': "7F|"},
                 down={'|': "|JL", '-': "", '7': "|JL", 'J': "", 'L': "", 'F': "|JL", 'S': "|JL"},
                 left={'|': "", '-': "-FL", '7': "-FL", 'J': "-FL", 'L': "", 'F': "", 'S': "-FL"},
                 right={'|': "", '-': "-J7", '7': "", 'J': "", 'L': "-J7", 'F': "-J7", 'S': "-J7"})

    def nbors(grid, r, c):
        moves = zip(("up", "down", "left", "right"), ((r - 2, c), (r + 2, c), (r, c - 2), (r, c + 2)))
        return {pos for name, pos in moves if grid.get(pos, '*') in valid[name][grid[(r, c)]]}

    def find_cycle() -> set:
        pos = start = next(pt for pt in grid if grid[pt] == 'S')
        cycle = set()
        while pos not in cycle:
            cycle.add(pos)
            nextpos = next(iter(nbors(grid, *pos) - cycle), start)
            yield tuple((a + b) // 2 for a, b in zip(pos, nextpos))
            yield nextpos
            pos = nextpos

    def prune(positions, pos):
        queue = [pos]
        while queue:
            r, c = queue[-1]
            queue.pop()
            if (r, c) in positions:
                positions.discard((r, c))
                for pos in ((r + 1, c), (r - 1, c), (r, c - 1), (r, c + 1)):
                    if pos in positions:
                        queue.append(pos)

    cycle = set(find_cycle())
    remaining = {(r, c) for r in range(-1, maxrow + 2) for c in range(-1, maxcol + 2)} - cycle
    prune(remaining, (-1, -1))
    return len(cycle) // 4, sum(r % 2 == 0 and c % 2 == 0 for r, c in remaining)


def day9(filename: str):
    data = [[int(nr) for nr in line.split(' ')] for line in open(filename).readlines()]

    def solve(nrs: list[int], i: int, j: int):
        diffs = list(map(operator.sub, nrs[1:], nrs))
        return 0 if all(y == 0 for y in nrs) else nrs[i] + j * solve(diffs, i, j)

    part1 = sum(solve(xs, -1, 1) for xs in data)
    part2 = sum(solve(xs, 0, -1) for xs in data)
    return part1, part2


def day8(filename: str):
    instructions, elements = open(filename).read().split('\n\n')
    data = zip(*(re.findall(r'[0-9A-Z][0-9A-Z][0-9A-Z]', line.strip()) for line in elements.split('\n')))
    network = {s: (l, r) for s, l, r in zip(*data)}

    def steps(start: str, done):
        path = accumulate(cycle(instructions), lambda cur, c: network[cur][0 if c == 'L' else 1], initial=start)
        return next(k for s, k in zip(path, count()) if done(s))

    part1 = steps('AAA', lambda s: s == 'ZZZ')
    part2 = lcm(*(steps(start, lambda s: s.endswith('Z')) for start in {s for s in network if s.endswith('A')}))
    return part1, part2


def day7(filename: str):
    values = (dict(T=10, J=11, Q=12, K=13, A=14))
    values.update({str(k): k for k in range(2, 10)})
    scores = {(5,): 6, (4, 1): 5, (3, 2): 4, (3, 1, 1): 3, (2, 2, 1): 2, (2, 1, 1, 1): 1, (1, 1, 1, 1, 1): 0}

    def apply_joker(hand: (str, str, str, str, str)) -> (str, str, str, str, str):
        ranked = sorted(((v, values[k], k) for k, v in Counter(hand).items()), reverse=True)
        best = next((card for count, _, card in ranked if card != 'J'), None)
        return hand if best is None else tuple(best if card == 'J' else card for card in hand)

    def score(hand: (str, str, str, str, str)) -> int:
        counts = tuple(sorted(Counter(hand if values['J'] > 10 else apply_joker(hand)).values(), reverse=True))
        return scores[counts]

    def order(hand1: (str, str, str, str, str), hand2: (str, str, str, str, str)):
        return s1 - s2 if (s1 := score(hand1)) != (s2 := score(hand2)) else next(
            (values[c1] - values[c2] for c1, c2 in zip(hand1, hand2) if c1 != c2), 0)

    data = zip(*(line.strip().split(' ') for line in open(filename).readlines()))
    bids = {tuple(hand): int(bid) for hand, bid in zip(*data)}

    ranked = sorted(bids.keys(), key=functools.cmp_to_key(order))
    part1 = sum(bids[y] * rank for y, rank in zip(ranked, count(1)))
    values['J'] = 1
    ranked = sorted(bids.keys(), key=functools.cmp_to_key(order))
    part2 = sum(bids[y] * rank for y, rank in zip(ranked, count(1)))
    return part1, part2


def day6(filename: str):
    def ways2win(time: int, distance: int):
        d = sqrt(time * time - 4 * distance)
        return floor((time + d) / 2) - ceil((time - d) / 2) + 1

    data = open(filename).read().split('\n')
    times = re.findall(r'(\d+)', data[0])
    distances = re.findall(r'(\d+)', data[1])
    part1 = reduce(operator.mul, map(ways2win, map(int, times), map(int, distances)))
    part2 = ways2win(int(''.join(times)), int(''.join(distances)))
    return part1, part2


def day5(filename: str):
    data = open(filename).read().split('\n\n')
    seeds = [int(nr) for nr in re.findall(r'(\d+)', data[0])]

    def build(idx):
        ms = [m.groups() for line in data[idx].split('\n') if (m := re.match(r'(\d+) (\d+) (\d+)', line.strip()))]
        return {(int(s), int(s) + int(n)): (int(d) - int(s)) for d, s, n in ms}

    def overlaps(a1, a2, b1, b2):
        return max(a1, b1) < min(a2, b2)

    def splitranges(a1, a2, b1, b2):
        return {(d1, d2) for d1, d2 in ((a1, min(b1, a2)), (max(a1, b1), min(a2, b2)), (max(a1, b2), a2))
                if d2 > d1}

    def transform(a1, b1, ranges: dict[(int, int), int]) -> (int, int):
        return next(((a1 + delta, b1 + delta) for (a2, b2), delta in ranges.items() if overlaps(a2, b2, a1, b1)),
                    (a1, b1))

    transformers = [build(i + 1) for i in range(7)]

    def combine(ranges: set[(int, int)], trans: dict[(int, int), int]) -> dict[(int, int), int]:
        combined = set()
        for range in ranges:
            result = reduce(lambda res, r2: reduce(set.union, (splitranges(*r, *r2) for r in res)), trans, {range})
            combined = combined.union({transform(*r, trans) for r in result})
        return combined

    part1 = min(a for a, _ in reduce(combine, transformers, {(a, a + 1) for a in seeds}))
    part2 = min(a for a, _ in
                reduce(combine, transformers,
                       {(seeds[i], seeds[i] + seeds[i + 1]) for i in range(0, len(seeds), 2)}))

    return part1, part2


def day4(filename: str):
    parts = [re.split(r'\||Card\s+(\d+)+:', line.strip()) for line in open(filename).readlines()]
    ids = [int(id) for _, id, _, _, _ in parts]
    wins = [{int(nr) for nr in re.findall(r'\d+', nrs)} for _, _, nrs, _, _ in parts]
    haves = [{int(nr) for nr in re.findall(r'\d+', nrs)} for _, _, _, _, nrs in parts]
    points = {id: list(range(id + 1, id + len(hs.intersection(ws)) + 1)) for id, ws, hs in zip(ids, wins, haves)}

    part1 = sum(2 ** (len(count) - 1) for count in points.values() if count)
    part2 = reduce(lambda counts, pair: counts + Counter({id: counts[pair[0]] for id in pair[1]}), points.items(),
                   Counter({id: 1 for id in ids}))

    return part1, sum(part2.values())


def day3(filename: str):
    def neighbours(row: int, col: int):
        return {(r, c) for r in range(row - 1, row + 2) for c in range(col - 1, col + 2)}

    parts = reduce(set.union, (neighbours(row, m.span()[0])
                               for row, s in enumerate(open(filename).readlines())
                               for m in re.finditer(r'[^0-9\.]', s.strip())))
    gears = {(row, m.span()[0]): set()
             for row, s in enumerate(open(filename).readlines())
             for m in re.finditer(r'\*', s.strip())}

    part1 = 0
    for row, line in enumerate(open(filename).readlines()):
        for m in re.finditer(r'(\d+)', line.strip()):
            number = int(m.group(1))
            if any((row, col) in parts for col in range(*m.span())):
                part1 += number
            adjacent_gears = (gear for col in range(*m.span()) for gear in gears if (row, col) in neighbours(*gear))
            for gear in adjacent_gears:
                gears[gear].add(number)
    part2 = sum(xs[0] * xs[1] for ns in gears.values() if len((xs := list(ns))) == 2)
    return part1, part2


def day2(filename: str):
    def collect(s):
        return np.array([int(match.group(1))
                         if (match := re.search(f'(\\d+) {k}', s)) else 0
                         for k in {'red', 'green', 'blue'}])

    lines = [re.split(r':|;', line.strip()) for line in open(filename).readlines()]
    ids = (int(header[5:]) for header, *_ in lines)
    maxs = [reduce(np.maximum, (collect(draw) for draw in draws)) for _, *draws in lines]
    part1 = sum(id for id, m in zip(ids, maxs) if all(m <= np.array([12, 13, 14])))
    part2 = sum(reduce(operator.mul, m) for m in maxs)
    return part1, part2


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


if __name__ == '__main__':
    solvers = [(key, value) for key, value in globals().items() if key.startswith("day") and callable(value)]
    solvers = sorted(((int(key.split('day')[-1]), value) for key, value in solvers), reverse=True)

    for idx, solver in solvers:
        ns1 = time.process_time_ns()
        p1, p2 = solver(f"input/day{idx}.txt")
        ns2 = time.process_time_ns()
        print(f"day {idx} - part 1: {p1}, part 2: {p2}. time: {(ns2 - ns1) * 1e-9} seconds")
        # break
