import torch
import os, random, json, math, re, time, unicodedata, heapq
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


seed_all(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = dev == "cuda" and torch.cuda.get_device_capability()[0] >= 8


def amp():
    return torch.autocast(dev, torch.bfloat16, enabled=use_bf16)
MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}  # SỬA TAY: chatbot quên dòng này


def step_cost(ch):
    return int(ch) if ch.isdigit() else 1

def dijkstra(g, src):
    rows = len(g)
    cols = len(g[0])
    dist = {}
    par = {}
    heap = []
    for r in range(rows):
        for c in range(cols):
            dist[(r, c)] = float('inf')
    dist[src] = 0
    heapq.heappush(heap, (0, src[0], src[1]))
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[(r, c)]:
            continue
        for move, (dr, dc) in MOVES.items():
            nr = r + dr
            nc = c + dc
            if 0 <= nr < rows and 0 <= nc < len(g[nr]):
                cell = g[nr][nc]
                if cell == '#':
                    continue
                cost = step_cost(cell)
                if dist[(nr, nc)] > d + cost:
                    dist[(nr, nc)] = d + cost
                    par[(nr, nc)] = ((r, c), move)
                    heapq.heappush(heap, (dist[(nr, nc)], nr, nc))
    return dist, par

def path(par, src, dst):
    moves = []
    current = dst
    while current != src:
        parent, move = par[current]
        moves.append(move)
        current = parent
    return ''.join(reversed(moves))

def solve_map(grid):
    g = grid.split('/')
    rows = len(g)
    cols = [len(row) for row in g]
    S = None
    G = None
    pts = []
    for r in range(rows):
        for c in range(len(g[r])):
            cell = g[r][c]
            if cell == 'S':
                S = (r, c)
            elif cell == 'G':
                G = (r, c)
            elif cell.isalpha() and cell not in ['S', 'G']:
                pts.append((r, c))
    K = len(pts)
    runs = {}
    for p in pts + [S]:
        runs[p] = dijkstra(g, p)
    def d(a, b):
        return runs[a][0][b]
    dp = {}
    prev = {}
    for i in range(K):
        dp[(1 << i, i)] = d(S, pts[i])
    for mask in range(1, 1 << K):
        for i in range(K):
            if (mask & (1 << i)) == 0:
                continue
            current_state = (mask, i)
            if current_state not in dp:
                continue
            for j in range(K):
                if (mask & (1 << j)) != 0:
                    continue
                new_mask = mask | (1 << j)
                new_cost = dp[current_state] + d(pts[i], pts[j])
                if (new_mask, j) not in dp or new_cost < dp[(new_mask, j)]:
                    dp[(new_mask, j)] = new_cost
                    prev[(new_mask, j)] = i
    full = (1 << K) - 1
    min_cost = float('inf')
    last = -1
    for i in range(K):
        total_cost = dp[(full, i)] + d(pts[i], G)
        if total_cost < min_cost:
            min_cost = total_cost
            last = i
    order = [last]
    m = full
    while m != (1 << order[-1]):
        i = prev[(m, order[-1])]
        m ^= (1 << order[-1])
        order.append(i)
    order.reverse()
    stops = [S] + [pts[i] for i in order] + [G]
    actions = ''.join(path(runs[a][1], a, b) for a, b in zip(stops, stops[1:]))
    return actions

def main():
    df = pd.read_csv('data/a_maps.csv')
    results = []
    for _, row in df.iterrows():
        grid = row['grid']
        id = row['id']
        actions = solve_map(grid)
        results.append({'id': id, 'actions': actions})
    submission = pd.DataFrame(results)
    submission.to_csv('submission_a.csv', index=False)

if __name__ == '__main__':
    main()
