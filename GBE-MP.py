import random
import argparse
import time
import csv
import os
import math
from itertools import combinations
from copy import deepcopy

from place_db import PlaceDB
from utils import greedy_placer_with_init_coordinate, write_final_placement, rank_macros, cal_hpwl
from common import grid_setting, my_inf


class GranularBall:
    def __init__(self, ball_id, macros):
        self.id = ball_id
        self.macros = macros  # List of node_id
        self.center_x = 0
        self.center_y = 0
        self.width = 0
        self.height = 0
        self.area = 0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _nearest_free_point(tx, ty, occupied, grid_num):
    """
    在网格内寻找离 (tx, ty) 最近的空闲点（曼哈顿距离优先）。
    """
    tx = clamp(int(tx), 0, grid_num - 1)
    ty = clamp(int(ty), 0, grid_num - 1)
    if (tx, ty) not in occupied:
        return tx, ty

    for r in range(1, grid_num):
        for dx in range(-r, r + 1):
            dy = r - abs(dx)
            cand = [
                (tx + dx, ty + dy),
                (tx + dx, ty - dy),
            ]
            for x, y in cand:
                if 0 <= x < grid_num and 0 <= y < grid_num and (x, y) not in occupied:
                    return x, y
    return tx, ty


def legalize_ball_centers(ball_centers, balls, grid_num):
    """
    球心合法化（硬约束）：禁止多个球共享同一坐标。
    采用"面积大优先"策略为冲突球分配最近空闲点。
    """
    centers = deepcopy(ball_centers)
    occupied = set()
    balls_sorted = sorted(balls, key=lambda b: b.area, reverse=True)

    for b in balls_sorted:
        if b.id not in centers:
            continue
        tx, ty = centers[b.id]
        nx, ny = _nearest_free_point(tx, ty, occupied, grid_num)
        centers[b.id] = (nx, ny)
        occupied.add((nx, ny))
    return centers


def get_bounding_box_for_ball(macros, placedb):
    area_sum = 0
    max_w = 0
    max_h = 0
    for m in macros:
        w = placedb.node_info[m]["x"]
        h = placedb.node_info[m]["y"]
        area_sum += w * h
        max_w = max(max_w, w)
        max_h = max(max_h, h)

    side = int(math.sqrt(area_sum))
    return side, side, area_sum


def get_macro_adjacency(placedb, node_id_ls):
    """
    构建宏单元之间的连线权重图（拓扑邻接矩阵字典）
    共享的Net越多，权重越大（拓扑距离越近）
    """
    adj = {u: {v: 0 for v in node_id_ls} for u in node_id_ls}
    for net_id, net_val in placedb.net_info.items():
        nodes_in_net = list(net_val["nodes"].keys())
        macros_in_net = [n for n in nodes_in_net if n in node_id_ls]
        weight = 1.0 / (len(macros_in_net) - 1) if len(macros_in_net) > 1 else 0
        for i in range(len(macros_in_net)):
            for j in range(i + 1, len(macros_in_net)):
                u, v = macros_in_net[i], macros_in_net[j]
                adj[u][v] += weight
                adj[v][u] += weight
    return adj


def split_with_bfs_assignment(macros, adj, seed_a, seed_b):
    """
    BFS分配策略（借鉴GBGC）：
    以种子为中心，按节点与种子的连接强度进行广度优先优先分配，
    保证拓扑空间邻近性
    """
    group_a = [seed_a]
    group_b = [seed_b]
    remaining = set(macros) - {seed_a, seed_b}

    while remaining:
        best_a_diff = -float('inf')
        best_a_node = None
        for m in remaining:
            conn_a = sum(adj[m][a] for a in group_a)
            conn_b = sum(adj[m][b] for b in group_b)
            diff = conn_a - conn_b
            if diff > best_a_diff:
                best_a_diff = diff
                best_a_node = m

        remaining.discard(best_a_node)
        if best_a_diff > 0:
            group_a.append(best_a_node)
        elif best_a_diff < 0:
            group_b.append(best_a_node)
        else:
            if len(group_a) <= len(group_b):
                group_a.append(best_a_node)
            else:
                group_b.append(best_a_node)

    return group_a, group_b


def quality_of_ball(macros, adj):
    """
    计算粒球的质量：quality(B) = 2*E_B/N_B
    其中E_B是球内边数(权重和)，N_B是球内宏单元数
    这里简化为平均度的计算
    """
    N = len(macros)
    if N <= 1:
        return 0.0
    E = sum(adj[u][v] for u in macros for v in macros if u < v)
    return 2.0 * E / N


def split_macros_by_connectivity(macros, adj):
    """
    基于连接关系将一个球内宏单元切分为两个子集：
    - 先选"最不相连"的两个种子
    - 其余节点按与两个子集总连接强度进行归属
    """
    if len(macros) <= 1:
        return macros, []
    if len(macros) == 2:
        return [macros[0]], [macros[1]]

    min_conn = float("inf")
    seed_a, seed_b = macros[0], macros[1]
    for i in range(len(macros)):
        for j in range(i + 1, len(macros)):
            u, v = macros[i], macros[j]
            if adj[u][v] < min_conn:
                min_conn = adj[u][v]
                seed_a, seed_b = u, v

    group_a = [seed_a]
    group_b = [seed_b]
    rest = [m for m in macros if m != seed_a and m != seed_b]

    rest.sort(key=lambda m: abs(adj[m][seed_a] - adj[m][seed_b]), reverse=True)
    for m in rest:
        score_a = sum(adj[m][a] for a in group_a)
        score_b = sum(adj[m][b] for b in group_b)
        if score_a > score_b:
            group_a.append(m)
        elif score_b > score_a:
            group_b.append(m)
        else:
            if len(group_a) <= len(group_b):
                group_a.append(m)
            else:
                group_b.append(m)

    if len(group_a) == 0 or len(group_b) == 0:
        macros_sorted = sorted(macros, key=lambda x: sum(adj[x][y] for y in macros), reverse=True)
        mid = len(macros_sorted) // 2
        group_a, group_b = macros_sorted[:mid], macros_sorted[mid:]

    return group_a, group_b


def split_if_improves(parent_macros, adj):
    """
    质量驱动的分裂判断：
    只有当子球的质量和大于父球时才执行分裂
    """
    child1, child2 = split_macros_by_connectivity(parent_macros, adj)
    q_parent = quality_of_ball(parent_macros, adj)
    q_child = quality_of_ball(child1, adj) + quality_of_ball(child2, adj)
    if q_child > q_parent:
        return child1, child2
    else:
        return parent_macros, []  # 不分裂


def granular_ball_clustering(node_id_ls, placedb):
    """
    基于粒球计算（GBC）思想的自适应拓扑聚类。
    采用质量驱动分裂：只要分裂能提升子球质量和，就持续分裂；
    否则停止细化。不再使用固定数量阈值。
    """
    adj = get_macro_adjacency(placedb, node_id_ls)

    # 初始化覆盖所有宏单元的粗粒度大球
    initial_ball = GranularBall(0, node_id_ls)
    w, h, area = get_bounding_box_for_ball(node_id_ls, placedb)
    initial_ball.width, initial_ball.height, initial_ball.area = w, h, area

    balls = [initial_ball]
    ball_id_counter = 1

    # 反复分裂直到无法通过质量提升
    changed = True
    while changed:
        changed = False
        new_balls = []
        for ball in balls:
            macros = ball.macros
            if len(macros) <= 1:
                new_balls.append(ball)
                continue

            child1, child2 = split_if_improves(macros, adj)
            if len(child2) == 0:
                # 无法通过分裂提升质量，保留父球
                new_balls.append(ball)
            else:
                # 分裂成功，将子球加入
                for child_macros in (child1, child2):
                    gb = GranularBall(ball_id_counter, child_macros)
                    cw, ch, carea = get_bounding_box_for_ball(child_macros, placedb)
                    gb.width, gb.height, gb.area = cw, ch, carea
                    new_balls.append(gb)
                    ball_id_counter += 1
                changed = True
        balls = new_balls

    return balls, adj  # 返回邻接矩阵供后续使用


def build_ball_level_adjacency(balls, macro_adj):
    ball_adj = {i: {j: 0.0 for j in range(len(balls))} for i in range(len(balls))}
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            w = 0.0
            for u in balls[i].macros:
                for v in balls[j].macros:
                    w += macro_adj[u][v]
            ball_adj[i][j] = w
            ball_adj[j][i] = w
    return ball_adj


def initialize_ball_centers(balls, grid_num):
    """
    为当前粒球分配初始球心：使用均匀网格点，面积大的球优先占位。
    """
    n = len(balls)
    if n == 0:
        return {}
    side = max(1, int(math.ceil(math.sqrt(n))))
    step = max(1, grid_num // (side + 1))

    anchor_points = []
    for i in range(1, side + 1):
        for j in range(1, side + 1):
            x = clamp(i * step, 0, grid_num - 1)
            y = clamp(j * step, 0, grid_num - 1)
            anchor_points.append((x, y))

    random.shuffle(anchor_points)
    balls_sorted = sorted(balls, key=lambda b: b.area, reverse=True)
    centers = {}
    for idx, b in enumerate(balls_sorted):
        if idx < len(anchor_points):
            centers[b.id] = anchor_points[idx]
        else:
            centers[b.id] = (random.randint(0, grid_num - 1), random.randint(0, grid_num - 1))
    centers = legalize_ball_centers(centers, balls, grid_num)
    return centers


def evaluate_granular_hpwl(ball_centers, balls, macro_adj):
    """
    粗粒度HPWL近似：
    使用球间连接权重 * 球心曼哈顿距离 作为代价。
    """
    hpwl = 0.0
    ball_adj = build_ball_level_adjacency(balls, macro_adj)
    for i in range(len(balls)):
        ci = ball_centers[balls[i].id]
        for j in range(i + 1, len(balls)):
            w = ball_adj[i][j]
            cj = ball_centers[balls[j].id]
            dist = abs(ci[0] - cj[0]) + abs(ci[1] - cj[1])
            if w > 0:
                hpwl += w * dist
            if dist == 0:
                overlap_penalty = max(1.0, 5.0 * w + 0.01 * (balls[i].area + balls[j].area))
                hpwl += overlap_penalty
    return hpwl


def build_place_record_from_balls(balls, ball_centers, grid_num):
    """
    将球心映射为宏的初始目标位置。
    同球宏围绕球心做小范围扰动，便于后续legalization。
    """
    place_record = {}
    for b in balls:
        cx, cy = ball_centers[b.id]
        if len(b.macros) == 1:
            m = b.macros[0]
            place_record[m] = {"loc_x": int(cx), "loc_y": int(cy)}
            continue
        r = max(1, int(math.ceil(math.sqrt(len(b.macros)))))
        idx = 0
        for m in b.macros:
            dx = idx % r - r // 2
            dy = idx // r - r // 2
            px = clamp(int(cx + dx), 0, grid_num - 1)
            py = clamp(int(cy + dy), 0, grid_num - 1)
            place_record[m] = {"loc_x": px, "loc_y": py}
            idx += 1
    return place_record


def count_overlaps_in_placement(placed_macros):
    """
    统计最终放置结果中是否存在宏单元重叠（基于离散网格矩形）。
    返回重叠对数。
    """
    nodes = list(placed_macros.keys())
    overlap_pairs = 0
    for i in range(len(nodes)):
        a = placed_macros[nodes[i]]
        ax1, ay1 = a["loc_x"], a["loc_y"]
        ax2, ay2 = ax1 + a["scaled_x"], ay1 + a["scaled_y"]
        for j in range(i + 1, len(nodes)):
            b = placed_macros[nodes[j]]
            bx1, by1 = b["loc_x"], b["loc_y"]
            bx2, by2 = bx1 + b["scaled_x"], by1 + b["scaled_y"]
            ov_x = (ax1 < bx2) and (bx1 < ax2)
            ov_y = (ay1 < by2) and (by1 < ay2)
            if ov_x and ov_y:
                overlap_pairs += 1
    return overlap_pairs


def optimize_coarse_with_splitting(initial_balls, macro_adj, placedb, grid_num,
                                   coarse_round, hpwl_writer=None, hpwl_save_file=None,
                                   run_tag="coarse", deadline_ts=None):
    """
    单次粗粒度优化 + 自适应分裂（Phase 2&3）。
    输入为Phase1得到的初始粒球划分，内部会随机初始化球心并执行完整降维优化流程。
    """
    balls = deepcopy(initial_balls)
    ball_centers = initialize_ball_centers(balls, grid_num)
    next_ball_id = max(b.id for b in balls) + 1 if len(balls) > 0 else 0

    macro_total = sum(len(b.macros) for b in balls)
    while len(balls) < macro_total:
        if deadline_ts is not None and time.time() >= deadline_ts:
            break

        current_score = evaluate_granular_hpwl(ball_centers, balls, macro_adj)
        local_round = max(1, coarse_round // 5)
        for _ in range(local_round):
            if deadline_ts is not None and time.time() >= deadline_ts:
                break
            if len(balls) < 2:
                break

            new_centers = deepcopy(ball_centers)
            if random.random() < 0.6:
                i, j = random.sample(range(len(balls)), 2)
                bi, bj = balls[i], balls[j]
                new_centers[bi.id], new_centers[bj.id] = new_centers[bj.id], new_centers[bi.id]
            else:
                i = random.randrange(len(balls))
                bi = balls[i]
                cx, cy = new_centers[bi.id]
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
                nx = clamp(cx + dx, 0, grid_num - 1)
                ny = clamp(cy + dy, 0, grid_num - 1)
                new_centers[bi.id] = (nx, ny)

            new_centers = legalize_ball_centers(new_centers, balls, grid_num)
            new_score = evaluate_granular_hpwl(new_centers, balls, macro_adj)
            if new_score < current_score:
                ball_centers = new_centers
                current_score = new_score

        if hpwl_writer is not None and hpwl_save_file is not None:
            hpwl_writer.writerow([current_score, time.time(), run_tag])
            hpwl_save_file.flush()

        balls.sort(key=lambda b: b.area, reverse=True)
        target = balls.pop(0)
        target_center = ball_centers.pop(target.id)

        if len(target.macros) > 1:
            # 使用质量驱动分裂判断
            child1_macros, child2_macros = split_if_improves(target.macros, macro_adj)
            if len(child2_macros) == 0:
                # 质量未提升，强制按连通性分裂（粗优化阶段需要继续细化）
                child1_macros, child2_macros = split_macros_by_connectivity(target.macros, macro_adj)
            child1 = GranularBall(next_ball_id, child1_macros)
            next_ball_id += 1
            child2 = GranularBall(next_ball_id, child2_macros)
            next_ball_id += 1
            child1.area = sum(placedb.node_info[m]["x"] * placedb.node_info[m]["y"] for m in child1.macros)
            child2.area = sum(placedb.node_info[m]["x"] * placedb.node_info[m]["y"] for m in child2.macros)
            balls.extend([child1, child2])

            tx, ty = target_center
            ball_centers[child1.id] = (tx, ty)
            ball_centers[child2.id] = (tx, ty)
            ball_centers = legalize_ball_centers(ball_centers, balls, grid_num)
        else:
            ball_centers[target.id] = target_center
            balls.append(target)
            break

    return balls, ball_centers


def main():
    parser = argparse.ArgumentParser(description='Granular-Ball WireMask-BBO')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--seed', required=True)
    parser.add_argument('--coarse_round', default=50)
    parser.add_argument('--fine_round', default=-1)
    parser.add_argument('--init_round', default=0)
    parser.add_argument('--time_limit_minutes', default=1000)
    args = parser.parse_args()

    dataset = args.dataset
    seed1 = int(args.seed)
    coarse_round = int(args.coarse_round)
    fine_round = int(args.fine_round)
    init_round = int(args.init_round)
    time_limit_minutes = float(args.time_limit_minutes)
    start_ts = time.time()
    deadline_ts = start_ts + max(0.0, time_limit_minutes) * 60.0

    random.seed(seed1)
    placedb = PlaceDB(dataset)

    grid_num = grid_setting[dataset]["grid_num"]
    grid_size = grid_setting[dataset]["grid_size"]

    node_id_ls = rank_macros(placedb)

    hpwl_save_dir = "result/GB_WireMask/curve/"
    placement_save_dir = "result/GB_WireMask/placement/"

    if not os.path.exists(hpwl_save_dir):
        os.makedirs(hpwl_save_dir)
    if not os.path.exists(placement_save_dir):
        os.makedirs(placement_save_dir)

    hpwl_save_path = hpwl_save_dir + "{}_seed_{}.csv".format(dataset, seed1)
    placement_save_path = placement_save_dir + "{}_seed_{}.csv".format(dataset, seed1)

    hpwl_save_file = open(hpwl_save_path, "a+")
    hpwl_writer = csv.writer(hpwl_save_file)

    # ---------------------------------------------------------
    # Phase 1: Quality-driven Granular Ball Clustering
    # ---------------------------------------------------------
    print("Phase 1: Quality-driven Granular-Ball Clustering (no fixed size threshold)")
    base_balls, macro_adj = granular_ball_clustering(node_id_ls, placedb)

    # ---------------------------------------------------------
    # Phase 2 & 3: Multi-start Coarse Optimization at Ball Level
    # ---------------------------------------------------------
    coarse_restart = max(1, init_round + 1)
    print(f"Phase 2: Coarse-grained Multi-start Optimization (restarts={coarse_restart})")
    print(f"Global time limit: {time_limit_minutes} minutes")

    best_hpwl = my_inf
    best_place_record = None
    best_placed_macro = None

    for rid in range(coarse_restart):
        if time.time() >= deadline_ts:
            print("Time limit reached during coarse multi-start. Stop coarse phase.")
            break

        balls, ball_centers = optimize_coarse_with_splitting(
            base_balls, macro_adj, placedb, grid_num, coarse_round,
            hpwl_writer=hpwl_writer, hpwl_save_file=hpwl_save_file,
            run_tag=f"coarse_r{rid}", deadline_ts=deadline_ts
        )

        init_place_record = build_place_record_from_balls(balls, ball_centers, grid_num)
        init_placed_macro, init_hpwl = greedy_placer_with_init_coordinate(
            node_id_ls, placedb, grid_num, grid_size, init_place_record
        )

        if not init_placed_macro:
            continue

        init_hpwl = cal_hpwl(init_placed_macro, placedb)
        hpwl_writer.writerow([init_hpwl, time.time(), "init_from_coarse", rid])
        hpwl_save_file.flush()

        if init_hpwl < best_hpwl:
            best_hpwl = init_hpwl
            best_place_record = deepcopy(init_place_record)
            best_placed_macro = init_placed_macro

    if best_place_record is None:
        fallback_centers = initialize_ball_centers(base_balls, grid_num)
        best_place_record = build_place_record_from_balls(base_balls, fallback_centers, grid_num)
        best_placed_macro, best_hpwl = greedy_placer_with_init_coordinate(
            node_id_ls, placedb, grid_num, grid_size, best_place_record
        )
        if not best_placed_macro:
            print("No legal placement found under current settings.")
            hpwl_save_file.close()
            return
        best_hpwl = cal_hpwl(best_placed_macro, placedb)
        hpwl_writer.writerow([best_hpwl, time.time(), "fallback_init"])
        hpwl_save_file.flush()

    # ---------------------------------------------------------
    # Phase 4: Fine-tune (Legalization and Local Swap)
    # ---------------------------------------------------------
    print("Phase 4: Fine-tune Optimization")

    if time.time() >= deadline_ts:
        print("Time limit reached before fine-tune phase. Skip fine-tune.")
        print(f"Optimization Finished. Best HPWL: {best_hpwl}")
        hpwl_save_file.close()
        return

    current_place_record = deepcopy(best_place_record)
    current_hpwl = best_hpwl
    eval_count = 1
    write_final_placement(best_placed_macro, placement_save_path)

    fine_iter = 0
    while time.time() < deadline_ts and (fine_round < 0 or fine_iter < fine_round):
        candidate_place_record = deepcopy(current_place_record)
        node_a, node_b = random.sample(node_id_ls, 2)
        (candidate_place_record[node_a]["loc_x"],
         candidate_place_record[node_a]["loc_y"],
         candidate_place_record[node_b]["loc_x"],
         candidate_place_record[node_b]["loc_y"]) = (
            candidate_place_record[node_b]["loc_x"],
            candidate_place_record[node_b]["loc_y"],
            candidate_place_record[node_a]["loc_x"],
            candidate_place_record[node_a]["loc_y"]
        )

        placed_macro, hpwl = greedy_placer_with_init_coordinate(
            node_id_ls, placedb, grid_num, grid_size, candidate_place_record
        )
        if not placed_macro:
            continue

        hpwl = cal_hpwl(placed_macro, placedb)
        eval_count += 1
        accept = hpwl < current_hpwl

        if accept:
            current_place_record = candidate_place_record
            current_hpwl = hpwl

        if hpwl < best_hpwl:
            best_hpwl = hpwl
            best_place_record = deepcopy(candidate_place_record)
            best_placed_macro = placed_macro
            write_final_placement(best_placed_macro, placement_save_path)

        hpwl_writer.writerow([hpwl, time.time(), "fine-tune_hill", int(accept)])
        hpwl_save_file.flush()
        fine_iter += 1

    print(f"Optimization Finished. Best HPWL: {best_hpwl}")
    print(f"Legalized HPWL evaluations used: {eval_count}")
    print(f"Fine-tune iterations used: {fine_iter}")
    overlap_pairs = count_overlaps_in_placement(best_placed_macro)
    print(f"Final overlap pair count: {overlap_pairs}")
    print(f"Elapsed time: {(time.time() - start_ts) / 60.0:.2f} minutes")
    hpwl_save_file.close()


if __name__ == "__main__":
    main()