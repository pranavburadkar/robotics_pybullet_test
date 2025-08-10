import heapq
import numpy as np

class Node:
    def __init__(self, x, y, cost, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent

    def __lt__(self, other):
        return self.cost < other.cost

def astar_search(occ_grid, start, goal):
    # D* grid search, avoid obstacles
    w, h = occ_grid.shape

    print(f"DEBUG: A* Search - Start: {start}, Goal: {goal}")
    print(f"DEBUG: A* Search - occ_grid shape: {occ_grid.shape}")

    # Check if start and goal are within grid boundaries
    if not (0 <= start[0] < w and 0 <= start[1] < h and 0 <= goal[0] < w and 0 <= goal[1] < h):
        print(f"DEBUG: A* Search - Start or Goal out of bounds. Start: {start}, Goal: {goal}, Grid: ({w}, {h})")
        return [] # Return empty path if start or goal is out of bounds

    # Check if start or goal is an obstacle
    if occ_grid[start[1], start[0]] >= 0.5:
        print(f"DEBUG: A* Search - Start is an obstacle (or unknown). Value: {occ_grid[start[1], start[0]]}")
        return []
    if occ_grid[goal[1], goal[0]] >= 0.5:
        print(f"DEBUG: A* Search - Goal is an obstacle (or unknown). Value: {occ_grid[goal[1], goal[0]]}")
        return []

    open_list = []
    g_score = {}
    
    start_node = Node(start[0], start[1], 0)
    g_score[(start[0], start[1])] = 0
    heapq.heappush(open_list, (0 + heuristic(start_node, goal), start_node))
    
    while open_list:
        f_score, node = heapq.heappop(open_list)
        key = (node.x, node.y)
        
        # print(f"DEBUG: A* Search - Popped node: {key}, f_score: {f_score}")

        if key == tuple(goal):
            print(f"DEBUG: A* Search - Path found to goal: {goal}")
            path = []
            while node:
                path.append((node.x, node.y))
                node = node.parent
            return path[::-1]
        
        # If we found a shorter path to this node already, skip
        if f_score > g_score.get(key, float('inf')) + heuristic(node, goal):
            continue

        # Explore neighbors
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = node.x + dx, node.y + dy
            
            # print(f"DEBUG: A* Search - Exploring neighbor: ({nx}, {ny})")
            if not (0 <= nx < w and 0 <= ny < h):
                # print(f"DEBUG: A* Search - Neighbor ({nx}, {ny}) out of bounds.")
                continue

            if occ_grid[ny, nx] >= 0.7: # Ensure the cell is not an obstacle (only allow free cells)
                # print(f"DEBUG: A* Search - Neighbor ({nx}, {ny}) is obstacle/unknown. Value: {occ_grid[ny, nx]}")
                continue

            new_g_score = node.cost + 1
            
            if new_g_score < g_score.get((nx, ny), float('inf')):
                nnode = Node(nx, ny, new_g_score, node)
                g_score[(nx, ny)] = new_g_score
                heapq.heappush(open_list, (new_g_score + heuristic(nnode, goal), nnode))
    
    print("DEBUG: A* Search - Open list empty, no path found.")
    return []  # No path found

def heuristic(node, goal):
    return abs(node.x - goal[0]) + abs(node.y - goal[1])
