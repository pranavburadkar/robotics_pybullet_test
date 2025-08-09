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
    open_list = []
    g_score = {}
    
    start_node = Node(start[0], start[1], 0)
    g_score[(start[0], start[1])] = 0
    heapq.heappush(open_list, (0 + heuristic(start_node, goal), start_node))
    
    while open_list:
        f_score, node = heapq.heappop(open_list)
        key = (node.x, node.y)
        
        if key == tuple(goal):
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
            
            if 0 <= nx < w and 0 <= ny < h and occ_grid[nx, ny] < 0.5: # Ensure the cell is not an obstacle
                new_g_score = node.cost + 1
                
                if new_g_score < g_score.get((nx, ny), float('inf')):
                    nnode = Node(nx, ny, new_g_score, node)
                    g_score[(nx, ny)] = new_g_score
                    heapq.heappush(open_list, (new_g_score + heuristic(nnode, goal), nnode))
    
    return []  # No path found

def heuristic(node, goal):
    return abs(node.x - goal[0]) + abs(node.y - goal[1])
