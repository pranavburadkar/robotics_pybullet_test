import heapq
import numpy as np

class Node:
    def __init__(self, x, y, cost):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = None

    def __lt__(self, other):
        return self.cost < other.cost

def astar_search(occ_grid, start, goal):
    # D* grid search, avoid obstacles
    w, h = occ_grid.shape
    open_list = []
    visited = set()
    
    start_node = Node(start[0], start[1], 0)
    heapq.heappush(open_list, (0, start_node))
    
    while open_list:
        _, node = heapq.heappop(open_list)
        key = (node.x, node.y)
        
        if key in visited:
            continue
        visited.add(key)
        
        if key == tuple(goal):
            path = []
            while node:
                path.append((node.x, node.y))
                node = node.parent
            return path[::-1]
        
        # Explore neighbors
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = node.x + dx, node.y + dy
            
            if 0 <= nx < w and 0 <= ny < h and occ_grid[nx, ny] < 0.5: # Ensure the cell is not an obstacle
                nnode = Node(nx, ny, node.cost + 1)
                nnode.parent = node
                heuristic_cost = heuristic(nnode, goal)
                heapq.heappush(open_list, (nnode.cost + heuristic_cost, nnode))
    
    return []  # No path found

def heuristic(node, goal):
    return abs(node.x - goal[0]) + abs(node.y - goal[1])
