import numpy as np
import random
import heapq
from collections import deque


class RoverEnv:
    def __init__(self, size=10, battery=100):
        self.size = size
        self.max_battery = battery

        self.NORMAL   = 0
        self.SAND     = 1
        self.ROCK     = 2
        self.OBSTACLE = 3

        self.energy_cost = {
            self.NORMAL:   1,
            self.SAND:     3,
            self.ROCK:     5
        }

        self.reset()

    def generate_terrain(self):
        grid = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                r = random.random()
                if   r < 0.55: grid[i][j] = self.NORMAL
                elif r < 0.70: grid[i][j] = self.SAND
                elif r < 0.85: grid[i][j] = self.ROCK
                else:           grid[i][j] = self.OBSTACLE
        return grid

    def _is_solvable(self):
        visited = set()
        q = deque([(0, 0)])
        visited.add((0, 0))
        while q:
            x, y = q.popleft()
            if (x, y) == self.goal:
                return True
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if (0 <= nx < self.size and 0 <= ny < self.size
                        and (nx, ny) not in visited
                        and self.grid[nx][ny] != self.OBSTACLE):
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return False

    def _dijkstra(self, start):
        """
        Compute minimum battery cost from `start` to every reachable cell.
        Returns a dist dict: (x,y) -> min_cost.
        Used for battery-awareness signals.
        """
        dist = {start: 0}
        heap = [(0, start[0], start[1])]
        while heap:
            cost, x, y = heapq.heappop(heap)
            if cost > dist.get((x, y), float('inf')):
                continue
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.grid[nx][ny] != self.OBSTACLE:
                        nc = cost + self.energy_cost.get(self.grid[nx][ny], 1)
                        if nc < dist.get((nx,ny), float('inf')):
                            dist[(nx,ny)] = nc
                            heapq.heappush(heap, (nc, nx, ny))
        return dist

    def reset(self):
        self.start = (0, 0)
        self.goal  = (self.size - 1, self.size - 1)

        for _ in range(200):
            self.grid = self.generate_terrain()
            self.grid[self.start] = self.NORMAL
            self.grid[self.goal]  = self.NORMAL
            for di in range(2):
                for dj in range(2):
                    if self.grid[di][dj] == self.OBSTACLE:
                        self.grid[di][dj] = self.NORMAL
                    gi, gj = self.size-1-di, self.size-1-dj
                    if self.grid[gi][gj] == self.OBSTACLE:
                        self.grid[gi][gj] = self.NORMAL
            if self._is_solvable():
                break

        self.position = list(self.start)
        self.battery  = self.max_battery
        self.steps    = 0

        # Pre-compute optimal cost from start to goal for the episode
        dist = self._dijkstra(self.start)
        self.optimal_cost = dist.get(self.goal, self.max_battery)

        return self.get_state()

    def _get_battery_awareness_signals(self):
        """
        ── BATTERY-AWARENESS BLOCK ──────────────────────────────────────────
        Computes three signals that together let the agent reason about
        whether it has enough energy to complete the mission:

        1. battery_ratio      — raw battery remaining (normalised)
        2. cost_to_goal_ratio — minimum cost to reach goal from here
                                (normalised by max_battery)
        3. budget_margin      — surplus/deficit relative to what is needed:
                                > 0  → agent has more than enough battery
                                = 0  → agent has exactly enough
                                < 0  → agent CANNOT reach goal (critical)

        These three numbers together form the "battery budget" the agent
        can use to decide whether to take a risky short route through rock
        or a safer long route through normal terrain.
        ─────────────────────────────────────────────────────────────────────
        """
        pos = tuple(self.position)

        # Minimum battery cost from current position to goal
        dist_from_here = self._dijkstra(pos)
        min_cost_to_goal = dist_from_here.get(self.goal, self.max_battery)

        battery_ratio      = self.battery / self.max_battery
        cost_to_goal_ratio = min_cost_to_goal / self.max_battery

        # budget_margin > 0: surplus; < 0: agent is stranded
        budget_margin = (self.battery - min_cost_to_goal) / self.max_battery

        return battery_ratio, cost_to_goal_ratio, budget_margin

    def get_state(self):
        """
        17-dimensional state vector:
          [0]   norm_x              — position x (normalised)
          [1]   norm_y              — position y (normalised)
          [2]   goal_dx             — x distance to goal (normalised)
          [3]   goal_dy             — y distance to goal (normalised)
          [4]   battery_ratio       — battery remaining / max_battery
          [5]   cost_to_goal_ratio  — min cost to goal / max_battery  ← NEW
          [6]   budget_margin       — (battery - min_cost) / max_battery ← NEW
          [7]   steps_ratio         — steps taken / max_battery  ← NEW
          [8-16] 3x3 terrain patch  — local terrain visibility
        """
        x, y = self.position
        norm_x = x / (self.size - 1)
        norm_y = y / (self.size - 1)
        dx = (self.goal[0] - x) / (self.size - 1)
        dy = (self.goal[1] - y) / (self.size - 1)

        # Battery-awareness signals
        battery_ratio, cost_to_goal_ratio, budget_margin = \
            self._get_battery_awareness_signals()

        # Steps taken as fraction of battery budget
        steps_ratio = self.steps / self.max_battery

        # 3x3 local terrain patch
        patch = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = x + di, y + dj
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    t = self.grid[ni][nj]
                    if   t == self.OBSTACLE: patch.append(1.0)
                    elif t == self.ROCK:     patch.append(0.6)
                    elif t == self.SAND:     patch.append(0.3)
                    else:                    patch.append(0.0)
                else:
                    patch.append(1.0)

        return np.array(
            [norm_x, norm_y, dx, dy,
             battery_ratio, cost_to_goal_ratio, budget_margin, steps_ratio]
            + patch,
            dtype=float
        )

    def step(self, action):
        x, y = self.position
        self.steps += 1

        nx, ny = x, y
        if   action == 0: nx -= 1
        elif action == 1: nx += 1
        elif action == 2: ny -= 1
        elif action == 3: ny += 1

        nx = max(0, min(self.size - 1, nx))
        ny = max(0, min(self.size - 1, ny))

        # Obstacle collision
        if self.grid[nx][ny] == self.OBSTACLE:
            self.battery -= 1
            if self.battery <= 0:
                return self.get_state(), -10, True
            return self.get_state(), -2, False

        self.position = [nx, ny]
        terrain = self.grid[nx][ny]
        step_cost = self.energy_cost.get(terrain, 1)
        self.battery -= step_cost

        # ── BATTERY-AWARE REWARD SHAPING ─────────────────────────────────────
        # Standard proximity reward
        old_dist = abs(x - self.goal[0]) + abs(y - self.goal[1])
        new_dist = abs(nx - self.goal[0]) + abs(ny - self.goal[1])

        # reward for moving closer to goal
        reward = (old_dist - new_dist) * 0.5

        # small step penalty to encourage efficiency
        reward -= 0.05

        # Terrain penalty
        if terrain == self.SAND: reward -= 0.05
        if terrain == self.ROCK: reward -= 0.1

        # Budget margin penalty — penalise the agent when it is burning
        # battery faster than needed, proportional to how deep into deficit
        # it is going. This is the core battery-awareness shaping term.
        _, _, budget_margin = self._get_battery_awareness_signals()

        if budget_margin < 0:
            # Agent no longer has enough battery to reach goal optimally —
            # scale penalty by how deep the deficit is
            reward += 0.5 * budget_margin   # budget_margin is negative here

        elif budget_margin < 0.1:
            # Danger zone: cutting it very close — mild warning
            reward -= 0.2

        # ─────────────────────────────────────────────────────────────────────

        # Goal reached
        if (nx, ny) == self.goal:
            # Efficiency bonus: reward proportional to remaining battery
            # relative to the optimal cost — not just raw battery remaining.
            # This rewards finding near-optimal paths, not just any path.
            efficiency = (self.battery / self.optimal_cost)
            efficiency = min(efficiency, 1.0)   # cap at 1.0
            return self.get_state(), 100 + 50 * efficiency, True

        # Battery exhausted
        if self.battery <= 0:
            return self.get_state(), -20, True

        return self.get_state(), reward, False

    def render(self):
        g = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                if   self.grid[i][j] == self.OBSTACLE: g[i][j] = -1
                elif self.grid[i][j] == self.SAND:     g[i][j] =  0.5
                elif self.grid[i][j] == self.ROCK:     g[i][j] =  0.7
        g[self.goal[0]][self.goal[1]]         = 2
        g[self.position[0]][self.position[1]] = 1
        print(g)
