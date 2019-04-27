# coding: utf-8
# value_planner_test.py

from maze_environment import CellType, Environment
from planner import ValueIterationPlanner

def main():
    grid = [[CellType.NORMAL, CellType.NORMAL, CellType.NORMAL, CellType.REWARD],
            [CellType.NORMAL, CellType.BLOCK,  CellType.NORMAL, CellType.DAMAGE],
            [CellType.NORMAL, CellType.NORMAL, CellType.NORMAL, CellType.NORMAL]]
    
    env = Environment(grid)
    planner = ValueIterationPlanner(env)
    value_grid = planner.plan()
    
    for row in value_grid:
        print(", ".join([str(value) for value in row]))

if __name__ == "__main__":
    main()
    