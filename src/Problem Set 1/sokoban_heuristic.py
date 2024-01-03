import math
from typing import FrozenSet
from sokoban import SokobanProblem, SokobanState
from mathutils import Point, manhattan_distance

# This heuristic returns the distance between the player and the nearest crate as an estimate for the path cost
# While it is consistent, it does a bad job at estimating the actual cost thus the search will explore a lot of nodes before finding a goal
def weak_heuristic(problem: SokobanProblem, state: SokobanState):
    return min(manhattan_distance(state.player, crate) for crate in state.crates) - 1
'''
Check if the given position is a goal point (Available in the goals frozen set or not).
If it's a goal point then we don't need to check for any deadlock since the crate has
already reached a goal

Parameters:
- position (Point): The current position to check.
- goals (FrozenSet[Point]): A set of goal points.

Returns:
- bool: True if the position is a goal point, False otherwise.
'''
def checkGoalPoint(position: Point, goals: FrozenSet[Point]) -> bool:
    if (Point(position.x, position.y) in goals):
        return True
    return False

'''
Check if the given position is in an L-shaped deadlock, surrounded by unwalkable points.
Example:
#    $#    #    $#
$#   #    #$    #

Here the player doesn't have anywhere to come in order to push the stuck crate, therefore it
will never reach the goal

Parameters:
- position (Point): The current position to check.
- walkable (FrozenSet[Point]): A set of walkable points.

Returns:
- bool: True if the position is in an L-shaped deadlock, False otherwise.
'''
def checkLshapedDeadlock(position: Point, walkable: FrozenSet[Point]) -> bool:
    x, y = position.x, position.y
    return ((Point(x, y + 1) not in walkable and Point(x + 1, y) not in walkable) or
            (Point(x, y + 1) not in walkable and Point(x - 1, y) not in walkable) or
            (Point(x, y - 1) not in walkable and Point(x + 1, y) not in walkable) or
            (Point(x, y - 1) not in walkable and Point(x - 1, y) not in walkable))

'''
Find the nearest goal from the given position.

Parameters:
- position (Point): The current position.
- goals (FrozenSet[Point]): A set of goal points.

Returns:
- Point: The nearest goal point to the current position, or None if there are no goals.
'''
def getNearestGoal(position: Point, goals: FrozenSet[Point]) -> Point:
    return min(goals, key=lambda goal: manhattan_distance(position, goal), default=None)

'''
Check if the nearest goal from the given position is unreachable due to a boundary.
This means that the crate won't ever be able to reach it since it can't be moved towards
that goal by the player
Example:
#########
#       #
#$     .#
#       #
#########
This crate won't be pushed to the goal on the right, deadlock!
Now check from the other 3 boundaries as well.

Parameters:
- position (Point): The current position.
- goals (FrozenSet[Point]): A set of goal points.
- width (int): The width of the grid.
- height (int): The height of the grid.

Returns:
- bool: True if the nearest goal is unreachable, False otherwise.
'''
def checkUnreachableGoal(position: Point, goals: FrozenSet[Point], width: int, height: int) -> bool:
    goal = getNearestGoal(position, goals)
    x, y = position.x, position.y
    return ((x - 1 == 0 and goal.x > x) or
            (x + 1 == width - 1 and goal.x < x) or
            (y - 1 == 0 and goal.y > y) or
            (y + 1 == height - 1 and goal.y < y))
   
'''
This function is used as a utility function to check for the square deadlock
The count of crates and walls has to be 4 in order for the deadlock to occur
And there must be atleast one crate in the square so that we can consider
Example:
##  ##  $$  $$
#$  $$  $#  $$

These crates are all stuck, there's no way for the player to push them apart because
they are surrounded from all directions they can be moved to. 

Parameters:
- positions (list[Point]): A list of positions to check.
- walkable (FrozenSet[Point]): A set of walkable points.
- crates (FrozenSet[Point]): A set of crate positions.

Returns:
- bool: True if the condition holds, False otherwise.
''' 
def checkStuckCrate(positions: list[Point], walkable: FrozenSet[Point], crates: FrozenSet[Point]) -> bool:
    c1, c2 = 0, 0
    for pos in positions:
        c1 = c1 + 1 if pos not in walkable else c1
        c2 = c2 + 1 if pos in crates else c2
    return c2 >= 1 and (c1 + c2) == 4
    
'''
Check if the given position is part of a square deadlock, where one or more crates are stuck.
It initializes the positions list with all possibilities of having a square around a crate
Then loops across these positions and uses the above function to check if there are walls and
crates enough to make this case a deadlock. If one of them returned true then we don't need to continue

Parameters:
- position (Point): The current position to check.
- walkable (FrozenSet[Point]): A set of walkable points.
- crates (FrozenSet[Point]): A set of crate positions.

Returns:
- bool: True if the position is part of a square deadlock, False otherwise.
'''
def checkSquareDeadlock(position: Point, walkable: FrozenSet[Point], crates: FrozenSet[Point]) -> bool:
    x, y = position.x, position.y
    positionsList = [[Point(x, y), Point(x + 1, y), Point(x, y + 1), Point(x + 1, y + 1)],
                    [Point(x, y), Point(x - 1, y), Point(x, y + 1), Point(x - 1, y + 1)],
                    [Point(x, y), Point(x - 1, y), Point(x, y - 1), Point(x - 1, y - 1)],
                    [Point(x, y), Point(x + 1, y), Point(x, y - 1), Point(x + 1, y - 1)]]
    for positions in positionsList:
        if checkStuckCrate(positions, walkable, crates):
            return True
    return False

'''
Our main deadlock function,
Check if a crate is in a deadlock state.
It calls all the above functions.
The checkGoalPoint is the only function which returns false in case the crate is on a goal.
The other functions return true in case the crate can't move anymore.
One of them is enough to return true

Parameters:
- state (SokobanState): The current state of the Sokoban puzzle.
- crate (Point): The position of the crate to check.

Returns:
- bool: True if the crate is in a deadlock state, False otherwise.
'''
def isDeadlock(state: SokobanState, crate: Point) -> bool:    
    if checkGoalPoint(crate, state.layout.goals):
        return False
    
    if (checkLshapedDeadlock(crate, state.layout.walkable) or
        checkUnreachableGoal(crate, state.layout.goals, state.layout.width, state.layout.height) or
        checkSquareDeadlock(crate, state.layout.walkable, state.crates)):
        return True
    
    return False

'''
The idea behind this problem is to calculate an estimate of the heuristic function for each state we receive

We first start by getting our cache from the problem and initializing our crates frozen set
Then check if the current crates positions appeared before in the cache, this means that we don't
need to recalculate the distances because they'll be the same value. This is because goals positions
are fixed, therefore when we also have the same crates positions as before, it'll be a duplicate state basically.
We'll return the heuristic value in the cache and add on it the minimum manhattan distance from the current player
position to the nearest crate [as an extra calculation for the heuristic function to make it more efficient]

After we've made sure that the crates are in new positions, we then see if this state is a goal state or not
so that we directly return 0. [h(n) at the goal state = 0]

Initialize the heuristic value with 0 so that we can accumulate on it the distances we are going to calculate

Loop over all crates positions in the current state and check if any one of the crates is impossible to move (deadlock)
we'll then just return infinity as our heuristic value because we don't want to pop this state from the frontier and expand
its useless successors because we are sure that it won't reach a goal state due to the deadlock condition

If not, then proceed to add on the heuristic value the minimum distance between each crate and the nearest goal

After we've calculated the heuristic value, we add it to the cache so that we can check on it when we enter the function again.

Finally we'll return the calculated heuristic value + the distance between the player & nearest crate (Weak heuristic) for good measure.
'''
def strong_heuristic(problem: SokobanProblem, state: SokobanState) -> float:
    cache = problem.cache()
    crates = state.crates
        
    if crates in cache:
        return cache[crates] + weak_heuristic(problem, state)
    
    if problem.is_goal(state):
        return 0
    
    heuristicValue = 0

    for crate in crates:
        if (isDeadlock(state, crate)):
            return math.inf
        heuristicValue += min(manhattan_distance(crate, goal) for goal in state.layout.goals)

    cache[crates] = heuristicValue

    return heuristicValue + weak_heuristic(problem, state)