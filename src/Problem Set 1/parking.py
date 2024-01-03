from typing import Any, Dict, Set, Tuple, List
from problem import Problem
from mathutils import Direction, Point
from helpers.utils import NotImplemented

# Makes sense for the state to be a tuple of points as these basically represent the positions of cars in each parking spot
ParkingState = Tuple[Point]

# An action of the parking problem is a tuple containing an index 'i' and a direction 'd' where car 'i' should move in the direction 'd'.
ParkingAction = Tuple[int, Direction]

# This is the implementation of the parking problem
class ParkingProblem(Problem[ParkingState, ParkingAction]):
    passages: Set[Point]    # A set of points which indicate where a car can be (in other words, every position except walls).
    cars: Tuple[Point]      # A tuple of points where state[i] is the position of car 'i'. 
    slots: Dict[Point, int] # A dictionary which indicate the index of the parking slot (if it is 'i' then it is the lot of car 'i') for every position.
                            # if a position does not contain a parking slot, it will not be in this dictionary.
    width: int              # The width of the parking lot.
    height: int             # The height of the parking lot.

    # This function should return the initial state
    def get_initial_state(self) -> ParkingState:
        # Car positions representing the current state will be our inital state to begin with
        return self.cars
    
    # This function should return True if the given state is a goal. Otherwise, it should return False.
    def is_goal(self, state: ParkingState) -> bool:
        # Let's loop over all the car position in the given state
        for carIndex in range(len(state)):
            # Get the position from the car index
            position = state[carIndex]
            # If this position isn't already in the parking slots dictionary, then the car is not yet parked
            if position not in self.slots:
                return False
            # Else if there is a car parked but it's in a wrong position.. Still not a goal
            elif carIndex != self.slots[position]:
                return False
        # At this point, we have made sure that all cars are in their correct slots, GOAL
        return True
                
    # This function returns a list of all the possible actions that can be applied to the given state
    def get_actions(self, state: ParkingState) -> List[ParkingAction]:
        # Initialize the list of actions that can be applied
        actions: List[ParkingAction] = []
        # Loop over all position in our state
        for carIndex in range(len(state)):
            # Set the current position
            position = state[carIndex]
            # Loop over all possible directions (LEFT/RIGHT/UP/DOWN) which are all the possible movements in which a car can make
            for direction in Direction:
                # Compute the new position of the car after moving in this direction
                newPosition = position + direction.to_vector()
                # If the new position is a passage and if it doesn't collide with another car's position, then we can apply this action
                if newPosition in self.passages and newPosition not in state:
                    # Append to the actions list: The car's number and the direction it moved towards
                    actions.append((carIndex, direction))
        return actions
    
    # This function returns a new state which is the result of applying the given action to the given state
    def get_successor(self, state: ParkingState, action: ParkingAction) -> ParkingState:
        # Set the index as the first element of the action and the direction after transforming the second element to a Point which is to be added to the new state
        carIndex, direction = action[0], action[1].to_vector()
        # Since we can't modify a tuple, we'll cast the state to a list and declare this as our new state which starts from the same previous state
        newState: List[Point] = list(state)
        # Apply the action by moving the car towards the direction computed
        newState[carIndex] += direction
        # Cast the new state back to a tuple and return it
        return tuple(newState)
    
    # This function returns the cost of applying the given action to the given state
    def get_cost(self, state: ParkingState, action: ParkingAction) -> float:
        # Again, name our variables and set them from the action parameter
        carIndex, direction = action[0], action[1].to_vector()
        # Compute the new position by applying the direction movement to this car
        newPosition = state[carIndex] + direction
        # The cost of moving a car is, as stated from the document, 26 moving from A down to 0 (Z), therefore any letter in between would yield 26 - i cost, so A 26, B 25, C 24, etc...
        cost = 26 - carIndex
        # Now if the car has parked wrongly in another employee's car, that's a 100 extra cost to be added on the original movement cost as penalty
        if newPosition in self.slots and self.slots[newPosition] != carIndex:
            cost += 100
        return cost
        
     # Read a parking problem from text containing a grid of tiles
    @staticmethod
    def from_text(text: str) -> 'ParkingProblem':
        passages =  set()
        cars, slots = {}, {}
        lines = [line for line in (line.strip() for line in text.splitlines()) if line]
        width, height = max(len(line) for line in lines), len(lines)
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char != "#":
                    passages.add(Point(x, y))
                    if char == '.':
                        pass
                    elif char in "ABCDEFGHIJ":
                        cars[ord(char) - ord('A')] = Point(x, y)
                    elif char in "0123456789":
                        slots[int(char)] = Point(x, y)
        problem = ParkingProblem()
        problem.passages = passages
        problem.cars = tuple(cars[i] for i in range(len(cars)))
        problem.slots = {position:index for index, position in slots.items()}
        problem.width = width
        problem.height = height
        return problem

    # Read a parking problem from file containing a grid of tiles
    @staticmethod
    def from_file(path: str) -> 'ParkingProblem':
        with open(path, 'r') as f:
            return ParkingProblem.from_text(f.read())
    
