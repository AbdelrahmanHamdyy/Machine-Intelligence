from typing import Dict, List, Optional, Set, Tuple
from mdp import MarkovDecisionProcess
from environment import Environment
from mathutils import Point, Direction
from helpers.mt19937 import RandomGenerator
from helpers.utils import NotImplemented
import json
from dataclasses import dataclass

"""
Environment Description:
    The snake is a 2D grid world where the snake can move in 4 directions.
    The snake always starts at the center of the level (floor(W/2), floor(H/2)) having a length of 1 and moving LEFT.
    The snake can wrap around the grid.
    The snake can eat apples which will grow the snake by 1.
    The snake can not eat itself.
    You win if the snake body covers all of the level (there is no cell that is not occupied by the snake).
    You lose if the snake bites itself (the snake head enters a cell occupied by its body).
    The action can not move the snake in the opposite direction of its current direction.
    The action can not move the snake in the same direction 
        i.e. (if moving right don't give an action saying move right).
    Eating an apple increases the reward by 1.
    Winning the game increases the reward by 100.
    Losing the game decreases the reward by 100.
"""

# IMPORTANT: This class will be used to store an observation of the snake environment


@dataclass(frozen=True)
class SnakeObservation:
    snake: Tuple[Point]     # The points occupied by the snake body
    # where the head is the first point and the tail is the last
    direction: Direction    # The direction that the snake is moving towards
    # The location of the apple. If the game was already won, apple will be None
    apple: Optional[Point]


class SnakeEnv(Environment[SnakeObservation, Direction]):

    rng: RandomGenerator  # A random generator which will be used to sample apple locations

    snake: List[Point]
    direction: Direction
    apple: Optional[Point]

    def __init__(self, width: int, height: int) -> None:
        super().__init__()
        assert width > 1 or height > 1, "The world must be larger than 1x1"
        self.rng = RandomGenerator()
        self.width = width
        self.height = height
        self.snake = []
        self.direction = Direction.LEFT
        self.apple = None

    def generate_random_apple(self) -> Point:
        """
        Generates and returns a random apple position which is not on a cell occupied 
        by the snake's body.
        """
        snake_positions = set(self.snake)
        possible_points = [Point(x, y)
                           for x in range(self.width)
                           for y in range(self.height)
                           if Point(x, y) not in snake_positions
                           ]
        return self.rng.choice(possible_points)

    def reset(self, seed: Optional[int] = None) -> Point:
        """
        Resets the Snake environment to its initial state and returns the starting state.
        Args:
            seed (Optional[int]): An optional integer seed for the random
            number generator used to generate the game's initial state.

        Returns:
            The starting state of the game, represented as a Point object.
        """
        if seed is not None:
            # Initialize the random generator using the seed
            self.rng.seed(seed)

        # Initialize the snake at the center of the level with length 1
        center_x = self.width // 2
        center_y = self.height // 2
        self.snake = [Point(center_x, center_y)]

        self.direction = Direction.LEFT  # Set the snake's direction

        # Generate a random apple position
        self.apple = self.generate_random_apple()

        return SnakeObservation(tuple(self.snake), self.direction, self.apple)

    def actions(self) -> List[Direction]:
        """
        Returns a list of the possible actions that can be taken from the current state of the Snake game.
        Returns:
            A list of Directions, representing the possible actions that can be taken from the current state.

        """
        # TODO add your code here
        # a snake can wrap around the grid
        # NOTE: The action order does not matter

        # Initialize the list of possible actions with Direction.NONE (no movement)
        actions = [Direction.NONE]

        # Get the current direction of the snake
        current_direction = self.direction

        # Iterate over all possible directions
        for direction in Direction:
            # Skip the direction that matches the current direction
            if direction == current_direction:
                continue

            # Skip directions that are opposite (in the same axis) to the current direction
            if (direction.value % 2 == current_direction.value % 2):
                continue

            # Add the valid direction to the list of possible actions
            actions.append(direction)

        # Return the list of possible actions
        return actions

    def step(self, action: Direction) -> \
            Tuple[SnakeObservation, float, bool, Dict]:
        """
        Updates the state of the Snake game by applying the given action.

        Args:
            action (Direction): The action to apply to the current state.

        Returns:
            A tuple containing four elements:
            - next_state (SnakeObservation): The state of the game after taking the given action.
            - reward (float): The reward obtained by taking the given action.
            - done (bool): A boolean indicating whether the episode is over.
            - info (Dict): A dictionary containing any extra information. You can keep it empty.
        """
        # TODO Complete the following function

        # Determines whether the game has ended or not (loss or win)
        done = False

        # The reward given to the snake. This is accumulated according to the resulting actions
        reward = 0

        # Get the head and tail of the snake
        head = self.snake[0]
        tail = self.snake[-1]

        # This boolean is true if the snake has collided with an apple
        is_apple = False

        # Set the width and height of our map
        WIDTH = self.width
        HEIGHT = self.height

        # Update snake direction if the action is not NONE
        if action != Direction.NONE:
            self.direction = action

        # Calculate the next position
        direction_vector = self.direction.to_vector()
        x, y = head + direction_vector

        # Wrap around the grid if the snake is out of bounds
        # If it went far left, then it'll come back far right
        if x < 0:
            x = self.width - 1
        # Same thing with the far right out of bounds, we'll see it on the left while keeping the same y
        elif x >= self.width:
            x = 0
        # If it's downwards below the map, then it should come back from above while maintaining the x
        if y < 0:
            y = self.height - 1
        # Same with if it's beyond the top, it'll resume from the bottom
        elif y >= self.height:
            y = 0

        # Get the next position
        next_step = Point(x, y)

        # Handle scenarios based on the next position
        if next_step == self.apple:  # If the next position is the apple
            self.snake.insert(0, next_step)
            # The snake ate an apple.. That's a +1 reward
            reward += 1
            # Set this boolean to true as it will decide whether to generate another apple or not
            is_apple = True
        elif next_step in self.snake:  # If the next position is the snake body
            done = True
            # Lost the game as the snake collided with itself
            reward -= 100
        else:  # If the next position is empty
            # Remove the tail and add the next position to the snake's head
            self.snake.pop()
            self.snake.insert(0, next_step)

        # Check if the snake covers all the grid
        # Get the snake length
        SNAKE_LENGTH = len(self.snake)
        # Area of a rectangle is width * height
        GRID_AREA = WIDTH * HEIGHT
        # If the length of the snake is the same as the whole area, then no more moves are possible and the game ends.
        if SNAKE_LENGTH == GRID_AREA:
            done = True
            # The whole grid is covered, game is won!
            reward += 100

        # Generate a new apple if it was eaten and the game is not over
        # The check for whether the game is over or not is important because if it is, then there's no point to generate an apple.
        if is_apple and not done:
            self.apple = self.generate_random_apple()

        # Finally set the observation from the state, direction and apple
        observation = SnakeObservation(
            tuple(self.snake), self.direction, self.apple)

        # Return the next state, reward, done, and an empty dictionary
        return observation, reward, done, {}

    ###########################
    #### Utility Functions ####
    ###########################

    def render(self) -> None:
        # render the snake as * (where the head is an arrow < ^ > v) and the apple as $ and empty space as .
        for y in range(self.height):
            for x in range(self.width):
                p = Point(x, y)
                if p == self.snake[0]:
                    char = ">^<v"[self.direction]
                    print(char, end='')
                elif p in self.snake:
                    print('*', end='')
                elif p == self.apple:
                    print('$', end='')
                else:
                    print('.', end='')
            print()
        print()

    # Converts a string to an observation
    def parse_state(self, string: str) -> SnakeObservation:
        snake, direction, apple = eval(str)
        return SnakeObservation(
            tuple(Point(x, y) for x, y in snake),
            self.parse_action(direction),
            Point(*apple)
        )

    # Converts an observation to a string
    def format_state(self, state: SnakeObservation) -> str:
        snake = tuple(tuple(p) for p in state.snake)
        direction = self.format_action(state.direction)
        apple = tuple(state.apple)
        return str((snake, direction, apple))

    # Converts a string to an action
    def parse_action(self, string: str) -> Direction:
        return {
            'R': Direction.RIGHT,
            'U': Direction.UP,
            'L': Direction.LEFT,
            'D': Direction.DOWN,
            '.': Direction.NONE,
        }[string.upper()]

    # Converts an action to a string
    def format_action(self, action: Direction) -> str:
        return {
            Direction.RIGHT: 'R',
            Direction.UP:    'U',
            Direction.LEFT:  'L',
            Direction.DOWN:  'D',
            Direction.NONE:  '.',
        }[action]
