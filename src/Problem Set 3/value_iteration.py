from typing import Dict, Optional
from agents import Agent
from environment import Environment
from mdp import MarkovDecisionProcess, S, A
import json
from helpers.utils import NotImplemented

# This is a class for a generic Value Iteration agent


class ValueIterationAgent(Agent[S, A]):
    mdp: MarkovDecisionProcess[S, A]  # The MDP used by this agent for training
    utilities: Dict[S, float]  # The computed utilities
    # The key is the string representation of the state and the value is the utility
    discount_factor: float  # The discount factor (gamma)

    def __init__(self, mdp: MarkovDecisionProcess[S, A], discount_factor: float = 0.99) -> None:
        super().__init__()
        self.mdp = mdp
        # We initialize all the utilities to be 0
        self.utilities = {state: 0 for state in self.mdp.get_states()}
        self.discount_factor = discount_factor

    # Given a state, compute its utility using the bellman equation
    # if the state is terminal, return 0
    def compute_bellman(self, state: S) -> float:
        if not self.mdp.is_terminal(state):
            actions = self.mdp.get_actions(state)
            max_val = float('-inf')
            for action in actions:
                # Initialize the sum of action values
                action_value = 0

                # Iterate over all possible next states for the given action
                for next_state in self.mdp.get_successor(state, action):
                    # Calculate the transition probability for the current (state, action, next_state) triplet
                    transition_prob = self.mdp.get_successor(state, action)[
                        next_state]

                    # Calculate the immediate reward for the current (state, action, next_state) triplet
                    immediate_reward = self.mdp.get_reward(
                        state, action, next_state)

                    # Calculate the discounted future utility for the next_state
                    discounted_utility = self.discount_factor * \
                        self.utilities[next_state]

                    # Sum up the product of transition probability and the sum of immediate reward and discounted future utility
                    action_value += transition_prob * \
                        (immediate_reward + discounted_utility)

                # Record the max value for each action
                max_val = max(max_val, action_value)

            # Return max
            return max_val
        else:
            # We've reached terminal state
            return 0

    # Applies a single utility update
    # then returns True if the utilities has converged (the maximum utility change is less or equal the tolerance)
    # and False otherwise
    def update(self, tolerance: float = 0) -> bool:
        # Get the Markov Decision Process current states
        states = self.mdp.get_states()
        # Compute new utilities using the Bellman equation for each state
        new_utilities = {state: self.compute_bellman(
            state) for state in states}

        # Calculate the maximum change in utilities between the current and new values
        max_change = max(abs(
            new_utilities[state] - self.utilities[state]) for state in states)

        # Update the current utilities with the new values
        self.utilities = new_utilities

        # Check if the maximum change is below or equal to the specified tolerance
        return True if max_change <= tolerance else False

    # This function applies value iteration starting from the current utilities stored in the agent and stores the new utilities in the agent
    # NOTE: this function does incremental update and does not clear the utilities to 0 before running
    # In other words, calling train(M) followed by train(N) is equivalent to just calling train(N+M)
    def train(self, iterations: Optional[int] = None, tolerance: float = 0) -> int:
        """Train the value iteration algorithm on the given Markov Decision Process."""
        iter = 0
        # Check if the maximum number of iterations is specified and if it has been reached
        while iterations is None or iter < iterations:
            # Increment the number of iterations
            iter += 1

            # Check if the update meets the convergence tolerance
            if self.update(tolerance):
                break
        # Return the total number of iterations performed
        return iter

    # Given an environment and a state, return the best action as guided by the learned utilities and the MDP
    # If the state is terminal, return None
    def act(self, env: Environment[S, A], state: S) -> A:
        """
        Find the best action for the given state using the current utility estimates.

        Args:
            state: The current state in the Markov Decision Process.

        Returns:
            The best action for the given state.
        """
        # Check if the state is terminal; return None if terminal
        if self.mdp.is_terminal(state):
            return None
        else:
            # Get all available actions for the given state
            available_actions = self.mdp.get_actions(state)

            # Find the action that maximizes the expected utility
            best_action = max(
                available_actions,
                key=lambda action: self.calculate_expected_utility(
                    state, action)
            )

        return best_action

    def calculate_expected_utility(self, state: S, action: A):
        """
        Calculate the expected utility for a specific action in the given state.

        Args:
            state: The current state in the Markov Decision Process.
            action: The action to evaluate.

        Returns:
            The expected utility for the specified action in the given state.
        """
        # Calculate the expected utility using the Bellman equation
        expected_utility = 0

        # Get the successor states and their probabilities for the given action
        successor_states = self.mdp.get_successor(state, action)

        # Iterate over each successor state
        for next_state in successor_states:
            # Get the probability of transitioning to the current successor state
            successor_probability = successor_states[next_state]

            # Get the immediate reward for taking the specified action in the current state
            successor_reward = self.mdp.get_reward(state, action, next_state)

            # Get the current estimated utility of the successor state
            successor_utility = self.utilities[next_state]

            # Calculate the discounted expected utility for the current successor state
            expected_utility += successor_probability * \
                (successor_reward + self.discount_factor * successor_utility)

        # Return the calculated expected utility
        return expected_utility

    # Save the utilities to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            utilities = {self.mdp.format_state(
                state): value for state, value in self.utilities.items()}
            json.dump(utilities, f, indent=2, sort_keys=True)

    # loads the utilities from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            utilities = json.load(f)
            self.utilities = {self.mdp.parse_state(
                state): value for state, value in utilities.items()}
