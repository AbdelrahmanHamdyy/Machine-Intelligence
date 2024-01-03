from typing import Tuple
from game import HeuristicFunction, Game, S, A
from helpers.utils import NotImplemented

# TODO: Import any modules you want to use

# All search functions take a problem, a state, a heuristic function and the maximum search depth.
# If the maximum search depth is -1, then there should be no depth cutoff (The expansion should not stop before reaching a terminal state)

# All the search functions should return the expected tree value and the best action to take based on the search results

# This is a simple search function that looks 1-step ahead and returns the action that lead to highest heuristic value.
# This algorithm is bad if the heuristic function is weak. That is why we use minimax search to look ahead for many steps.


def greedy(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    agent = game.get_turn(state)

    terminal, values = game.is_terminal(state)
    if terminal:
        return values[agent], None

    actions_states = [(action, game.get_successor(state, action))
                      for action in game.get_actions(state)]
    value, _, action = max((heuristic(game, state, agent), -index, action)
                           for index, (action, state) in enumerate(actions_states))
    return value, action

# Apply Minimax search and return the game tree value and the best action
# Hint: There may be more than one player, and in all the testcases, it is guaranteed that
# game.get_turn(state) will return 0 (which means it is the turn of the player). All the other players
# (turn > 0) will be enemies. So for any state "s", if the game.get_turn(s) == 0, it should a max node,
# and if it is > 0, it should be a min node. Also remember that game.is_terminal(s), returns the values
# for all the agents. So to get the value for the player (which acts at the max nodes), you need to
# get values[0].


def sort_actions(game: Game[S, A], state: S, heuristic: HeuristicFunction):
    '''
    Sort the actions based on the heuristic values of their successors.

    Parameters:
    - game (Game[S, A]): The game instance.
    - state (S): The current state of the game.
    - heuristic (HeuristicFunction): The heuristic function used to evaluate states.

    Returns:
    - List[A]: The sorted list of actions.
    '''
    # Retrieve the available actions for the given state
    actions = game.get_actions(state)

    # Generate the successors for each action
    successors = [game.get_successor(state, action) for action in actions]

    # Combine actions with their successors into pairs
    action_successor_pairs = zip(actions, successors)

    # Sort action-successor pairs based on the heuristic values of successors
    sorted_action_successor_pairs = sorted(
        action_successor_pairs,
        key=lambda pair: heuristic(game, pair[1], game.get_turn(state)),
        reverse=True
    )

    # Extract and return the sorted actions
    sorted_actions = [action for action, _ in sorted_action_successor_pairs]
    return sorted_actions


def search(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1, alpha=None, beta=None, order=False, expectimax=False) -> Tuple[float, A]:
    """
    Perform a search in the game tree to find the best action with the best value for the current player.
    It has a recursive solve function inside which is used to traverse the search tree and update actions
    on the way. We check inside whether we sort the actions based on successors heuristic values or not based
    on the order boolean parameter. We also have alpha & beta values in case of alpha-beta pruning, they are -inf and inf from the outside calling function so we know that we'll use them, else None. If expectimax is enabled, then we apply the expectimax algorithm. How? By computing chance nodes. For chance nodes, expectiminimax computes the expected value, which is the sum of the value over all outcomes, weighted by the probability of each chance action. Treat other players than the max node as chance nodes to differentiate between expectimax and minimax. If all these parameters are turned off, then we're applying the basic minimax algorithm, where we check for the current player and alternate between min & max nodes. Loop over the actions, get successors and recursively go deeper to get the other successors. Once we've hit the terminal state, we return the value and start computing the best value & action we've found.

    Parameters:
    - game (Game): The game instance.
    - state (S): The current state of the game.
    - heuristic (HeuristicFunction): The heuristic function to evaluate states.
    - max_depth (int): The maximum depth to search in the game tree. If -1, there is no depth cutoff.
    - alpha (float): Alpha value for alpha-beta pruning.
    - beta (float): Beta value for alpha-beta pruning.
    - order (bool): If True, sort actions based on the heuristic function before evaluation.
    - expectimax (bool): If True, use expectimax search instead of minimax.

    Returns:
    Tuple[float, A]: The best value found in the game tree and the corresponding best action.
    """

    # Get the current agent's turn
    player = game.get_turn(state)

    def solve(state: S, depth: int = 0, alpha=None, beta=None) -> Tuple[float, A]:
        # Check if the current state is a terminal state, if yes then returh the corresponding value of the player
        terminal, values = game.is_terminal(state)
        if terminal:
            return values[player], None
        # Here, we check if we've reached the max depth or not and if yes, then we compute the utility at this leaf node to be returned as the stopping condition for the recursion
        elif max_depth != -1 and depth == max_depth:
            return heuristic(game, state, player), None

        # Sort the actions in case we're applying move ordering, else just get them from the game instance given the state
        actions = sort_actions(
            game, state, heuristic) if order else game.get_actions(state)

        if game.get_turn(state) == 0:  # MAX
            # Initialize the best max value as -inf so that nothing comes lower & best action to none as we haven't traversed yet
            best_value, best_action = float('-inf'), None
            # Loop over all possible actions of the current state
            for action in actions:
                # Get the successor state given that this action is applied
                successor = game.get_successor(state, action)
                # Recursively call solve for the successor state by increasing the current depth to be compared with the max depth, meaning that we've reached the end of the search tree and extract the value of it
                child_value, _ = solve(successor, depth + 1, alpha, beta)
                # Check if the obtained value is better than the best value we've got
                if child_value > best_value:
                    # Set both value and action as the maximum ones
                    best_value = child_value
                    best_action = action
                # In case of alpha-beta pruning, then we need to save our alpha as well (since this is a max node)
                if alpha is not None and beta is not None:
                    # A max node develops its decision based on a min node, so we compare the best value we have with the beta (which is the least value a min node can retrieve)
                    if best_value >= beta:
                        # If it's better than it, then we've made sure that no other value will pass the best value of the max node, therefore return it
                        return best_value, best_action
                    # At the same time, we'll store the max alpha to be used by the min node as well
                    alpha = max(alpha, best_value)
            return best_value, best_action
        elif expectimax:  # EXPECTIMAX
            # If it's not the player's turn and expectimax is enabled, calculate the chance node
            # Initialize the expected value by 0
            chance = 0
            for action in actions:
                # Loop over actions and get the successors as explained above
                successor = game.get_successor(state, action)
                # Recursively traverse the tree like the max node but without alpha-beta here
                child_value, _ = solve(successor, depth + 1)
                # Sum the child value obtained and weight it by the reciprocal of the actions length, coming from the probability of each action
                chance += (child_value / len(actions))
            return chance, None
        else:  # MIN
            # Initialize the best value of a min node to be infinity so that we're sure nothing is higher and everything else is minimum
            best_value, best_action = float('inf'), None
            for action in actions:
                # Just like the max node, we'll recursively call solve and reach the terminal state to get the value
                successor = game.get_successor(state, action)
                child_value, _ = solve(successor, depth + 1, alpha, beta)
                # Here, we check if the current child value is lower than the best value we have, then it'll be optimal for the min node to choose it
                if child_value <= best_value:
                    # Surely the value is lower here, so just set the best action and value as such
                    best_value = child_value
                    best_action = action
                # Again, in case of alpha-beta pruning, we'll check for the alpha value
                if alpha is not None and beta is not None:
                    # If the alpha (max value obtained by the max node) is lower than the best value the min node got, then we're 100% sure that no other value is smaller, therefore we don't need to continue and prune the search tree. Return the value and action
                    if best_value <= alpha:
                        return best_value, best_action
                    # Update the beta by comparing the current beta value with the best value, if the current beta is minimum then we won't change it, else set it to the lower best value for the min node
                    beta = min(beta, best_value)
            return best_value, best_action

    # Start the recursive solve function with initial parameters
    return solve(state, 0, alpha, beta)


def minimax(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    return search(game, state, heuristic, max_depth)


def alphabeta(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    return search(game, state, heuristic, max_depth, alpha=float('-inf'), beta=float('inf'))


def alphabeta_with_move_ordering(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    return search(game, state, heuristic, max_depth, alpha=float('-inf'), beta=float('inf'), order=True)


def expectimax(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    return search(game, state, heuristic, max_depth, alpha=None, beta=None, order=False, expectimax=True)
