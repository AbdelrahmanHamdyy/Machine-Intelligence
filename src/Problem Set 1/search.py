from problem import HeuristicFunction, Problem, S, A, Solution

#TODO: Import any modules you want to use
import heapq

# All search functions take a problem and a state
# If it is an informed search function, it will also receive a heuristic function
# S and A are used for generic typing where S represents the state type and A represents the action type

# All the search functions should return one of two possible type:
# 1. A list of actions which represent the path from the initial state to the final state
# 2. None if there is no solution

'''
This is the node structure which we'll use to represent each state. An object of type
node is what'll be pushed to the frontier for the search algorithms that have a cost.
We'll store the cost, state, path and node order. The cost here varies from what algorithm
we're using. So for the UCS for example, it'll be the accumulative path cost up until this
particular node, and so on.. The state is the SokobanState object. We'll also use the terminology
of storing the path of a node inside it so that we can easily return it when we're checking for a goal
and easily append to it whenever we're advancing in our state space. The node order is used to distinguish
same nodes wich have different parents and to avoid conflicts in the heapq comparisons.
The __init__ basically defines the assignment through the constructor and since the heapq won't know
how to compare 2 nodes when popping from the queue, we'll have to override __lt__ as well to specify
how the comparisons should be done. So we'll first start with comparing the cost, if the current node isn't
cheaper than the other one then we return false, and if the costs are equal, we return the node order comparison
else true.
'''
class Node:
    def __init__(self, cost, state, path, order):
        self.cost = cost
        self.state = state
        self.path = path
        self.order = order

    def __lt__(self, otherNode):
        if self.cost < otherNode.cost:
            return True
        elif self.cost == otherNode.cost:
            return self.order < otherNode.order
        else:
            return False

def BreadthFirstSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    # Start with an empty path
    path = []
    
    # This will represent our starting node which will contain the initial state and an empty path
    root = (initial_state, path)
    
    # Before going any further, we'll check if this initial state is the goal state, then we won't need to do anything
    # and just return the empty path we initialized
    if (problem.is_goal(initial_state)):
        return path

    # Add the root node to the top of our frontier
    frontier = [root]

    # Initialize our explored set where The goal of it is to check
    # whether we've already expanded this node or not
    exploredSet = set()

    # Loop over the frontier and return None in case it becomes empty (No Solution)
    while len(frontier) > 0:
        # Pop the first node from the frontier (FIFO)
        node = frontier.pop(0)
        
        # Get the state and path from this node (state, path)
        state, path = node
        
        # Check if we haven't expanded this node before
        if state not in exploredSet:
            # Add the current state to our explored set
            exploredSet.add(state)
            
            # Loop over all possible actions to generate successors
            for action in problem.get_actions(state):
                # Get the new state which will be the result of applying the current action on this state
                child = problem.get_successor(state, action)
                
                # If the new state wasn't explored before and isn't available in our frontier
                if child not in exploredSet and child not in frontier:
                    # Update the path by appending the latest action to the already scanned path
                    updatedPath = path + [action]
                    
                    # [Only for BFS] We undergo a goal test before inserting to the frontier in case this
                    # new state is a goal because if we insert it and pop it in the next iteration, we'll unnecessarily
                    # expand more nodes on the same level which won't lead to a shallower goal so eventually we'll get to this
                    # new state. That's why it's better to check for it from the beginning to decrease the number of expanded nodes.
                    if problem.is_goal(child):
                        return updatedPath
                    
                    # Insert into our frontier
                    frontier.append((child, updatedPath))
    
    # If no solution is found, return None
    return None

def DepthFirstSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    # Start with an empty path
    path = []
    
    # This will represent our starting node which will contain the initial state and an empty path
    root = (initial_state, path)

    # Before going any further, we'll check if this initial state is the goal state, then we won't need to do anything
    # and just return the empty path we initialized
    if problem.is_goal(initial_state):
        return path

    # Add the root node to the top of our frontier
    frontier = [root]

    # Initialize our explored set where The goal of it is to check
    # whether we've already expanded this node or not
    exploredSet = set()
    
    while len(frontier) > 0:
        # Pop the last node from the frontier (LIFO) to simulate depth-first exploration
        node = frontier.pop()
        
        # Get the state and path from this node (state, path)
        state, path = node
        
        # Check if we haven't expanded this node before
        if state not in exploredSet:
            # [Unlike BFS] We check here for a goal before expanding the node to its successors
            # and after we have popped it from the frontier in case a better path is found.
            if problem.is_goal(state):
                return path
            
            # Update our explored set
            exploredSet.add(state)
            
            # Loop over all possible actions
            for action in problem.get_actions(state):
                # Generate the new state from applying this action
                child = problem.get_successor(state, action)

                # If this new state wasn't explored before and isn't already in the frontier
                if child not in exploredSet and child not in frontier:
                    # Insert into our frontier and repeat again
                    frontier.append((child, path + [action]))
    
    # If no solution is found after exploring all nodes, return None
    return None
    
def UniformCostSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    # Initialize the Node variables by first setting the order to zero (Starting node) and an empty path. Since
    # we haven't expanded any nodes yet and we're considering only the root, path cost is zero.
    nodeOrder, cost, path = 0, 0, []
    
    # Form our Node object by passing the necessary parameters including the path cost, state which is the
    # initial state in this case, the empty path and our node order.
    root = Node(cost, initial_state, path, nodeOrder)
    
    # Check for a goal from the initial state before any exploration, you never know!
    if problem.is_goal(initial_state):
        return path
    
    # Insert the root node as the first one to our frontier and intitialize the explored set
    frontier = [root]
    exploredSet = set()
    
    # This dictionary keeps track of all states and their cost so that we can
    # update the cost of a node easily whenever a better path is found
    states = {initial_state: cost}
    
    while len(frontier) > 0:
        # Pop the node with the lowest path cost from the frontier.
        node = heapq.heappop(frontier)
        
        # Get the cost, state and path from the node object
        cost, state, path = node.cost, node.state, node.path
        
        if state not in exploredSet:
            # Check for a goal state once we popped from the frontier
            if problem.is_goal(state):
                return path
            
            # Here we have made sure that the state isn't in the explored set so we add it
            exploredSet.add(state)
            
            # Since we're about to expand this node and get all it's children, pop it as well
            # from the stated dictionary like the frontier
            states.pop(state)
            
            # Loop over all possible actions
            for action in problem.get_actions(state):
                # Increment node oreder
                nodeOrder += 1
                # Get the new state as a result of applying the current action
                child = problem.get_successor(state, action)
                # Compute the new cost by accumulating the old path cost with the new edge cost
                childCost = cost + problem.get_cost(state, action)
                
                # Form the new node object containing the updated cost, the new state, the updated path
                # and the incremented node order
                newNode = Node(
                    childCost, 
                    child, 
                    path + [action], 
                    nodeOrder
                )
                
                # If the child state wasn't explored before and isn't already in the frontier
                if child not in exploredSet and child not in frontier:
                    # Insert it to the frontier and the states dictionary with the state as
                    # the key and its cost as it's value for future comparisons
                    heapq.heappush(frontier, newNode)
                    states.update({child: childCost})
                # If the state is indeed in the frontier, check if it makes a better path (cheaper cost)
                # from the one already in the states, if yes then replace it
                elif child in states and childCost < states.get(child, 0):
                    # Update the child cost with the new cheaper one
                    states[child] = childCost
                    # Find this state in the frontier and replace it, finally heapify and repeat again
                    for existingNode in frontier:
                        if existingNode.state == child:
                            existingNode = newNode
                            heapq.heapify(frontier)
    
    # If no solution is found after exploring all nodes, return None.
    return None

def AStarSearch(problem: Problem[S, A], initial_state: S, heuristic: HeuristicFunction) -> Solution:
    # Initialize variables for path cost, node order, path, and heuristic value from the heuristic function given
    # the initial state and problem
    pathCost, nodeOrder, path, heuristicValue = 0, 0, [], heuristic(problem, initial_state)
    # Calculate the total cost for the initial state using path cost and heuristic value
    cost = pathCost + heuristicValue
    
    # Check if the initial state is the goal state
    if problem.is_goal(initial_state):
        return path
    
    # Create the root node with the initial state and cost
    root = Node(cost, initial_state, path, nodeOrder)
    # Initialize the frontier with the root node
    frontier = [root]
    
    # Initialize the explored set to keep track of visited states
    exploredSet = set()
    # Initialize a dictionary to keep track of the cost of reaching each state
    states = {initial_state: cost}

    # Main loop for A* search
    while len(frontier) > 0:
        # Pop the node with the lowest cost from the priority queue
        node = heapq.heappop(frontier)
        state, path, cost = node.state, node.path, node.cost
        
        # Check if the current state is the goal state
        if problem.is_goal(state):
            return path
        
        # Check if the current state has not been explored
        if state not in exploredSet:
            # Add the current state to the explored set
            exploredSet.add(state)
            # Remove the current state from the dictionary of states
            states.pop(state)

            # Explore successors of the current state
            for action in problem.get_actions(state):
                nodeOrder += 1
                # Generate the child state based on the action
                child = problem.get_successor(state, action)
                # Get the cost of the action
                actionCost = problem.get_cost(state, action)
                # Calculate the new path cost for the child state and subtract the heuristic of the child's ancestor
                # Because heuristic values aren't accumulative
                newPathCost = actionCost + cost - heuristic(problem, state)
                # Calculate the total cost for the child state
                childCost = newPathCost + heuristic(problem, child)
                
                # Create a new node for the child state
                newNode = Node(
                    childCost,
                    child, 
                    path + [action], 
                    nodeOrder
                )
                
                # Check if the child state is neither explored nor in the frontier
                if child not in exploredSet and child not in states:
                    # Add the new node to the frontier and update the dictionary of states with the new child cost
                    heapq.heappush(frontier, newNode)
                    states.update({child: childCost})
                else:
                    # Check if the child state is in the frontier and has a lower cost
                    currentCost = states.get(child, 0)
                    if child in states and childCost < currentCost:
                        # Update the cost in the dictionary and replace the existing node in the frontier
                        states[child] = childCost
                        for existingNode in frontier:
                            if existingNode.state == child:
                                existingNode = newNode
                                # Reorder the frontier based on the updated costs
                                heapq.heapify(frontier)
    
    # Return None if the goal state is not found
    return None

def BestFirstSearch(problem: Problem[S, A], initial_state: S, heuristic: HeuristicFunction) -> Solution:
    # For the greedy-best first search, the cost function here only considers the heuristic value of the current state
    # Initialize variables for node order, cost, and path
    nodeOrder, cost, path = 0, heuristic(problem, initial_state), []
    
    # Create the root node with the initial state, cost, and path
    root = Node(cost, initial_state, path, nodeOrder)
    
    # Check if the initial state is the goal state
    if problem.is_goal(initial_state):
        return path
    
    # Initialize the explored set and insert root node to the frontier
    exploredSet = set()
    frontier = [root]
    
    while len(frontier) > 0:
        # Pop the node with the lowest heuristic cost from the priority queue
        node = heapq.heappop(frontier)
        cost, state, path = node.cost, node.state, node.path

        # Check if the current state is the goal state
        if problem.is_goal(state):
            return path

        # Check if the current state has not been explored
        if state not in exploredSet:
            # Add the current state to the explored set
            exploredSet.add(state)

            # Explore successors of the current state
            for action in problem.get_actions(state):
                # Generate the child state based on the action
                child = problem.get_successor(state, action)
                # Calculate the heuristic cost for the child state
                # We won't need problem.get_cost here since the path cost is irrelevant in this algorithm
                childCost = heuristic(problem, child)
                # Increment the node order
                nodeOrder += 1
                # Create a new node for the child state
                newNode = Node(
                    childCost, 
                    child, 
                    path + [action], 
                    nodeOrder
                )

                # Check if the child state is neither explored nor in the frontier
                if child not in exploredSet and child not in frontier:
                    # Add the new node to the frontier
                    heapq.heappush(frontier, newNode)
    
    # If no solution is found after exploring all nodes, return None.
    return None