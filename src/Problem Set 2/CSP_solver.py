from typing import Any, Dict, List, Optional
from CSP import Assignment, BinaryConstraint, Problem, UnaryConstraint
from helpers.utils import NotImplemented


# This function applies 1-Consistency to the problem.
# In other words, it modifies the domains to only include values that satisfy their variables' unary constraints.
# Then all unary constraints are removed from the problem (they are no longer needed).
# The function returns False if any domain becomes empty. Otherwise, it returns True.
def one_consistency(problem: Problem) -> bool:
    remaining_constraints = []
    solvable = True
    for constraint in problem.constraints:
        if not isinstance(constraint, UnaryConstraint):
            remaining_constraints.append(constraint)
            continue
        variable = constraint.variable
        new_domain = {
            value for value in problem.domains[variable] if constraint.condition(value)}
        if not new_domain:
            solvable = False
        problem.domains[variable] = new_domain
    problem.constraints = remaining_constraints
    return solvable


# This function returns the variable that should be picked based on the MRV heuristic.
# NOTE: We don't use the domains inside the problem, we use the ones given by the "domains" argument
#       since they contain the current domains of unassigned variables only.
# NOTE: If multiple variables have the same priority given the MRV heuristic,
#       we order them in the same order in which they appear in "problem.variables".
def minimum_remaining_values(problem: Problem, domains: Dict[str, set]) -> str:
    _, _, variable = min((len(domains[variable]), index, variable) for index, variable in enumerate(
        problem.variables) if variable in domains)
    return variable


# This function should implement forward checking
# The function is given the problem, the variable that has been assigned and its assigned value and the domains of the unassigned values
# The function should return False if it is impossible to solve the problem after the given assignment, and True otherwise.
# In general, the function should do the following:
#   - For each binary constraints that involve the assigned variable:
#       - Get the other involved variable.
#       - If the other variable has no domain (in other words, it is already assigned), skip this constraint.
#       - Update the other variable's domain to only include the values that satisfy the binary constraint with the assigned variable.
#   - If any variable's domain becomes empty, return False. Otherwise, return True.
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument
#            since they contain the current domains of unassigned variables only.
def forward_checking(problem: Problem, assigned_variable: str, assigned_value: Any, domains: Dict[str, set]) -> bool:
    '''
    Apply forward checking after assigning a value to a variable in the CSP.

    Params:
    - problem (Problem): The CSP problem instance.
    - assigned_variable (str): The variable that has been assigned a value.
    - assigned_value (Any): The value assigned to the variable.
    - domains (Dict[str, set]): The domains of unassigned variables.

    Returns:
    - bool: True if the forward checking is successful (no domain becomes empty), False otherwise.

    Forward checking is a technique used in CSPs to update the domains of unassigned variables after a variable is assigned a value.
    It checks and reduces the domains of neighboring variables based on the constraints of the CSP.

    This function iterates over the binary constraints involving the assigned variable.
    For each constraint, it identifies the other variable involved in the constraint and updates its domain.
    The domain of the other variable is reduced to only include values that satisfy the binary constraint with the assigned variable.

    If, after updating the domains, any variable's domain becomes empty, the function returns False indicating failure.
    Otherwise, it returns True indicating success.

    Important: This function modifies the given domains dictionary in-place.
    '''
    for constraint in problem.constraints:
        # Check if the constraint is binary and involves the assigned variable
        if isinstance(constraint, BinaryConstraint) and assigned_variable in constraint.variables:
            other_variable = constraint.get_other(assigned_variable)

            # Skip the constraint if the other variable is already assigned
            if other_variable not in domains:
                continue

            # Update the domain of the other variable based on the constraint
            new_domain = set()
            for value in domains[other_variable]:
                check_assignment = {
                    assigned_variable: assigned_value,
                    other_variable: value
                }
                if constraint.is_satisfied(check_assignment):
                    new_domain.add(value)

            # If the new domain is empty, return False
            if not new_domain:
                return False

            # Update the domain of the other variable
            domains[other_variable] = new_domain

    # Return True if all updates were successful
    return True


# This function should return the domain of the given variable order based on the "least restraining value" heuristic.
# IMPORTANT: This function should not modify any of the given arguments.
# Generally, this function is very similar to the forward checking function, but it differs as follows:
#   - You are not given a value for the given variable, since you should do the process for every value in the variable's
#     domain to see how much it will restrain the neigbors domain
#   - Here, you do not modify the given domains. But you can create and modify a copy.
# IMPORTANT: If multiple values have the same priority given the "least restraining value" heuristic,
#            order them in ascending order (from the lowest to the highest value).
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument
#            since they contain the current domains of unassigned variables only.
def least_restraining_values(problem: Problem, variable_to_assign: str, domains: Dict[str, set]) -> List[Any]:
    '''
    Calculate the forward impact on neighbors of a variable in the CSP after assigning a value.

    Params:
    - problem (Problem): The CSP problem instance.
    - variable_to_assign (str): The variable for which the forward impact on neighbors is calculated.
    - domains (Dict[str, set]): The domains of variables in the CSP.

    Returns:
    - List[Any]: A sorted list of values for the variable_to_assign based on their impact on neighbors.

    The forward impact is a measure of how much assigning a specific value to a variable affects its neighbors.
    This function calculates the impact on neighbors for each value in the domain of the given variable.

    The impact is calculated by counting how many times the assigned value violates binary constraints with neighbors.
    The values are sorted based on their impact, with the most impacting values appearing first in the list.

    Important: This function does not modify the given domains dictionary.

    Note: This function assumes that the input CSP problem uses BinaryConstraints only.
    '''
    # Create a copy of the domain of the variable to assign
    domains_copy = domains[variable_to_assign].copy()

    # Initialize a dictionary to store the impact on neighbors for each value
    impact_on_neighbors = {value: 0 for value in domains_copy}

    # Iterate over each value in the domain of the variable to assign
    for value in domains[variable_to_assign]:
        # Iterate over each binary constraint in the CSP
        for constraint in problem.constraints:
            # Check if the constraint involves the variable to assign and is a BinaryConstraint
            if variable_to_assign in constraint.variables and isinstance(constraint, BinaryConstraint):
                second_variable = constraint.get_other(variable_to_assign)

                # Check if the other variable in the constraint has a domain
                if second_variable in domains:
                    second_domains_copy = domains[second_variable].copy()

                    # Iterate over each value in the domain of the other variable
                    for second_value in second_domains_copy:
                        # Check if the constraint is satisfied with the given values
                        if constraint.is_satisfied({
                            variable_to_assign: value,
                            second_variable: second_value
                        }):
                            continue  # Constraint is satisfied, no impact

                        # Increment the impact count for the current value
                        impact_on_neighbors[value] += 1

    # Return a sorted list of values based on their impact on neighbors
    return sorted(impact_on_neighbors, key=lambda value: (impact_on_neighbors[value], value))


# This function should solve CSP problems using backtracking search with forward checking.
# The variable ordering should be decided by the MRV heuristic.
# The value ordering should be decided by the "least restraining value" heurisitc.
# Unary constraints should be handled using 1-Consistency before starting the backtracking search.
# This function should return the first solution it finds (a complete assignment that satisfies the problem constraints).
# If no solution was found, it should return None.
# IMPORTANT: To get the correct result for the explored nodes, you should check if the assignment is complete only once using "problem.is_complete"
#            for every assignment including the initial empty assignment, EXCEPT for the assignments pruned by the forward checking.
#            Also, if 1-Consistency deems the whole problem unsolvable, you shouldn't call "problem.is_complete" at all.
def solve(problem: Problem) -> Optional[Assignment]:
    """
    Solve the Constraint Satisfaction Problem (CSP) using backtracking search.

    Params:
    - problem (Problem): The CSP problem instance to solve.

    Returns:
    - Optional[Assignment]: The assignment that satisfies all constraints or None if no solution is found.

    This function initiates the backtracking search to find a solution to the given CSP problem.
    It uses the minimum remaining values heuristic and the least constraining values heuristic for variable ordering.

    Note: The one-consistency check is performed at the beginning, and the forward checking mechanism is employed
    during the search to prune the search space and improve efficiency.
    """
    if not one_consistency(problem):
        return None  # Exit if the CSP is not one-consistent

    assignment = {}  # Initialize an empty assignment

    # Start the backtracking search
    return backtrack_search(problem, problem.domains, assignment)


def backtrack_search(problem: Problem, domains: Dict[str, set], assignment: Assignment) -> Optional[Assignment]:
    """
    Recursive function for backtracking search.

    Params:
    - problem (Problem): The CSP problem instance.
    - domains (Dict[str, set]): The domains of variables in the CSP.
    - assignment (Assignment): The current assignment.

    Returns:
    - Optional[Assignment]: The assignment that satisfies all constraints or None if no solution is found.

    This function explores the solution space through recursive backtracking.
    It selects an unassigned variable based on minimum remaining values heuristic and
    explores possible values based on the least constraining values heuristic.

    Forward checking is used to prune the search space by updating domains during the search.
    """
    # Check if the assignment is complete
    if problem.is_complete(assignment):
        return assignment  # Return the assignment if complete

    # Select the next unassigned variable based on minimum remaining values heuristic
    unassigned_variable = minimum_remaining_values(problem, domains)

    # Order the possible values for the unassigned variable based on least constraining values heuristic
    possible_values = least_restraining_values(
        problem, unassigned_variable, domains)

    # Explore each possible value for the unassigned variable
    for value in possible_values:
        # Get copies of the assignment and domains parameters
        new_assignment = assignment.copy()
        new_domains = domains.copy()

        # Assign the current value to the unassigned variable
        new_assignment[unassigned_variable] = value

        # Since the variable has been assigned, we no longer need to keep its domain
        del new_domains[unassigned_variable]

        # Perform forward checking to prune the search space
        if forward_checking(problem, unassigned_variable, value, new_domains):
            # Go deeper into the search tree after we've made sure nothing is violated
            # According to forward checking so we're safe to proceed to the next assignment
            result = backtrack_search(problem, new_domains, new_assignment)

            if result is not None:
                return result  # Return the result if a solution is found

    return None  # No solution found for the current assignment
