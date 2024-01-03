from typing import Tuple
import re
from CSP import Assignment, Problem, UnaryConstraint, BinaryConstraint

# TODO (Optional): Import any builtin library or define any helper function you want to use

# This is a class to define for cryptarithmetic puzzles as CSPs


class CryptArithmeticProblem(Problem):
    LHS: Tuple[str, str]
    RHS: str

    # Convert an assignment into a string (so that is can be printed).
    def format_assignment(self, assignment: Assignment) -> str:
        LHS0, LHS1 = self.LHS
        RHS = self.RHS
        letters = set(LHS0 + LHS1 + RHS)
        formula = f"{LHS0} + {LHS1} = {RHS}"
        postfix = []
        valid_values = list(range(10))
        for letter in letters:
            value = assignment.get(letter)
            if value is None:
                continue
            if value not in valid_values:
                postfix.append(f"{letter}={value}")
            else:
                formula = formula.replace(letter, str(value))
        if postfix:
            formula = formula + " (" + ", ".join(postfix) + ")"
        return formula

    @staticmethod
    def from_text(text: str) -> 'CryptArithmeticProblem':
        # Given a text in the format "LHS0 + LHS1 = RHS", the following regex
        # matches and extracts LHS0, LHS1 & RHS
        # For example, it would parse "SEND + MORE = MONEY" and extract the
        # terms such that LHS0 = "SEND", LHS1 = "MORE" and RHS = "MONEY"
        pattern = r"\s*([a-zA-Z]+)\s*\+\s*([a-zA-Z]+)\s*=\s*([a-zA-Z]+)\s*"
        match = re.match(pattern, text)
        if not match:
            raise Exception("Failed to parse:" + text)
        LHS0, LHS1, RHS = [match.group(i+1).upper() for i in range(3)]

        problem = CryptArithmeticProblem()
        problem.LHS = (LHS0, LHS1)
        problem.RHS = RHS

        # TODO Edit and complete the rest of this function
        # problem.variables:    should contain a list of variables where each variable is string (the variable name)
        # problem.domains:      should be dictionary that maps each variable (str) to its domain (set of values)
        #                       For the letters, the domain can only contain integers in the range [0,9].
        # problem.constaints:   should contain a list of constraint (either unary or binary constraints).

        # Create a list of unique letters from the cryptarithmetic equation
        letters = list(set(LHS0 + LHS1 + RHS))

        # Initialize the problem variables with these letters, we'll later append to it
        problem.variables = letters

        # Max range for letters is from 0 to 9 since this is just 1 digit so these are the only possible values
        # Initialize these to be the domains of our problem
        problem.domains = {letter: set(range(10)) for letter in letters}

        # Check if 2 terms are equal to each other
        def equal(x, y): return x == y
        # Check if 2 terms are not equal
        def not_equal(x, y): return x != y
        # Check if the given value is not zero
        def non_zero(x): return x != 0
        # Check if the last digit of x is equal to y
        def units(x, y): return x % 10 == y
        # Check if the tens digit is equal to y [Divide by 10 to remove the units then take mod 10 to extract the second to last value]
        def tens(x, y): return (x // 10) % 10 == y
        # Check if the hundreds value is equal to y [Simply divide by 100, since the max length of a new variable is 3]
        def hundreds(x, y): return x // 100 == y
        # This is the main addition equation when solving the cryptarithmetic problem
        # E.g C0 + E + T = 10 * C1 + B
        # Here, we'll have C0ET as one variable so we need to extract each digit to add them together in the left hand side
        # As for the RHS, We also extract each letter in C1B where C1 is the carry (either 0 or 1) so we multiply it by 10 and add it
        # To B, which will represent the carry transfer addition process. Finally equate both sides as this is our goal
        def equation(x, y): return (x // 100) + ((x // 10) %
                                                 10) + (x % 10) == (10 * (y // 10)) + (y % 10)

        # Initialize empty constraints list for the problem
        problem.constraints = []

        # Add unary constraints for the left-most digit of each term, it can't be zero.
        problem.constraints.append(UnaryConstraint(RHS[0], non_zero))
        problem.constraints.append(UnaryConstraint(LHS0[0], non_zero))
        problem.constraints.append(UnaryConstraint(LHS1[0], non_zero))

        # All assigned values for the letters should be different from each other
        for i in range(len(letters)):
            for j in range(i + 1, len(letters)):
                problem.constraints.append(BinaryConstraint(
                    (letters[i], letters[j]), not_equal))

        # Our first carry will have a zero index (This will be increased ofcourse)
        carry_num = 0
        # We'll loop on the output length, because we're sure that it's the longest term so all digits will be covered
        for i in range(len(RHS)):
            # Index LHS0, LHS1 and RHS from the end  (Reversed) as if we're adding normally
            l0_idx = len(LHS0) - i - 1
            l1_idx = len(LHS1) - i - 1
            r_idx = len(RHS) - i - 1

            # Check if a letter exists for both of the LHS terms, if not then set it to None for checking in the conditions below
            letter_lhs0 = LHS0[l0_idx] if l0_idx >= 0 else None
            letter_lhs1 = LHS1[l1_idx] if l1_idx >= 0 else None
            letter_rhs = RHS[r_idx]

            # Check if both LHS letters exist
            if letter_lhs0 is not None and letter_lhs1 is not None:
                # Assign a carry for the equation, append it to the variables and set its domain to only have the values 0 or 1
                # If this the first carry, then it can't be zero so it's domain will by just 0
                first_carry = 'C' + str(carry_num)
                problem.variables.append(first_carry)
                problem.domains[first_carry] = {
                    0} if carry_num == 0 else set(range(2))

                # Append the LHS terms including the carry in a single variable. We do this because we can only deal with binary constraints and not ternary or quadraple. This method groups multiple variables and uses it for the desired constraints
                L = first_carry + letter_lhs0 + letter_lhs1

                # Temporarily set the RHS with the corresponding output letter and we'll see if it changes or not
                R = letter_rhs

                # Append the LHS of the equation to the variables list and set its domain to be from 0 to 199. The carry being left-most means that it can only take 0 & 1 while the other 2 letters from 0 to 9 so we've only got up to and not including 200 in this case
                problem.variables.append(L)
                problem.domains[L] = set(range(200))

                # Add binary constraints for each digit in the assigned value, meaning that the hundreds digits must equal the carry, the tens digit to LHS0 and the units digit to LHS1 as concatenated in L above in the same order
                problem.constraints.append(
                    BinaryConstraint((L, first_carry), hundreds))
                problem.constraints.append(
                    BinaryConstraint((L, letter_lhs0), tens))
                problem.constraints.append(
                    BinaryConstraint((L, letter_lhs1), units))

                # If we haven't reached the leftmost digit of the RHS yet
                if i < len(RHS) - 1:
                    # This means that a carry is still possible to be propagated
                    # Create this carry after increasing the carry_num variable.
                    # Append this new carry to the variables list and set its domain to {0, 1}
                    carry_num += 1
                    second_carry = 'C' + str(carry_num)
                    problem.variables.append(second_carry)
                    problem.domains[second_carry] = set(range(2))

                    # Since we have a carry, the output is no longer just the RHS letter. To form the equation we'll need to concatenate the carry we've just created with the RHS letter but making sure that this carry is on the left to satisfy the addition
                    # Append the new variable R to variables and set its domain from 0 to 19 inclusive, same methodology followed here.
                    R = second_carry + letter_rhs
                    problem.variables.append(R)
                    problem.domains[R] = set(range(20))

                    # Apply the tens and units binary constraints on the output variable
                    problem.constraints.append(
                        BinaryConstraint((R, second_carry), tens))
                    problem.constraints.append(
                        BinaryConstraint((R, letter_rhs), units))

                # Here, we have our LHS and RHS of the equation ready, we just need to apply it.
                problem.constraints.append(BinaryConstraint((L, R), equation))

            # If both LHS letters are non-existent
            elif letter_lhs0 is None and letter_lhs1 is None:
                # Therefore, we're sure that the only thing that comes down as output is the carry from the previous iteration which is just carry_num without incrementing it
                carry = 'C' + str(carry_num)

                # Then just make sure that the carry and the RHS letter are equal, thanks to our lambda function.
                problem.constraints.append(
                    BinaryConstraint((carry, letter_rhs), equal))

            # If only one of the LHS letters exist
            elif (letter_lhs0 is None and letter_lhs1 is not None) or (letter_lhs0 is not None and letter_lhs1 is None):
                # Assign a carry for the lhs and rhs of the equation, with the second being the increment of the first
                first_carry = 'C' + str(carry_num)
                carry_num += 1
                second_carry = 'C' + str(carry_num)

                # Get the letter which is not none from LHS0 and LHS1
                letter = letter_lhs0 if letter_lhs0 is not None else letter_lhs1

                # Add both carries to the variables and set their domains {0, 1}
                problem.variables.append(first_carry)
                problem.variables.append(second_carry)
                problem.domains[first_carry] = set(range(2))
                problem.domains[second_carry] = set(range(2))

                # Having only binary constraints, we'll again need to group both carries and the lhs,rhs letters into one and append them to the variables list
                L = first_carry + letter
                R = second_carry + letter_rhs
                problem.variables.append(L)
                problem.variables.append(R)

                # Since both are only 2 terms, their domains will be from 0 to 19, with the left digit being the carry
                problem.domains[L] = set(range(20))
                problem.domains[R] = set(range(20))

                # Apply tens and units binary constraints to L and R to ensure each digit is in place of the correct variable
                problem.constraints.append(BinaryConstraint(
                    (L, first_carry), tens))
                problem.constraints.append(BinaryConstraint(
                    (L, letter), units))
                problem.constraints.append(BinaryConstraint(
                    (R, second_carry), tens))
                problem.constraints.append(BinaryConstraint(
                    (R, letter_rhs), units))

                # Finally, apply the equation to L and R just like the previous condition
                problem.constraints.append(
                    BinaryConstraint((L, R), equation))

        return problem

    # Read a cryptarithmetic puzzle from a file
    @staticmethod
    def from_file(path: str) -> "CryptArithmeticProblem":
        with open(path, 'r') as f:
            return CryptArithmeticProblem.from_text(f.read())


def print_lengths(problem: CryptArithmeticProblem):
    print(len(problem.variables))
    print(len(problem.domains))
    print(len(problem.constraints))


def print_problem(problem: CryptArithmeticProblem):
    print(problem.variables)
    print(problem.domains)
    print(problem.constraints)


def print_unary_constraints(problem: CryptArithmeticProblem):
    for constraint in problem.constraints:
        if isinstance(constraint, UnaryConstraint):
            print(constraint.variable, constraint.condition)


def print_binary_constraints(problem: CryptArithmeticProblem):
    for constraint in problem.constraints:
        if (isinstance(constraint, BinaryConstraint)):
            print(constraint.variables, constraint.condition)
