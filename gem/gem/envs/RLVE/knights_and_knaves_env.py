from typing import Any, Dict, Optional, SupportsFloat, Tuple, List
import numpy as np
import re
import copy
import itertools

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


"""Knight and Knave problems.

Each person can have the following (recursive) statements:
    - assertion: (telling-truth, i), (lying, i)
    - negation: (not, statement)
    - conjunction: (and, statement1, statement2), could support more than 2
    - disjunction: (or, statement1, statement2), could support more than 2
    - implication: (->, statement1, statement2)
    - equivalence: (<=>, statement1, statement2)

Original link: https://github.com/AlphaPav/mem-kk-logic/blob/main/data_prep/lib_kk.py
"""

####################################################################################
# Problem Solving
####################################################################################
def find_solution(statements):
    """Find solutions given a list of statements."""
    n_people = len(statements)
    single_statement = ('and',) + tuple(('<=>', ('telling-truth', i), statements[i])
                                        for i in range(len(statements)))
    solutions = []
    for assignments in itertools.product([True, False], repeat=n_people):
        if test_satisfiability(single_statement, assignments):
            solutions.append(assignments)
    return solutions


def test_satisfiability(statement, assignments):
    """Recursive satisfiability testing."""
    if statement[0] == 'telling-truth':
        return assignments[statement[1]]
    if statement[0] == 'lying':
        return not assignments[statement[1]]
    if statement[0] == 'not':
        return not test_satisfiability(statement[1], assignments)
    if statement[0] == 'and':
        return np.all([test_satisfiability(statement[i], assignments) for i in range(1, len(statement))])
    if statement[0] == 'or':
        return np.any([test_satisfiability(statement[i], assignments) for i in range(1, len(statement))])
    if statement[0] == '->':
        val1 = test_satisfiability(statement[1], assignments)
        val2 = test_satisfiability(statement[2], assignments)
        return (not val1) or val2
    if statement[0] == '<=>':
        val1 = test_satisfiability(statement[1], assignments)
        val2 = test_satisfiability(statement[2], assignments)
        return (val1 and val2) or ((not val1) and (not val2))
    raise KeyError(f'Unknown statement: {statement}')


####################################################################################
# Problem Sampling
####################################################################################
class KKProblemSamplerEnv:
    """Problem Sampler for Knights and Knaves.

    Args:
      rand_seed: seed for random number generators.
      n_people: number of people for K&K problems.
      depth_constraint: the max depth of each person's statement. The depth refer to the level of
          recursion of operators such as 'and', 'or', etc. Increasing the depth would allow
          increasing the difficulty. Though currently the automatic formatting of the problems
          into natural languages does not support depth more than 2.
      width_constraint: the max width (number of branches in operators such as 'and', 'or') of each
          person's statement.
    """

    def __init__(self, rand_seed: int, n_people: int, depth_constraint: int = 2, width_constraint: int = 2):
        self.rng = np.random.default_rng(rand_seed)
        self.rng_wrong = np.random.default_rng(rand_seed + 1)
        self.n_people = n_people
        self.depth_constraint = depth_constraint
        self.width_constraint = width_constraint

    def sample(self):
        """Sample a single Knights and Knaves problem."""
        statements = tuple(self._sample_statement(person_id, self.depth_constraint)
                           for person_id in range(self.n_people))
        return self._immutable_statements(statements)

    def sample_valid_problems(
        self,
        n_problems: int,
        max_retry: int = 1000,
        skip_no_solution: bool = True,
        skip_multiple_solutions: bool = True
    ):
        """Sample valid (has 1 unique solution) problems.

        Args:
          n_problems: how many problems to sample.
          max_retry: max number of retries per problem before giving up.
          skip_no_solution: skip problems without a valid solution.
          skip_multiple_solutions: skip problems with more than one solutions.

        Returns:
          A list of problems, each a dict with keys 'statements' and 'solution'.
        """
        problems = []
        unique_statements = set()
        for i_problem in range(n_problems):
            success = False
            for _ in range(max_retry):
                statements = self.sample()
                if statements in unique_statements:
                    continue
                solutions = find_solution(statements)
                if len(solutions) == 0 and skip_no_solution:
                    continue
                if len(solutions) > 1 and skip_multiple_solutions:
                    continue
                sol = solutions[0] if len(solutions) > 0 else None
                problems.append({'statements': statements, 'solution': sol, 'all_solutions': solutions})
                unique_statements.add(statements)
                success = True
                break
            if not success:
                raise RuntimeError(f'Failed to generate a valid problem after {max_retry} retries.')
        return problems

    def sample_flipped_solution(self, solution):
        length_of_solution = len(solution)
        num_to_perturb = self.rng_wrong.integers(1, length_of_solution)
        indices_to_perturb = list(self.rng_wrong.choice(list(range(length_of_solution)), size=num_to_perturb, replace=False))
        perturbed_solution = tuple(
            not solution[i] if i in indices_to_perturb else solution[i]
            for i in range(length_of_solution)
        )
        return perturbed_solution

    def sample_invalid_problems(
        self,
        n_problems: int,
        max_retry: int = 1000,
        skip_no_solution: bool = True,
        skip_multiple_solutions: bool = True
    ):
        """Sample valid (has 1 unique solution) problems and then perturb the solution."""
        problems = []
        unique_statements = set()
        for i_problem in range(n_problems):
            success = False
            for _ in range(max_retry):
                statements = self.sample()
                if statements in unique_statements:
                    continue
                solutions = find_solution(statements)
                if len(solutions) == 0 and skip_no_solution:
                    continue
                if len(solutions) > 1 and skip_multiple_solutions:
                    continue
                sol = solutions[0] if len(solutions) > 0 else None
                perturbed_sol = self.sample_flipped_solution(sol)
                problems.append({'statements': statements, 'solution': perturbed_sol, 'all_solutions': [perturbed_sol]})
                unique_statements.add(statements)
                success = True
                break
            if not success:
                raise RuntimeError(f'Failed to generate a valid problem after {max_retry} retries.')
        return problems

    def perturb_problems(self, problems, max_retry: int = 1000, perturb_type: str = 'statement',
                         num_perturb: int = 1):
        """Perturb the problems (generated by this sampler)."""
        return [self._perturb_problem(p, max_retry=max_retry, perturb_type=perturb_type, num_perturb=num_perturb)
                for p in problems]

    def _perturb_problem(self, problem, max_retry: int, perturb_type: str, num_perturb: int):
        assert len(problem['statements']) == self.n_people
        results_set = set()
        results_list = []
        for _ in range(max_retry):
            statements = self._copy_statements_as_mutable(problem['statements'])
            if perturb_type == 'statement':
                person = self.rng.integers(0, self.n_people)
                statements[person] = self._sample_statement(person, depth_constraint=self.depth_constraint)
            elif perturb_type == 'leaf':
                person = self.rng.integers(0, self.n_people)
                idx = person
                container = statements
                while not self._is_leaf_node(container[idx]):
                    container = container[idx]
                    idx = self.rng.integers(1, len(container))
                assert self._is_leaf_node(container[idx])
                container[idx] = self._sample_statement(person, depth_constraint=1)
            statements = self._immutable_statements(statements)
            if len(set([statements, problem['statements']])) <= 1:
                continue
            solutions = find_solution(statements)
            if len(solutions) != 1:
                continue
            if len(set([solutions[0], problem['solution']])) <= 1:
                continue
            if statements in results_set:
                continue
            results_set.add(statements)
            results_list.append({'statements': statements, 'solution': solutions[0]})
            if len(results_list) >= num_perturb:
                break
        if len(results_list) == 0:
            return [None]
        return results_list

    def _copy_statements_as_mutable(self, statements):
        """Make a deep copy of the statements of a problem, turning the tuples into (mutable) lists."""
        statements = copy.deepcopy(statements)

        def _make_mutable(x):
            if isinstance(x, tuple):
                return [_make_mutable(child) for child in x]
            return x

        return [_make_mutable(s) for s in statements]

    def _immutable_statements(self, mutable_statements):
        """Change list back to tuples."""
        def _make_immutable(x):
            if isinstance(x, (list, tuple)):
                return tuple(_make_immutable(child) for child in x)
            if isinstance(x, np.str_):
                return str(x)
            if isinstance(x, np.int64):
                return int(x)
            return x

        return tuple(_make_immutable(s) for s in mutable_statements)

    def _is_leaf_node(self, statement):
        if statement[0] in ['telling-truth', 'lying']:
            return True
        return False

    def _sample_statement(self, person_id: int, depth_constraint: int):
        """Sample a single statement."""
        dice = self.rng.integers(0, 6)
        if depth_constraint == 1 or dice == 0:
            while True:
                knight_or_knave = self.rng.choice(['telling-truth', 'lying'])
                person = self.rng.integers(0, self.n_people)
                if not (knight_or_knave == 'lying' and person == person_id):
                    return (knight_or_knave, person)

        if dice == 1:
            return ('not', self._sample_statement(person_id, depth_constraint - 1))
        if dice in [2, 3]:
            operator = ['and', 'or'][dice - 2]
            n_substatements = self.rng.integers(2, self.width_constraint + 1)
            return (operator,) + self._sample_substatements(person_id, depth_constraint, n_substatements)
        if dice in [4, 5]:
            operator = ['->', '<=>'][dice - 4]
            return (operator,) + self._sample_substatements(person_id, depth_constraint, 2)

    def _sample_substatements(self, person_id: int, depth_constraint: int, count: int, dedup: bool = True):
        """Sample substatements for an operator."""
        sub_statements = []
        dedup_set = set()
        while True:
            stmt = self._sample_statement(person_id, depth_constraint - 1)
            if dedup:
                if stmt in dedup_set:
                    continue
                dedup_set.add(stmt)

            sub_statements.append(stmt)
            if len(sub_statements) == count:
                break
        return tuple(sub_statements)


####################################################################################
# Problem Formatting in natural language
####################################################################################
COMMON_NAMES = ['Emma', 'Liam', 'Olivia', 'Noah', 'Ava', 'Ethan', 'Sophia',
                'Mason', 'Isabella', 'William', 'Mia', 'James', 'Charlotte',
                'Benjamin', 'Amelia', 'Lucas', 'Harper', 'Henry', 'Evelyn',
                'Alexander', 'Abigail', 'Michael', 'Emily', 'Daniel', 'Elizabeth',
                'Jacob', 'Sofia', 'Logan', 'Avery', 'Jackson', 'Ella', 'Sebastian',
                'Scarlett', 'Jack', 'Grace', 'Aiden', 'Chloe', 'Owen', 'Victoria',
                'Samuel', 'Riley', 'Matthew', 'Aria', 'Joseph', 'Lily', 'Luke',
                'Aurora', 'David', 'Zoey', 'Oliver', 'Penelope']
UNCOMMON_NAMES = [
    'Zephyr', 'Elowen', 'Caspian', 'Isolde', 'Osiris', 'Vesper', 'Thaddeus', 'Ondine',
    'Lysander', 'Xanthe', 'Oberon', 'Calliope', 'Leander', 'Eulalia', 'Florian', 'Forsythe',
    'Nephele', 'Peregrine', 'Ianthe', 'Lazarus', 'Elodie', 'Cillian', 'Ottoline', 'Evander',
    'Saffron', 'Caius', 'Zora', 'Cyprian', 'Amaryllis', 'Theron', 'Perdita', 'Ignatius',
    'Zephyrine', 'Balthazar', 'Melisande', 'Zinnia', 'Sylvester', 'Cosima', 'Leocadio',
    'Percival', 'Oceane', 'Evanthe', 'Zenobia', 'Eurydice', 'Quillan', 'Aeronwen',
    'Thorsten', 'Xiomara', 'Zephyrus', 'Ysolde'
]

KNIGHT_KNAVE_PAIRS = [
    ['a pioneer', 'a laggard'],
    ['a saint', 'a sinner'],
    ['a hero', 'a villain'],
    ['an angel', 'a devil'],
    ['an altruist', 'an egoist'],
    ['a sage', 'a fool'],
]
PREFIX = ('A very special island is inhabited only by {knight}s and {knave}s. ' +
          '{Knight}s always tell the truth, and {knave}s always lie. ')
POSTFIX = 'So who is {a_knight} and who is {a_knave}?'
TEMPLATES = [
    '{name} said that {content}.',
    '{name} told you that {content}.',
    '{name} said, "{content}."',
    '{name} stated, "{content}".',
    'According to {name}, "{content}".',
    'In {name}\'s words: "{content}".',
    '{name} remarked, "{content}".',
    '"{content}," {name} declared.',
    '{name} was heard saying, "{content}".',
    '{name} expressed that {content}.',
    '"{content}" - {name}.',
    'As {name} put it, "{content}".',
    '{name} asserted: "{content}".',
    '"{content}," {name} mentioned.',
    '{name} commented, "{content}".',
    'In a statement by {name}: "{content}".',
    '{name} noted, "{content}".',
    '"{content}," {name} claimed.',
]


class KKProblemFormatter:
    def __init__(self, rand_seed, problem):
        self.rng = np.random.default_rng(rand_seed)
        self.rng_perturb = np.random.default_rng(rand_seed + 1)
        self.problem = problem

    def format_problem(self, random_names=True, random_saying_template=True,
                       random_knight_knave_pairs=False,
                       flip_knight_knave_pair=False, uncommon_name=False, reorder_statement=False):
        statements = copy.deepcopy(self.problem['statements'])

        n_people = len(statements)
        names = COMMON_NAMES[:n_people]
        if random_names:
            if uncommon_name is False:
                names = list(self.rng.choice(COMMON_NAMES, size=n_people, replace=False))
            else:
                names = list(self.rng.choice(UNCOMMON_NAMES, size=n_people, replace=False))
        names = [str(x) for x in names]

        knight_knave = ['a knight', 'a knave']
        if random_knight_knave_pairs:
            knight_knave = self.rng.choice(KNIGHT_KNAVE_PAIRS)
        knight_knave = [str(x) for x in knight_knave]

        if flip_knight_knave_pair:
            knight_knave = knight_knave[::-1]

        knight_knave_dict = {'knight': knight_knave[0].split()[1],
                             'knave': knight_knave[1].split()[1],
                             'a_knight': knight_knave[0], 'a_knave': knight_knave[1]}
        knight_knave_dict['Knight'] = knight_knave_dict['knight'].capitalize()
        knight_knave_dict['Knave'] = knight_knave_dict['knave'].capitalize()

        text = PREFIX.format(**knight_knave_dict)
        text += f'You meet {n_people} inhabitants: '
        text += ', '.join(names[:-1]) + ', and ' + names[-1] + '.'

        text_statements = []
        for i, stmt in enumerate(statements):
            tmpl = TEMPLATES[0]
            if random_saying_template:
                tmpl = self.rng.choice(TEMPLATES)

            content = format_statement(names, knight_knave_dict, stmt)
            text_statements.append(' ' + tmpl.format(name=names[i], content=content))

        if reorder_statement:
            original_order = list(range(n_people))
            shuffled_order = original_order.copy()
            while True:
                self.rng_perturb.shuffle(shuffled_order)
                if shuffled_order != original_order:
                    break
            for i in shuffled_order:
                text += text_statements[i]
        else:
            text += ''.join(text_statements)

        text += ' ' + POSTFIX.format(**knight_knave_dict)
        if self.problem['solution'] is None:
            solution_text = 'No valid solution exists.'
        else:
            solution_stmts = []
            for name, indicator in zip(names, self.problem['solution']):
                if indicator:
                    solution_stmts.append(name + ' is ' + knight_knave_dict['a_knight'])
                else:
                    solution_stmts.append(name + ' is ' + knight_knave_dict['a_knave'])
            solution_text = ', '.join(solution_stmts[:-1]) + ', and ' + solution_stmts[-1] + '.'
        return {'quiz': text, 'names': names, 'knight_knave': knight_knave_dict,
                'solution': self.problem['solution'],
                'solution_text': solution_text}


def format_knight_knave(names, knight_knave, statement, negation=False):
    """Format a leaf knight/knave statement."""
    assert statement[0] in ('telling-truth', 'lying')
    text = names[statement[1]] + ' is '
    if negation:
        text += 'not '
    text += {'telling-truth': knight_knave['a_knight'],
             'lying': knight_knave['a_knave']}[statement[0]]
    return text


def format_statement(names, knight_knave, statement):
    """Format a compound statement."""
    if statement[0] == 'not':
        return format_knight_knave(names, knight_knave, statement[1], negation=True)
    if statement[0] in ['and', 'or']:
        text = (' ' + statement[0] + ' ').join(
            format_knight_knave(names, knight_knave, sub_stmt) for sub_stmt in statement[1:])
        return text
    if statement[0] == '->':
        return ('If ' + format_knight_knave(names, knight_knave, statement[1]) + ' then ' +
                format_knight_knave(names, knight_knave, statement[2]))
    if statement[0] == '<=>':
        return (format_knight_knave(names, knight_knave, statement[1]) + ' if and only if ' +
                format_knight_knave(names, knight_knave, statement[2]))
    return format_knight_knave(names, knight_knave, statement)


####################################################################################
# GEM Environment
####################################################################################
class KnightsAndKnavesEnv(Env):
    """Knights and Knaves logic puzzle environment - single-turn Q&A."""

    def __init__(
        self,
        N: int = 3,
        depth_constraint: int = 2,
        width_constraint: int = 2,
        wrong_format_reward: float = -0.1,
        skip_no_solution: bool = True,
        skip_multiple_solutions: bool = True,
    ):
        super().__init__()
        # Parameter validation
        assert N >= 2, "N should be greater than or equal to 2"
        assert depth_constraint >= 1, "depth_constraint should be greater than or equal to 1"
        assert width_constraint >= 1, "width_constraint should be greater than or equal to 1"

        self.N = N
        self.depth_constraint = depth_constraint
        self.width_constraint = width_constraint
        self.wrong_format_reward = wrong_format_reward
        self.skip_no_solution = skip_no_solution
        self.skip_multiple_solutions = skip_multiple_solutions

        # State
        self.current_problem: Optional[str] = None
        self.names: List[str] = []
        self.reference_answer_text: Optional[str] = None
        self.gold_answer: Optional[Dict[str, str]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving Knights and Knaves logic puzzles.\n"
            "Output Format: Your final answer must list the identity of each character "
            "inside \\boxed{...} using the exact phrasing 'Name is a knight' or 'Name is a knave'.\n"
            "Example: \\boxed{Ella is a knight, Jacob is a knave, Benjamin is a knight, Lucas is a knave, and Samuel is a knight.}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate problem using the sampler
        sampler = KKProblemSamplerEnv(
            rand_seed=(seed if seed is not None else 0),
            n_people=self.N,
            depth_constraint=self.depth_constraint,
            width_constraint=self.width_constraint
        )
        problems = sampler.sample_valid_problems(
            n_problems=1,
            max_retry=1000,
            skip_no_solution=self.skip_no_solution,
            skip_multiple_solutions=self.skip_multiple_solutions
        )

        if not problems:
            raise RuntimeError("Failed to generate a valid problem")

        problem = problems[0]
        formatter = KKProblemFormatter(rand_seed=(seed if seed is not None else 0), problem=problem)
        formatted = formatter.format_problem()

        # Store problem data
        self.names = formatted["names"]
        self.reference_answer_text = formatted["solution_text"]
        self.gold_answer = self._process(self.reference_answer_text)

        # Build prompt
        statements_text = formatted["quiz"].split("So who is")[0].strip()
        self.current_problem = (
            f"{statements_text}\n\n"
            f"So who is a knight and who is a knave?\n\n"
            f"Output Format: Your final answer must list each character's identity inside \\boxed{{...}}.\n"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step to validate the answer."""
        # Parse the boxed content
        boxed_text = self._parse_answer(action)
        if boxed_text is None:
            return TERMINAL_STATE, self.wrong_format_reward, True, False, {"error": "format_error"}

        # Process to mapping
        parsed_mapping = self._process(boxed_text)
        if parsed_mapping is None:
            return TERMINAL_STATE, self.wrong_format_reward, True, False, {"error": "format_error"}

        # Compare with gold mapping
        assert self.gold_answer is not None, "Gold answer is not available."
        model_solution = [parsed_mapping[name] for name in self.names]
        true_solution = [self.gold_answer[name] for name in self.names]

        is_correct = all(a == b for a, b in zip(model_solution, true_solution))
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer_text": self.reference_answer_text,
            "gold_mapping": self.gold_answer,
            "user_mapping": parsed_mapping
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _process(self, answer: str) -> Optional[Dict[str, str]]:
        """Parse the answer text into a mapping from names to roles.

        The answer must include an identification for each name using 'Name is a knight' or 'Name is a knave'.
        """
        status_dict: Dict[str, str] = {}

        if not isinstance(answer, str):
            return None

        knight_count = answer.lower().count('knight')
        knave_count = answer.lower().count('knave')
        if knight_count + knave_count != self.N:
            return None

        for name in self.names:
            pattern = re.compile(rf'{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)', re.IGNORECASE)
            match = pattern.search(answer)
            if match:
                role = match.group(1).lower()
                status_dict[name] = role
            else:
                return None

        return status_dict

    def sample_random_action(self) -> str:
        """Sample a random valid format action (random assignment)."""
        roles = []
        for name in self.names:
            role = np.random.choice(['knight', 'knave'])
            roles.append(f"{name} is a {role}")
        # Join into the expected sentence format
        if len(roles) > 1:
            content = ', '.join(roles[:-1]) + ', and ' + roles[-1] + '.'
        else:
            content = roles[0] + '.'
        return f"\\boxed{{{content}}}"