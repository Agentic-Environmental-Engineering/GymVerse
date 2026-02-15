from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List

class FestivalCensusMarshalEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 12,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 12

        self.complexity_params = {
            # Number of NPCs in the festival roster: more entities increases search space and counting difficulty
            'num_entities': (8, 50),
            # Number of attribute predicates in the query: more conjuncts require more careful filtering
            'num_predicates': (1, 4),
            # Additional irrelevant attributes included to distract the agent: more distractors increase cognitive load
            'distraction_attrs': (0, 3),
            # Breadth of categorical values for core attributes: more unique values increases variety and complexity
            'value_breadth': (3, 8),
            # Frequency driver for sum-aggregation queries (0-7 mapped to probability ~0.0-0.7): higher means sum queries occur more often and are harder than pure counts
            'sum_agg_ratio': (0, 7),
            # Frequency driver for including one negated predicate (0-5 mapped to probability ~0.0-0.5): higher means more negations, harder to reason correctly
            'negation_ratio': (0, 5),
            # Maximum holdings count per NPC per item: larger caps produce larger sums and more variation
            'numeric_holdings_max': (2, 6),
        }

        self.param_variance = {
            'num_entities': 4,
            'num_predicates': 1,
            'distraction_attrs': 1,
            'value_breadth': 1,
            'sum_agg_ratio': 1,
            'negation_ratio': 1,
            'numeric_holdings_max': 1,
        }

        self.num_entities: int = 0
        self.num_predicates: int = 0
        self.distraction_attrs: int = 0
        self.value_breadth: int = 0
        self.sum_agg_ratio: int = 0
        self.negation_ratio: int = 0
        self.numeric_holdings_max: int = 0

        self.turn_count: int = 0
        self.roster: List[Dict[str, Any]] = []
        self.query: Dict[str, Any] = {}
        self.truth: int = 0
        self.marks: List[int] = []
        self.last_answer: Optional[int] = None

        self._core_attr_keys = ['role', 'faction', 'badge', 'game', 'shift']
        self._holdings_keys = ['tokens', 'tickets', 'dice_sets', 'flyers', 'ribbons']
        self._distraction_attr_space = {
            'mood': ['cheerful', 'tired', 'focused', 'nervous', 'calm', 'excited', 'stern', 'curious'],
            'mask': ['none', 'fox', 'dragon', 'wolf', 'owl', 'cat', 'stag', 'lion'],
            'region': ['North', 'South', 'East', 'West', 'Harbor', 'Highlands', 'Lowlands', 'Midway'],
            'band': ['solo', 'duo', 'trio', 'quartet', 'ensemble', 'orchestra', 'chorus', 'street'],
            'stance': ['idle', 'walking', 'patrolling', 'performing', 'trading', 'overseeing', 'resting', 'hosting'],
        }

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value
            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
            # Clamp to bounds irrespective of direction
            low = min(min_val, max_val)
            high = max(min_val, max_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "You are the Festival Census Marshal. Compute the requested aggregate from the roster and query.\n"
            "Goal: Either count NPCs matching the predicate, or sum a specified holding across those NPCs.\n"
            "The roster lists each NPC with attributes: role, faction, badge, game, shift, holdings, and extra distractors.\n"
            "Only the listed predicate attributes matter for the answer; distractors never affect eligibility.\n"
            "Available functions:\n"
            "- show_roster(): Reprint the entire roster and the query.\n"
            "- show_query(): Show the query alone.\n"
            "- mark(ids=...): Mark NPC IDs you want to track. Accepts a list or a string '1,3,5'.\n"
            "- clear_marks(): Clear all marked NPCs.\n"
            "- show_marks(): Show current marked IDs.\n"
            "- inspect_marked(): Show full lines for currently marked NPCs.\n"
            "- note(text='...'): Add a note to your scratchpad (no effect on answer).\n"
            "- propose(answer=INT): Submit your final answer and end the episode.\n"
            "Rules:\n"
            "- The episode ends when you call propose(). Correct answers yield reward 1.0; wrong answers yield 0.0.\n"
            "- Invalid action format terminates with a format error.\n"
            "- Unknown functions do not terminate but waste a turn.\n"
            "Format your action as <action>[function_name(param1=value1, ...)]</action>\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        lines = []
        lines.append("Festival Roster:")
        for e in self.roster:
            holdings_str = ", ".join([f"{k}:{e['holdings'][k]}" for k in self._holdings_keys])
            distractor_parts = []
            for dkey in e.get('distractors', {}):
                distractor_parts.append(f"{dkey}={e['distractors'][dkey]}")
            dstr = "; " + ", ".join(distractor_parts) if distractor_parts else ""
            lines.append(
                f"{e['id']}: role={e['role']}, faction={e['faction']}, badge={e['badge']}, "
                f"game={e['game']}, shift={e['shift']}, holdings: {{{holdings_str}}}{dstr}"
            )
        q = self._query_to_text(self.query)
        lines.append("")
        lines.append(f"Query: {q}")
        lines.append("Enter your action: <action>[function_name(param1=value1, ...)]</action>")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.marks = []
        self.last_answer = None

        roles_all = ['Vendor', 'Performer', 'Referee', 'Guard', 'Visitor', 'Host', 'Healer', 'Bard']
        factions_all = ['Dawn', 'Dusk', 'River', 'Ember', 'Gale', 'Stone', 'Bloom', 'Echo']
        badges_all = ['Red', 'Blue', 'Green', 'Yellow', 'Purple', 'White', 'Black', 'Gold']
        games_all = ['Archery', 'Dice', 'Cards', 'Darts', 'Joust', 'Chess', 'Riddles', 'Maze']
        shifts_all = ['Dawn', 'Morning', 'Noon', 'Afternoon', 'Dusk', 'Evening', 'Night', 'Midnight']

        b = self.value_breadth
        roles = random.sample(roles_all, min(b, len(roles_all)))
        factions = random.sample(factions_all, min(b, len(factions_all)))
        badges = random.sample(badges_all, min(b, len(badges_all)))
        games = random.sample(games_all, min(b, len(games_all)))
        shifts = random.sample(shifts_all, min(b, len(shifts_all)))

        distractor_keys = random.sample(list(self._distraction_attr_space.keys()), k=min(self.distraction_attrs, len(self._distraction_attr_space)))

        self.roster = []
        for i in range(1, self.num_entities + 1):
            e = {
                'id': i,
                'role': random.choice(roles),
                'faction': random.choice(factions),
                'badge': random.choice(badges),
                'game': random.choice(games),
                'shift': random.choice(shifts),
                'holdings': {hk: random.randint(0, self.numeric_holdings_max) for hk in self._holdings_keys},
                'distractors': {}
            }
            for dk in distractor_keys:
                vals = self._distraction_attr_space[dk]
                e['distractors'][dk] = random.choice(vals[:max(2, min(len(vals), b))])
            self.roster.append(e)

        # Build query with solvability check
        for _attempt in range(12):
            chosen_attrs = random.sample(self._core_attr_keys, k=min(self.num_predicates, len(self._core_attr_keys)))
            predicate = {}
            negated = None
            # Attribute value pools from actual used values to ensure realism
            pools = {
                'role': roles,
                'faction': factions,
                'badge': badges,
                'game': games,
                'shift': shifts,
            }
            for ak in chosen_attrs:
                predicate[ak] = random.choice(pools[ak])

            include_neg = random.random() < (self.negation_ratio / 10.0)
            if include_neg and len(chosen_attrs) >= 1:
                negated = random.choice(chosen_attrs)
                # For negation, ensure chosen value is from pool and does not make impossible contradictions
                predicate_neg = random.choice(pools[negated])

            # Aggregation mode
            use_sum = random.random() < (self.sum_agg_ratio / 10.0)
            sum_item = None
            if use_sum:
                sum_item = random.choice(self._holdings_keys)

            # Compute ground truth
            def matches(e):
                for ak, av in predicate.items():
                    if ak == negated:
                        if e[ak] == predicate_neg:
                            return False
                    else:
                        if e[ak] != av:
                            return False
                return True

            if use_sum:
                total = 0
                for e in self.roster:
                    if matches(e):
                        total += e['holdings'][sum_item]
                result = total
            else:
                result = sum(1 for e in self.roster if matches(e))

            # Keep instances solvable and non-trivial while allowing zeros sometimes
            if result == 0 and random.random() < 0.65 and self.num_entities >= 6:
                continue

            self.query = {
                'predicate': predicate,
                'negated_attr': negated,
                'negated_value': predicate_neg if include_neg else None,
                'mode': 'sum' if use_sum else 'count',
                'sum_item': sum_item,
            }
            self.truth = int(result)
            break
        else:
            # Fallback trivial but consistent
            self.query = {
                'predicate': {},
                'negated_attr': None,
                'negated_value': None,
                'mode': 'count',
                'sum_item': None,
            }
            self.truth = self.num_entities

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use <action>[function_name(param1=value1, ...)]</action>."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed["name"]
        params = parsed["parameters"]
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        if name == "show_roster":
            obs = self.get_task_suffix()

        elif name == "show_query":
            obs = "Query: " + self._query_to_text(self.query)

        elif name == "mark":
            ids_param = params.get("ids", params.get("id", None))
            if ids_param is None:
                obs = "Protocol violation: mark(ids=...) requires a list of IDs or a string of IDs."
            else:
                ids = self._parse_ids(ids_param)
                valid_ids = [i for i in ids if 1 <= i <= self.num_entities]
                self.marks = list(sorted(set(self.marks + valid_ids)))
                obs = f"Marked IDs: {self.marks}"

        elif name == "clear_marks":
            self.marks = []
            obs = "Marks cleared."

        elif name == "show_marks":
            obs = f"Current marked IDs: {self.marks}"

        elif name == "inspect_marked":
            if not self.marks:
                obs = "No IDs are marked."
            else:
                lines = []
                for mid in self.marks:
                    e = self.roster[mid - 1]
                    holdings_str = ", ".join([f"{k}:{e['holdings'][k]}" for k in self._holdings_keys])
                    distractor_parts = []
                    for dkey in e.get('distractors', {}):
                        distractor_parts.append(f"{dkey}={e['distractors'][dkey]}")
                    dstr = "; " + ", ".join(distractor_parts) if distractor_parts else ""
                    lines.append(
                        f"{e['id']}: role={e['role']}, faction={e['faction']}, badge={e['badge']}, "
                        f"game={e['game']}, shift={e['shift']}, holdings: {{{holdings_str}}}{dstr}"
                    )
                obs = "Marked NPC details:\n" + "\n".join(lines)

        elif name == "note":
            text = params.get("text", "")
            obs = f"Note recorded: {text}"

        elif name == "propose":
            ans = params.get("answer", params.get("value", None))
            if ans is None:
                obs = "Protocol violation: propose(answer=INT) requires an integer 'answer'."
            else:
                try:
                    if isinstance(ans, bool):
                        raise ValueError
                    ans_int = int(ans)
                    self.last_answer = ans_int
                    if ans_int == self.truth:
                        obs = f"Success! Correct answer {ans_int}."
                        reward = 1.0
                    else:
                        obs = f"Failed. Your answer was {ans_int}. The correct answer was {self.truth}."
                        reward = 0.0
                    terminated = True
                except Exception:
                    obs = "Protocol violation: propose(answer=INT) requires an integer 'answer'."

        else:
            obs = f"Unknown function '{name}'. Use show_roster, show_query, mark, clear_marks, show_marks, inspect_marked, note, propose."

        if not terminated and self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns}). Episode timed out."
            terminated = True
            truncated = True
            reward = 0.0

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        try:
            from gem.utils.parsing import extract_action_parameters
        except Exception:
            return None

        content = extract_action_parameters(action)
        if not content:
            return None

        content = content.strip()
        if not (content.startswith('[') and content.endswith(']')):
            return None
        func_call_str = content[1:-1].strip()

        func_pattern = re.compile(r'^(\w+)\((.*)\)$', re.DOTALL)
        func_match = func_pattern.match(func_call_str)
        if not func_match:
            return None

        func_name = func_match.group(1)
        params_str = func_match.group(2).strip()

        parameters: Dict[str, Any] = {}
        if params_str:
            param_pairs = re.findall(r'(\w+)\s*=\s*([^,]+?)(?:,|$)', params_str)
            for key, value in param_pairs:
                value = value.strip()
                try:
                    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                        parameters[key] = value[1:-1]
                    elif value.startswith('[') and value.endswith(']'):
                        inner = value[1:-1].strip()
                        if not inner:
                            parameters[key] = []
                        else:
                            items = [v.strip() for v in inner.split(',')]
                            parsed_items = []
                            for it in items:
                                if (it.startswith('"') and it.endswith('"')) or (it.startswith("'") and it.endswith("'")):
                                    parsed_items.append(it[1:-1])
                                elif re.match(r'^-?\d+$', it):
                                    parsed_items.append(int(it))
                                else:
                                    parsed_items.append(it)
                            parameters[key] = parsed_items
                    elif re.match(r'^-?\d+\.\d+$', value):
                        parameters[key] = float(value)
                    elif re.match(r'^-?\d+$', value):
                        parameters[key] = int(value)
                    elif value.lower() == 'true':
                        parameters[key] = True
                    elif value.lower() == 'false':
                        parameters[key] = False
                    else:
                        parameters[key] = value
                except Exception:
                    parameters[key] = value
        return {"name": func_name, "parameters": parameters}

    def sample_random_action(self) -> str:
        options = ["show_roster", "show_query", "show_marks", "inspect_marked", "clear_marks", "note", "mark", "propose"]
        fn = random.choice(options)
        if fn == "note":
            return "<action>[note(text='working on it')]</action>"
        if fn == "mark":
            a = random.sample(range(1, max(2, self.num_entities + 1)), k=min(2, max(1, self.num_entities // 5)))
            return f"<action>[mark(ids=[{', '.join(str(x) for x in a)}])]</action>"
        if fn == "propose":
            guess = random.randint(0, max(3, self.num_entities // 2))
            return f"<action>[propose(answer={guess})]</action>"
        return f"<action>[{fn}()]</action>"

    def _query_to_text(self, query: Dict[str, Any]) -> str:
        predicate = query['predicate']
        neg_attr = query['negated_attr']
        neg_val = query['negated_value']
        mode = query['mode']
        sum_item = query['sum_item']

        parts = []
        for ak in predicate:
            if ak == neg_attr:
                parts.append(f"{ak} != {neg_val}")
            else:
                parts.append(f"{ak} = {predicate[ak]}")
        cond = " AND ".join(parts) if parts else "no conditions (everyone)"
        if mode == 'count':
            return f"Count NPCs where {cond}."
        else:
            return f"Sum '{sum_item}' across NPCs where {cond}."

    def _parse_ids(self, ids_val: Any) -> List[int]:
        result: List[int] = []
        if isinstance(ids_val, list):
            for v in ids_val:
                try:
                    iv = int(v)
                    result.append(iv)
                except Exception:
                    continue
        elif isinstance(ids_val, str):
            s = ids_val.replace(',', ' ').replace(';', ' ')
            for token in s.split():
                if token.strip().lstrip('-').isdigit():
                    result.append(int(token.strip()))
        elif isinstance(ids_val, int):
            result.append(ids_val)
        return result


class FestivalCensusMarshalEnvWithFeedback(FestivalCensusMarshalEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "invalid action format" in text:
            error_type = "FormatError"
            error_detail["issue"] = "invalid_action_tags_or_syntax"
            hint = "Wrap a single function call inside <action>[...]</action> like <action>[show_roster()]</action>."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "propose" in text:
                error_detail["violation"] = "propose_requires_integer_answer"
                hint = "Call propose(answer=INT) with an integer, e.g., <action>[propose(answer=5)]</action>."
            elif "mark" in text:
                error_detail["violation"] = "mark_requires_ids"
                hint = "Call mark(ids=[1,2,3]) or mark(ids='1,2,3') to mark NPC IDs."
            else:
                error_detail["violation"] = "general_protocol"
                hint = "Check function signature in instructions and provide required parameters."
        elif "unknown function" in text:
            error_type = "UnsupportedAction"
            error_detail["function"] = "unknown"
            hint = "Use one of: show_roster, show_query, mark, clear_marks, show_marks, inspect_marked, note, propose."
        elif "reached max turns" in text or "timed out" in text:
            error_type = "Timeout"
            error_detail["max_turns"] = self.max_turns
            hint = "Use show_query() to restate the task, then compute and call propose(answer=INT) before the limit."
        elif "failed." in text and "correct answer was" in text:
            error_type = "WrongDecision"
            user_ans = None
            correct = None
            m1 = re.search(r"your answer was (\-?\d+)", text)
            m2 = re.search(r"correct answer was (\-?\d+)", text)
            if m1:
                user_ans = int(m1.group(1))
            if m2:
                correct = int(m2.group(1))
            error_detail["got"] = user_ans
            error_detail["expected"] = correct
            hint = "Re-read the predicate conjunction; ensure the negated condition is applied and use the correct aggregation."
        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Well done!"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            q = getattr(self, "query", {})
            diagnostic["query"] = {
                "mode": q.get("mode"),
                "predicate": q.get("predicate"),
                "negated_attr": q.get("negated_attr"),
                "negated_value": q.get("negated_value"),
                "sum_item": q.get("sum_item"),
            }
            diagnostic["truth"] = getattr(self, "truth", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by calling show_roster() or show_query() to review the task, then compute and propose(answer=INT).",
            "turn": 0,
            "query": {
                "mode": self.query.get("mode"),
                "predicate": self.query.get("predicate"),
                "negated_attr": self.query.get("negated_attr"),
                "negated_value": self.query.get("negated_value"),
                "sum_item": self.query.get("sum_item"),
            },
            "truth": self.truth,
        }
        return obs, info