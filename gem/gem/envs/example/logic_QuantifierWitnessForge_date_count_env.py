from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class QuantifierWitnessForgeEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 20,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, complexity))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 20

        # Evolvable parameters (logic-native difficulty)
        self.complexity_params = {
            # Domain size for finite Herbrand-style evaluation over constants 0..n-1; larger = harder search
            "domain_size": (2, 6),
            # Number of distinct predicate symbols (arity 1 only for clarity); more predicates = richer structure
            "num_predicates": (2, 5),
            # Number of clauses (CNF/DNF-like atoms under quantifiers) composing the matrix; more = harder
            "num_clauses": (2, 8),
            # Quantifier alternations depth (e.g., ∀x ∃y ∀z ...); deeper alternation = harder game
            "quantifier_depth": (1, 4),
            # Noise literals per clause (extra disjuncts/conjuncts that increase ambiguity); more = harder
            "lits_per_clause": (2, 5),
        }

        # Variance: modest randomness to prevent overfitting
        self.param_variance = {
            "domain_size": 1,
            "num_predicates": 1,
            "num_clauses": 1,
            "quantifier_depth": 1,
            "lits_per_clause": 1,
        }

        # Placeholder attributes set in _apply_complexity_params
        self.domain_size: int = 0
        self.num_predicates: int = 0
        self.num_clauses: int = 0
        self.quantifier_depth: int = 0
        self.lits_per_clause: int = 0

        # State
        self.turn_count: int = 0
        self.active: bool = True
        self.quantifiers: List[Tuple[str, str]] = []  # list of (Q, var), Q in {"forall","exists"}
        self.predicates: List[str] = []
        self.assignment_table: Dict[str, List[bool]] = {}  # predicate name -> truth table over domain
        self.form_structure: Dict[str, Any] = {}  # matrix structure: list of clauses with literals
        self.semantics_mode: str = ""  # "CNF" or "DNF" matrix mode
        self.hidden_truth: Optional[bool] = None  # truth value of the closed sentence
        self.requires_witness: bool = False  # whether success requires a witness
        self.witness_shape: Dict[str, Any] = {}  # variables that require witness mapping (for exists)
        self.spec_text: str = ""

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for p, (lo, hi) in self.complexity_params.items():
            center = lo + (hi - lo) * normalized
            val = center
            if self.enable_param_randomization:
                var = self.param_variance.get(p, 0)
                if var > 0:
                    val = center + random.uniform(-var, var)
                # clamp (supports normal ranges)
                val = max(min(lo, hi), min(max(lo, hi), val))
            setattr(self, p, int(round(val)))

        # Ensure feasibility safeguards
        self.domain_size = max(2, self.domain_size)
        self.num_predicates = max(1, self.num_predicates)
        self.num_clauses = max(1, self.num_clauses)
        self.quantifier_depth = max(1, self.quantifier_depth)
        self.lits_per_clause = max(1, self.lits_per_clause)

    def _sample_quantifiers(self) -> List[Tuple[str, str]]:
        # Alternate starting with forall at lower complexity; introduce starts with exists at higher levels
        start_forall = True if self.complexity <= 5 else random.choice([True, False])
        seq = []
        for i in range(self.quantifier_depth):
            Q = "forall" if ((i % 2 == 0 and start_forall) or (i % 2 == 1 and not start_forall)) else "exists"
            var = chr(ord('x') + i)
            seq.append((Q, var))
        return seq

    def _sample_predicates(self) -> List[str]:
        # Unary predicate symbols P1..Pk
        return [f"P{i+1}" for i in range(self.num_predicates)]

    def _sample_interpretation(self) -> Dict[str, List[bool]]:
        # For each predicate, truth table over domain elements [0..n-1]
        table = {}
        for pname in self.predicates:
            # bias a bit to avoid trivial always true/false at low complexity
            truth_row = []
            for d in range(self.domain_size):
                if self.complexity <= 3:
                    prob_true = 0.5
                elif self.complexity <= 7:
                    prob_true = random.uniform(0.3, 0.7)
                else:
                    prob_true = random.uniform(0.2, 0.8)
                truth_row.append(random.random() < prob_true)
            # avoid all True or all False tables at easy levels
            if self.complexity <= 4 and (all(truth_row) or not any(truth_row)):
                idx = random.randrange(self.domain_size)
                truth_row[idx] = not truth_row[idx]
            table[pname] = truth_row
        return table

    def _sample_matrix(self) -> Dict[str, Any]:
        # Build clauses with literals like Pk(v) or ¬Pk(v); choose var references among bound variables
        bound_vars = [v for _, v in self.quantifiers]
        # Choose matrix mode: CNF or DNF
        self.semantics_mode = random.choice(["CNF", "DNF"]) if self.complexity >= 4 else "CNF"
        clauses = []
        for _ in range(self.num_clauses):
            lits = []
            for _j in range(self.lits_per_clause):
                p = random.choice(self.predicates)
                v = random.choice(bound_vars)
                neg = random.choice([True, False])
                lits.append({"pred": p, "var": v, "neg": neg})
            # De-duplicate same literal within clause
            norm = {(lit["pred"], lit["var"], lit["neg"]) for lit in lits}
            lits = [{"pred": a, "var": b, "neg": c} for (a, b, c) in norm]
            clauses.append({"lits": lits})
        return {"mode": self.semantics_mode, "clauses": clauses}

    def _eval_literal(self, lit: Dict[str, Any], env_assign: Dict[str, int]) -> bool:
        pred = lit["pred"]
        var = lit["var"]
        neg = lit["neg"]
        dval = env_assign[var]
        base = self.assignment_table[pred][dval]
        return (not base) if neg else base

    def _eval_matrix(self, env_assign: Dict[str, int]) -> bool:
        mode = self.form_structure["mode"]
        clauses = self.form_structure["clauses"]
        if mode == "CNF":
            # AND over clauses; each clause is OR over its literals
            for clause in clauses:
                satisfied = False
                for lit in clause["lits"]:
                    if self._eval_literal(lit, env_assign):
                        satisfied = True
                        break
                if not satisfied:
                    return False
            return True
        else:
            # DNF: OR over clauses; each clause is AND over its literals
            for clause in clauses:
                ok = True
                for lit in clause["lits"]:
                    if not self._eval_literal(lit, env_assign):
                        ok = False
                        break
                if ok:
                    return True
            return False

    def _eval_quantified(self) -> Tuple[bool, Dict[str, int]]:
        # Returns (truth, canonical_witness) where the witness maps each existential variable
        # to a Skolem-like function of preceding universals; we return a representative
        quant = self.quantifiers
        dom = list(range(self.domain_size))

        # For existential witness requirement: only require witnesses for the outermost block of existentials
        # if the sentence is true; if false, witness is irrelevant
        # Canonical strategy: produce a mapping for each exists var that depends on previously fixed universals
        # For simplicity, we compute truth by backtracking. To extract a witness, when choosing an existential value
        # we store the first successful choice conditional on the current universal prefix.
        witness_map = {}  # var -> dict of prefix tuple -> chosen value

        def eval_rec(i: int, env_assign: Dict[str, int], prefix_univ: List[int]) -> bool:
            if i == len(quant):
                return self._eval_matrix(env_assign)
            Q, v = quant[i]
            if Q == "forall":
                for a in dom:
                    env_assign[v] = a
                    if not eval_rec(i + 1, env_assign, prefix_univ + [a]):
                        return False
                return True
            else:  # exists
                # record witness per universal prefix key
                key = tuple(prefix_univ)
                for a in dom:
                    env_assign[v] = a
                    if eval_rec(i + 1, env_assign, prefix_univ):
                        # save first successful choice for this prefix
                        if v not in witness_map:
                            witness_map[v] = {}
                        if key not in witness_map[v]:
                            witness_map[v][key] = a
                        return True
                return False

        truth = eval_rec(0, {}, [])
        return truth, {v: mapping for v, mapping in witness_map.items()}

    def _render_formula_text(self) -> str:
        # Build human-readable text
        qtxt = " ".join([("∀" if Q == "forall" else "∃") + v for Q, v in self.quantifiers])
        mode = self.form_structure["mode"]
        clause_texts = []
        for cl in self.form_structure["clauses"]:
            lit_texts = []
            for lit in cl["lits"]:
                atom = f"{lit['pred']}({lit['var']})"
                lit_texts.append(("¬" if lit["neg"] else "") + atom)
            if mode == "CNF":
                clause_texts.append("(" + " ∨ ".join(lit_texts) + ")")
            else:
                clause_texts.append("(" + " ∧ ".join(lit_texts) + ")")
        if mode == "CNF":
            matrix = " ∧ ".join(clause_texts)
        else:
            matrix = " ∨ ".join(clause_texts)
        return f"{qtxt}. {matrix}"

    def _decide_requires_witness(self, truth: bool) -> bool:
        # Require a witness if there exists at the outermost position and the statement is true
        if not truth:
            return False
        if len(self.quantifiers) == 0:
            return False
        firstQ, _ = self.quantifiers[0]
        return firstQ == "exists"

    def _get_instructions(self) -> str:
        return (
            "You are evaluating a finite-domain first-order logic sentence with unary predicates.\n"
            "Goal: Decide if the sentence is TRUE over the given finite interpretation. If the outermost quantifier\n"
            "is existential and the sentence is TRUE, you must also provide a witness mapping for each existential variable.\n"
            "Domain: elements are integers 0..(n-1). Predicates Pi(d) are evaluated in the hidden interpretation.\n"
            "Format your action inside \\boxed{...} using one of the following commands:\n"
            "- verdict value=TRUE\n"
            "- verdict value=FALSE\n"
            "- verdict value=TRUE witness var=x mapping=(prefix->value; ...)\n"
            "  Witness format: For each existential variable 'v', list mappings from the tuple of preceding universal\n"
            "  assignments to a chosen domain value. Use comma-separated entries; prefix is comma-separated integers.\n"
            "Examples:\n"
            f"{r'\\boxed{verdict value=TRUE}'}\n"
            f"{r'\\boxed{verdict value=FALSE}'}\n"
            f"{r'\\boxed{verdict value=TRUE witness x=( ->1 ); y=(0->2,1->0) }'}\n"
        )

    def get_task_suffix(self) -> str:
        lines = []
        lines.append("Current problem:")
        lines.append(f"- Domain size: {self.domain_size} (elements 0..{self.domain_size-1})")
        lines.append(f"- Predicates: {', '.join(self.predicates)} (unary)")
        lines.append(f"- Quantified sentence: {self.spec_text}")
        lines.append(f"- Matrix mode: {self.semantics_mode}")
        lines.append("Submit exactly one action in \\boxed{...} with 'verdict' command as per instructions.")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        self.turn_count = 0
        self.active = True

        self.quantifiers = self._sample_quantifiers()
        self.predicates = self._sample_predicates()
        self.assignment_table = self._sample_interpretation()
        self.form_structure = self._sample_matrix()
        self.spec_text = self._render_formula_text()
        truth, witness_map = self._eval_quantified()
        self.hidden_truth = truth
        self.requires_witness = self._decide_requires_witness(truth)
        # Record witness shape (which exist vars and expected prefixes)
        self.witness_shape = {}
        univ_vars = [v for Q, v in self.quantifiers if Q == "forall"]
        exist_vars = [v for Q, v in self.quantifiers if Q == "exists"]
        # Outermost exists requirement: only variables up to the first universal break?
        # Simpler: require all exists variables to provide mapping keyed by all preceding universals.
        prefix_lists = {}
        prefix_accum = []
        for Q, v in self.quantifiers:
            if Q == "forall":
                prefix_accum.append(v)
            if Q == "exists":
                prefix_lists[v] = list(prefix_accum)
        self.witness_shape["exist_vars"] = exist_vars
        self.witness_shape["prefixes"] = prefix_lists
        # Store a hidden canonical witness only if true; used to verify submissions
        self._hidden_witness = witness_map if truth else {}

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        if parsed is None:
            obs = "INVALID ACTION FORMAT: expected \\boxed{...} enclosing a 'verdict' command."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if parsed.get("action") != "verdict":
            obs = "UNSUPPORTED ACTION: Only 'verdict' is allowed."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        val = parsed.get("value", "").upper()
        if val not in ("TRUE", "FALSE"):
            obs = "PROTOCOL VIOLATION: verdict must specify value=TRUE or value=FALSE."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        declared_truth = (val == "TRUE")

        # Witness parsing if provided
        provided_witness_raw = parsed.get("witness", "").strip()
        provided_maps: Dict[str, Dict[Tuple[int, ...], int]] = {}
        witness_parse_error = None

        if provided_witness_raw:
            # Expect form like: x=( ->1 , 0->2 ); y=(0->1,1->1)
            try:
                provided_maps = self._parse_witness_block(provided_witness_raw)
            except Exception as e:
                witness_parse_error = f"FORMAT ERROR in witness: {e}"

        # Evaluate correctness
        correct = (declared_truth == self.hidden_truth)

        if not correct:
            obs = f"Result: WRONG DECISION. The sentence is actually {'TRUE' if self.hidden_truth else 'FALSE'}."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Correct verdict; check witness requirement
        if self.requires_witness:
            # Must provide witness for all existential vars
            if not provided_maps:
                obs = "PROTOCOL VIOLATION: A witness is required for a TRUE sentence with outermost ∃. None provided."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            if witness_parse_error:
                obs = f"INVALID WITNESS: {witness_parse_error}"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            # Validate witness coverage and values
            ok, reason = self._validate_witness(provided_maps)
            if not ok:
                obs = f"INVALID WITNESS: {reason}"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            obs = "Success! Correct verdict and valid witness."
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            # No witness required; but if provided, optionally verify consistency and ignore
            if provided_maps and not witness_parse_error:
                # Optional: check consistency; if inconsistent, mark as protocol violation
                ok, reason = self._validate_witness(provided_maps)
                if not ok:
                    obs = f"PROTOCOL VIOLATION: Witness not required for this case, and provided witness is invalid: {reason}"
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            obs = "Success! Correct verdict."
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        m = re.search(r"\\boxed\{(.+?)\}\s*$", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()
        if not inner:
            return None

        # Tokenize: we allow a 'witness' block that may include parentheses and punctuation.
        # Pattern: 'verdict' then key=value pairs where value may be a simple token or a parenthesized block
        # For simplicity, extract 'verdict' presence, 'value=...' and 'witness ...' remainder.
        tokens = {}
        # Ensure starts with 'verdict'
        if not inner.lower().startswith("verdict"):
            return None
        tokens["action"] = "verdict"

        # Extract value=TRUE/FALSE
        mval = re.search(r"value\s*=\s*(TRUE|FALSE)", inner, flags=re.IGNORECASE)
        if mval:
            tokens["value"] = mval.group(1)

        # Extract witness block after keyword 'witness'
        mwit = re.search(r"witness\s+(.+)$", inner, flags=re.IGNORECASE | re.DOTALL)
        if mwit:
            tokens["witness"] = mwit.group(1).strip()

        return tokens

    def sample_random_action(self) -> str:
        # Random valid-looking action; may or may not include witness
        if random.random() < 0.5:
            return r"\boxed{verdict value=TRUE}"
        else:
            if random.random() < 0.5:
                return r"\boxed{verdict value=FALSE}"
            # fabricate a simplistic witness stub
            ex_vars = [v for Q, v in self.quantifiers if Q == "exists"]
            parts = []
            for v in ex_vars[:2]:
                parts.append(f"{v}=( ->0 )")
            witness_str = "; ".join(parts)
            return rf"\boxed{{verdict value=TRUE witness {witness_str}}}"

    def _parse_witness_block(self, block: str) -> Dict[str, Dict[Tuple[int, ...], int]]:
        # Expect sequences like: x=( ->1 , 0->2 ); y=(0->2,1->0)
        result: Dict[str, Dict[Tuple[int, ...], int]] = {}
        # Split on ';' between variables
        var_specs = [s.strip() for s in re.split(r";", block) if s.strip()]
        for vs in var_specs:
            m = re.match(r"([a-zA-Z])\s*=\s*\((.*?)\)\s*$", vs)
            if not m:
                raise ValueError(f"Bad variable mapping segment: '{vs}'")
            var = m.group(1)
            body = m.group(2).strip()
            mapping: Dict[Tuple[int, ...], int] = {}
            if body:
                entries = [e.strip() for e in re.split(r",", body) if e.strip()]
                for e in entries:
                    # entry like "0,1->2" or "->3"
                    if "->" not in e:
                        raise ValueError(f"Bad mapping entry '{e}'")
                    left, right = e.split("->", 1)
                    left = left.strip()
                    right = right.strip()
                    if left == "" or left == "()":
                        prefix = ()
                    elif left == " " or left == "->":
                        prefix = ()
                    elif left == "":
                        prefix = ()
                    elif left == "":
                        prefix = ()
                    else:
                        # parse comma-separated ints
                        parts = [p for p in re.split(r"\s*,\s*", left) if p != ""]
                        try:
                            prefix = tuple(int(x) for x in parts)
                        except:
                            raise ValueError(f"Non-integer in prefix '{left}'")
                    try:
                        val = int(right)
                    except:
                        raise ValueError(f"Non-integer value '{right}'")
                    if val < 0 or val >= self.domain_size:
                        raise ValueError(f"value {val} out of domain 0..{self.domain_size-1}")
                    mapping[prefix] = val
            result[var] = mapping
        return result

    def _validate_witness(self, provided_maps: Dict[str, Dict[Tuple[int, ...], int]]) -> Tuple[bool, str]:
        # Ensure all existential vars present
        exist_vars = self.witness_shape.get("exist_vars", [])
        prefixes = self.witness_shape.get("prefixes", {})
        for v in exist_vars:
            if v not in provided_maps:
                return False, f"missing mapping for existential variable '{v}'"
        # Check coverage for each existential variable over all universal prefixes that precede it.
        # Build all prefix tuples according to the specified preceding universal vars.
        # Enumerate domain tuples
        univ_order = [var for (Q, var) in self.quantifiers if Q == "forall"]
        # For each exists var, determine which universals precede it
        for v in exist_vars:
            preceding = prefixes.get(v, [])
            # enumerate all assignments for preceding universals
            if preceding:
                idxs = [univ_order.index(u) for u in preceding]
                # generate all combinations over domain for the preceding vars in their order
                def gen(prefix_vars):
                    if not prefix_vars:
                        yield ()
                        return
                    head, tail = prefix_vars[0], prefix_vars[1:]
                    for val in range(self.domain_size):
                        for rest in gen(tail):
                            yield (val,) + rest
                # Build mapping from preceding order to stored tuple order
                # We store prefixes as raw tuples in the given order; expect the same order
                required_prefixes = list(gen(preceding))
            else:
                required_prefixes = [()]

            pmap = provided_maps.get(v, {})
            for pf in required_prefixes:
                if pf not in pmap:
                    return False, f"missing mapping for '{v}' at prefix {pf}"

        # Finally, verify that provided witness indeed makes the sentence true via Skolem-like strategy:
        # For all universals, the existential choices given by mapping should satisfy the matrix.
        quant = self.quantifiers
        dom = list(range(self.domain_size))

        # Build ordering arrays
        univ_vars = [v for (Q, v) in quant if Q == "forall"]

        def eval_with_witness(i: int, env_assign: Dict[str, int], prefix_univ_vals: List[int]) -> bool:
            if i == len(quant):
                return self._eval_matrix(env_assign)
            Q, var = quant[i]
            if Q == "forall":
                for a in dom:
                    env_assign[var] = a
                    if not eval_with_witness(i + 1, env_assign, prefix_univ_vals + [a]):
                        return False
                return True
            else:
                # get preceding universals for this var
                preceding = self.witness_shape["prefixes"].get(var, [])
                # extract current prefix tuple in that order
                cur_prefix = []
                for u in preceding:
                    # find index of u among univ_vars to pull from prefix_univ_vals
                    idx = univ_vars.index(u)
                    if idx >= len(prefix_univ_vals):
                        # Should not happen if recursion is right
                        return False
                    cur_prefix.append(prefix_univ_vals[idx])
                cur_prefix_t = tuple(cur_prefix)
                valmap = provided_maps.get(var, {})
                if cur_prefix_t not in valmap:
                    return False
                chosen = valmap[cur_prefix_t]
                env_assign[var] = chosen
                return eval_with_witness(i + 1, env_assign, prefix_univ_vals)

        ok = eval_with_witness(0, {}, [])
        if not ok:
            return False, "witness does not satisfy the formula for some universal choices"
        return True, "ok"


class QuantifierWitnessForgeEnvWithFeedback(QuantifierWitnessForgeEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()

        error_type = "OK"
        error_detail = {}
        hint = None

        if "invalid action format" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_or_malformed"
            hint = "Wrap your command like \\boxed{verdict value=TRUE}."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed"] = ["verdict"]
            hint = "Use 'verdict' as the action name."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "value=true or value=false" in text:
                error_detail["violation"] = "missing_value"
                hint = "Include value=TRUE or value=FALSE."
            elif "witness is required" in text:
                error_detail["violation"] = "missing_witness_on_true_outer_exists"
                hint = "Provide witness mappings for each existential variable."
            elif "witness not required" in text:
                error_detail["violation"] = "unnecessary_invalid_witness"
                hint = "If not required, omit witness or ensure it is consistent."
            else:
                error_detail["violation"] = "general_protocol"
                hint = "Follow the exact key=value format described in instructions."

        elif "invalid witness" in text:
            error_type = "ProtocolViolation"
            # Extract detail snippets
            if "format error in witness" in text:
                error_detail["violation"] = "witness_format"
                hint = "Use var=(prefix->value, ...). Example: x=( ->1 ); y=(0->2,1->0)"
            elif "missing mapping for" in text:
                m = re.search(r"missing mapping for '([a-zA-Z])' at prefix (\([0-9,\s]*\))", obs, flags=re.IGNORECASE)
                if m:
                    error_detail["violation"] = "witness_missing_prefix"
                    error_detail["var"] = m.group(1)
                    error_detail["prefix"] = m.group(2)
                hint = "List a mapping for every required universal prefix."
            elif "out of domain" in text:
                error_detail["violation"] = "value_out_of_domain"
                hint = "Map values must be integers within the domain range."
            elif "does not satisfy the formula" in text:
                error_detail["violation"] = "witness_incorrect"
                hint = "Ensure each existential choice makes the matrix true for all universal assignments."
            else:
                error_detail["violation"] = "witness_general"
                hint = "Recheck your witness formatting and coverage."

        elif "wrong decision" in text:
            error_type = "WrongDecision"
            truth = "true" if getattr(self, "hidden_truth", False) else "false"
            error_detail["expected_truth"] = truth
            hint = "Re-evaluate the quantifier structure: ∀ needs all cases, ∃ needs at least one."

        elif "reached max turns" in text or truncated:
            error_type = "Timeout"
            hint = "Decide and submit a single verdict within the limit."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            # add relevant state snapshot
            diagnostic["state"] = {
                "domain_size": getattr(self, "domain_size", None),
                "quantifier_depth": getattr(self, "quantifier_depth", None),
                "requires_witness": getattr(self, "requires_witness", None),
                "predicates": getattr(self, "predicates", []),
                "mode": getattr(self, "semantics_mode", ""),
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by deciding TRUE or FALSE. If the sentence starts with ∃ and is TRUE, include a witness.",
            "turn": 0,
            "state": {
                "domain_size": getattr(self, "domain_size", None),
                "quantifier_depth": getattr(self, "quantifier_depth", None),
                "requires_witness": getattr(self, "requires_witness", None),
                "predicates": getattr(self, "predicates", []),
                "mode": getattr(self, "semantics_mode", ""),
            },
        }
        return obs, info