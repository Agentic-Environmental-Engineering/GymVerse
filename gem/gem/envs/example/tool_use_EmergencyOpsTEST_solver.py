import sys
from pathlib import Path
import importlib.util
import argparse
import random
import time

# ============ GEM 依赖配置 ============
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent  # GymVerse 根目录
sys.path.insert(0, str(project_root))

gem_core_path = project_root / "gem" / "gem" / "core.py"
spec_core = importlib.util.spec_from_file_location("gem.core", gem_core_path)
gem_core_module = importlib.util.module_from_spec(spec_core)
sys.modules['gem.core'] = gem_core_module

gem_utils_seeding_path = project_root / "gem" / "gem" / "utils" / "seeding.py"
spec_seeding = importlib.util.spec_from_file_location("gem.utils.seeding", gem_utils_seeding_path)
gem_utils_seeding = importlib.util.module_from_spec(spec_seeding)
sys.modules['gem.utils'] = type(sys)('gem.utils')
sys.modules['gem.utils.seeding'] = gem_utils_seeding
spec_seeding.loader.exec_module(gem_utils_seeding)
spec_core.loader.exec_module(gem_core_module)

gem_utils_constants_path = project_root / "gem" / "gem" / "utils" / "constants.py"
spec_constants = importlib.util.spec_from_file_location("gem.utils.constants", gem_utils_constants_path)
gem_utils_constants = importlib.util.module_from_spec(spec_constants)
sys.modules['gem.utils.constants'] = gem_utils_constants
spec_constants.loader.exec_module(gem_utils_constants)

gem_utils_parsing_path = project_root / "gem" / "gem" / "utils" / "parsing.py"
spec_parsing = importlib.util.spec_from_file_location("gem.utils.parsing", gem_utils_parsing_path)
gem_utils_parsing = importlib.util.module_from_spec(spec_parsing)
sys.modules['gem.utils.parsing'] = gem_utils_parsing
spec_parsing.loader.exec_module(gem_utils_parsing)

# Import the environment module
env_module_name = "tool_use_EmergencyOpsTEST_env"
possible_path = Path(__file__).with_name(env_module_name + ".py")
if not possible_path.exists():
    raise ImportError(f"Cannot find environment module {env_module_name} at {possible_path}")

spec_env = importlib.util.spec_from_file_location(env_module_name, possible_path)
env_module = importlib.util.module_from_spec(spec_env)
sys.modules[env_module_name] = env_module
spec_env.loader.exec_module(env_module)

EmergencyOpsTESTEnv = getattr(env_module, "EmergencyOpsTESTEnv")


def boxed(cmd: str) -> str:
    return r"\boxed{" + cmd + "}"


def safe_step(env, cmd: str):
    obs, reward, terminated, truncated, info = env.step(cmd)
    return obs, reward, terminated, truncated, info


def solve_level(complexity: int, seed: int):
    """Solver for EmergencyOpsTEST environment."""
    env = EmergencyOpsTESTEnv(complexity=complexity, max_turns=220)
    obs, info = env.reset(seed=seed)

    details = {
        "complexity": complexity,
        "seed": seed,
        "actions": [],
        "required_steps": getattr(env, "required_steps", 0),
        "task": getattr(env, "task", {}),
        "solution": getattr(env, "task", {}).get("solution", None),
    }

    # Access the internal task structure
    task = env.task
    base_table = task.get("base_table")
    ops = task.get("ops", [])
    metric = task.get("metric")
    metric_column = task.get("metric_column")
    required_steps = env.required_steps

    print(f"  Task: base={base_table}, ops={len(ops)}, metric={metric}({metric_column}), required_steps={required_steps}")

    # Step 1: Open the base table
    cmd = boxed(f"tool:open_table(name='{base_table}')")
    obs, reward, terminated, truncated, info = safe_step(env, cmd)
    details["actions"].append(cmd)
    if terminated or truncated:
        return obs, reward, terminated, truncated, {"details": details, "info": info}

    # Step 2: Apply operations
    for op in ops:
        if op["op"] == "join":
            tname = op["table"]
            left_on = op["left_on"]
            right_on = op["right_on"]
            cmd = boxed(f"tool:join_table(name='{tname}', left_on='{left_on}', right_on='{right_on}', how='inner')")
            obs, reward, terminated, truncated, info = safe_step(env, cmd)
            details["actions"].append(cmd)
            if terminated or truncated:
                return obs, reward, terminated, truncated, {"details": details, "info": info}

        elif op["op"] == "filter":
            col = op["column"]
            operator = op["operator"]
            val = op["value"]
            op_map = {"eq": "eq", "gt": "gt", "lt": "lt"}
            op_str = op_map.get(operator, "eq")
            cmd = boxed(f"tool:filter_rows(column='{col}', op={op_str}, value='{val}')")
            obs, reward, terminated, truncated, info = safe_step(env, cmd)
            details["actions"].append(cmd)
            if terminated or truncated:
                return obs, reward, terminated, truncated, {"details": details, "info": info}

        elif op["op"] == "dispatch":
            iid = op["incident_id"]
            tid = op["team_id"]
            cmd = boxed(f"tool:dispatch_team(incident_id={iid}, team_id={tid})")
            obs, reward, terminated, truncated, info = safe_step(env, cmd)
            details["actions"].append(cmd)
            if terminated or truncated:
                return obs, reward, terminated, truncated, {"details": details, "info": info}

        elif op["op"] == "escalate":
            iid = op["incident_id"]
            level = op["level"]
            cmd = boxed(f"tool:escalate_alert(incident_id={iid}, level='{level}')")
            obs, reward, terminated, truncated, info = safe_step(env, cmd)
            details["actions"].append(cmd)
            if terminated or truncated:
                return obs, reward, terminated, truncated, {"details": details, "info": info}

        elif op["op"] == "request":
            hid = op["hospital_id"]
            item = op["item"]
            qty = op["qty"]
            cmd = boxed(f"tool:request_supplies(hospital_id={hid}, item='{item}', qty={qty})")
            obs, reward, terminated, truncated, info = safe_step(env, cmd)
            details["actions"].append(cmd)
            if terminated or truncated:
                return obs, reward, terminated, truncated, {"details": details, "info": info}

        elif op["op"] == "perimeter":
            iid = op["incident_id"]
            radius = op["radius"]
            cmd = boxed(f"tool:set_perimeter(incident_id={iid}, radius={radius})")
            obs, reward, terminated, truncated, info = safe_step(env, cmd)
            details["actions"].append(cmd)
            if terminated or truncated:
                return obs, reward, terminated, truncated, {"details": details, "info": info}

    # Step 3: Compute the metric
    if metric == "count":
        if metric_column:
            cmd = boxed(f"tool:compute_count(column='{metric_column}')")
        else:
            cmd = boxed("tool:compute_count()")
    elif metric == "sum":
        cmd = boxed(f"tool:compute_sum(column='{metric_column}')")
    elif metric == "unique":
        cmd = boxed(f"tool:unique_count(column='{metric_column}')")
    elif metric == "avg":
        cmd = boxed(f"tool:compute_avg(column='{metric_column}')")
    else:
        cmd = boxed("tool:compute_count()")

    obs, reward, terminated, truncated, info = safe_step(env, cmd)
    details["actions"].append(cmd)
    if terminated or truncated:
        return obs, reward, terminated, truncated, {"details": details, "info": info}

    # Parse the result
    last_metric = env.execution_state.get("last_metric", 0)
    candidate = last_metric
    details["derived_candidate"] = candidate

    # Step 4: Submit the answer
    cmd = boxed(f"answer:{candidate}")
    obs, reward, terminated, truncated, info = safe_step(env, cmd)
    details["actions"].append(cmd)

    extra = {"details": details, "info": info}
    return obs, reward, terminated, truncated, extra


def main():
    parser = argparse.ArgumentParser(description="EmergencyOpsTEST solver")
    parser.add_argument("--complexities", type=str, default="1,2,3,4,5",
                        help="Comma-separated complexity levels")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Comma-separated seeds")
    args = parser.parse_args()

    complexities = [int(x.strip()) for x in args.complexities.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    log_path = Path(__file__).with_name("EmergencyOpsTEST_solver_log.txt")
    log_lines = []
    start_time = time.time()

    total = 0
    successes = 0
    failures = []

    for c in complexities:
        for s in seeds:
            total += 1
            print(f"\n[Complexity {c} | Seed {s}]")
            obs, reward, terminated, truncated, extra = solve_level(c, s)
            status = "SUCCESS" if (reward == 1.0 and terminated) else "FAIL"
            if status == "SUCCESS":
                successes += 1
            else:
                failures.append((c, s))

            details = extra.get("details", {})
            actions = details.get("actions", [])
            candidate = details.get("derived_candidate")
            solution = details.get("solution")
            required_steps = details.get("required_steps")

            print(f"  => {status} | reward={reward} | terminated={terminated} | truncated={truncated}")
            print(f"  Required steps: {required_steps} | Actual steps: {len(actions)}")
            print(f"  Candidate: {candidate} | Solution: {solution}")
            print(f"  Final obs: {obs[:200]}...")

            log_lines.append(f"[Complexity {c} | Seed {s}] => {status} | reward={reward}")
            log_lines.append(f"  Required steps: {required_steps} | Actual steps: {len(actions)}")
            log_lines.append(f"  Candidate: {candidate} | Solution: {solution}")
            log_lines.append(f"  Final obs: {obs}")
            log_lines.append("  Action sequence:")
            for a in actions:
                log_lines.append(f"    {a}")
            log_lines.append("")

    elapsed = time.time() - start_time
    print("\n===== Summary =====")
    print(f"Total cases: {total}")
    print(f"Successes: {successes}")
    print(f"Failures: {total - successes}")
    if failures:
        print("Failed cases (complexity, seed):")
        for c, s in failures:
            print(f"  ({c}, {s})")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Success rate: {100.0 * successes / total:.1f}%")

    log_lines.append("\n===== Summary =====")
    log_lines.append(f"Total cases: {total}")
    log_lines.append(f"Successes: {successes}")
    log_lines.append(f"Failures: {total - successes}")
    if failures:
        log_lines.append("Failed cases (complexity, seed):")
        for c, s in failures:
            log_lines.append(f"  ({c}, {s})")
    log_lines.append(f"Elapsed: {elapsed:.2f}s")
    log_lines.append(f"Success rate: {100.0 * successes / total:.1f}%")

    try:
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        print(f"\nLog saved to: {log_path}")
    except Exception as e:
        print(f"Warning: failed to write log file: {e}")


if __name__ == "__main__":
    main()
