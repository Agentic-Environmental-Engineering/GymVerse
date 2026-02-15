from gem.core import Env
import json
import re
import copy
import importlib
import inspect
import threading
import os
from typing import List, Dict, Any, Tuple, Optional
import argparse

# --- Original Imports Preserved ---
from bfcl_eval.constants.executable_backend_config import STATELESS_CLASSES, CLASS_FILE_PATH_MAPPING
from bfcl_eval.model_handler.utils import extract_prompt_format_from_id, formulate_system_prompt, default_decode_execute_prompting
from bfcl_eval.constants.default_prompts import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import state_checker, response_checker
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import is_empty_execute_response
from bfcl_eval._llm_response_generation import get_involved_test_entries, sort_key
from bfcl_eval.utils import load_ground_truth_entry

# =============================================================================
#  BFCLEnv CLASS (Refactored)
# =============================================================================
class BFCLEnv(Env):
    def __init__(self, test_categories: List[str], run_ids: List[str] = None):
        """
        初始化环境。
        内部自动加载 Dataset 和 Ground Truth。
        """
        super().__init__()
        
        # 1. --- 数据加载逻辑 (Data Loading Inside Env) ---
        print("Loading dataset and ground truth inside Environment...")
        if run_ids is None:
            run_ids = []

        # A. 加载测试条目
        _, all_entries = get_involved_test_entries(test_categories, run_ids)
        
        # B. 加载 Ground Truth
        # 这里硬编码了常见的 key，你也可以作为参数传入
        gt_keys = ["multi_turn_base", "multi_turn_long_context", "multi_turn_miss_func", "multi_turn_miss_param"]
        possible_answer = []
        for key in gt_keys:
            try:
                possible_answer.extend(load_ground_truth_entry(key))
            except Exception as e:
                print(f"Warning: Failed to load GT for {key}: {e}")

        # C. 构建 GT 映射表
        self.ground_truth_map = {item['id']: item for item in possible_answer}

        # D. 过滤并排序 Dataset (只保留有 GT 的数据)
        raw_dataset = sorted(all_entries, key=sort_key)
        self.dataset = []
        for item in raw_dataset:
            if item['id'] in self.ground_truth_map:
                self.dataset.append(item)
            else:
                print(f"Skipping {item['id']}: No Ground Truth found.")
        
        print(f"Environment initialized with {len(self.dataset)} valid test entries.")

        # 2. --- Gym 空间定义 ---
        # self.action_space = gym.spaces.Text(max_length=50000)
        # self.observation_space = gym.spaces.Dict({
            # "messages": gym.spaces.Text(max_length=200000)
        # })

        # 3. --- 运行时状态初始化 ---
        self.test_entry_id = None
        self.data = None
        self.ground_truth = None
        self.all_turns = []
        self.initial_config = {}
        self.involved_classes = []
        self.tool_definitions = []
        self.holdout_function = {}
        
        self.active_instances = {}
        self.class_method_name_mapping = {}
        self.messages = []
        self.turn_idx = 0
        self.step_in_turn = 0
        self.max_steps_per_turn = 20
        self.is_done = False
        
        # 评测记录
        self.is_success = False
        self.current_turn_model_exec_results = []
        self.all_turn_model_exec_results = []
        self.model_snapshots = []
        self.step_counts_per_turn = []
        self.current_turn_decoded_responses = [] 
        self.all_turn_decoded_responses = []
        self.gt_snapshots = []

    def __len__(self):
        """返回数据集大小"""
        return len(self.dataset)

    def reset(self, idx: int, seed=None, options=None) -> Tuple[List[Dict], Dict]:
        """
        通过索引 (idx) 重置环境到特定的测试用例。
        """
        super().reset(seed=seed)
        
        # 1. 边界检查与数据提取
        if idx < 0 or idx >= len(self.dataset):
            raise IndexError(f"Index {idx} out of bounds for dataset size {len(self.dataset)}")

        raw_data = self.dataset[idx]
        self.test_entry_id = raw_data.get("id", "unknown")
        gt_entry = self.ground_truth_map[self.test_entry_id]

        # 2. 深拷贝数据 (防止修改源数据)
        self.data = copy.deepcopy(raw_data)
        self.ground_truth = copy.deepcopy(gt_entry["ground_truth"])
        
        # 3. 初始化核心状态
        self.all_turns = self.data.get("question", []) 
        self.initial_config = self.data.get("initial_config", {})
        self.involved_classes = self.data.get("involved_classes", [])
        self.tool_definitions = self.data.get("function", [])
        self.holdout_function = self.data.get("missed_function", {}) 

        # 处理 Diverse Hint (Optional)
        is_diverse = options.get("is_diverse", False) if options else False
        if is_diverse:
            for i in range(3):
                if i < len(self.ground_truth) and len(self.ground_truth[i]) > 0 and self.ground_truth[i][0]:
                    if i < len(self.all_turns) and len(self.all_turns[i]) > 0:
                        hint = f"\nHint: considering start with the following operation: \n{self.ground_truth[i][0]}" 
                        self.all_turns[i][0]["content"] = self.all_turns[i][0]["content"] + hint

        # 4. 初始化 Model Instances
        self._init_instances(is_gt=False)
        self.is_done = False 
        
        # 5. 预计算 GT 快照
        self.gt_snapshots = self._precompute_gt_snapshots()
        
        # 6. 重置计数器和历史
        self.messages = []
        self.turn_idx = 0
        self.step_in_turn = 0
        self.is_success = False
        self.current_turn_model_exec_results = []
        self.all_turn_model_exec_results = []
        self.model_snapshots = []
        self.step_counts_per_turn = []
        self.current_turn_decoded_responses = [] 
        self.all_turn_decoded_responses = []
        
        # 7. System Prompt
        system_prompt = self._system_prompt_pre_processing(self.tool_definitions, self.test_entry_id)
        self.messages.append({"role": "system", "content": system_prompt})
            
        # 8. 加载第一轮
        self.turn_idx = -1 
        self._load_next_turn() 
        
        return self.messages, {"id": self.test_entry_id}

    def step(self, action: str):
        if self.is_done:
            return self.messages, 0, True, True, {"id": self.test_entry_id, "success": self.is_success}
        
        self.step_in_turn += 1
        reward = 0
        done = False
        truncated = False
        info = {"id": self.test_entry_id, "success": False}
        
        # 1. 记录 Assistant 回复
        self.messages.append({"role": "assistant", "content": action})
        
        # 2. 解析
        try:
            decoded_model_responses = default_decode_execute_prompting(
                    action, has_tool_call_tag=False
            )
        except Exception:
            decoded_model_responses = None

        if decoded_model_responses:
            # === 分支 A: 执行工具 ===
            execution_results = self._execute_func_calls_logic(decoded_model_responses, self.active_instances)
            self.current_turn_model_exec_results.extend(execution_results)
            self.current_turn_decoded_responses.append(decoded_model_responses)
            
            for res, decoded_model_response in zip(execution_results, decoded_model_responses):
                self.messages.append(
                    {"role": "tool", "content": res, "name": decoded_model_response}
                )
            
            if self.step_in_turn >= self.max_steps_per_turn:
                truncated = True
                
        else:
            # === 分支 B: 文本回复/结束 ===
            self.step_counts_per_turn.append(self.step_in_turn)
            self.all_turn_model_exec_results.extend(self.current_turn_model_exec_results)
            self.all_turn_decoded_responses.append(self.current_turn_decoded_responses)
            
            self._save_model_snapshot()
            self.current_turn_decoded_responses = []
            self.current_turn_model_exec_results = []
            
            has_next_turn = self._load_next_turn()
            
            if has_next_turn:
                self.step_in_turn = 0
            else:
                done = True
                self.is_done = True 
                is_success = self._evaluate_full_episode()
                
                info["success"] = is_success
                self.is_success = is_success
                reward = 1 if is_success else 0
        
        return self.messages, reward, done, truncated, info

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _system_prompt_pre_processing(self, function_docs: list[dict], test_entry_id: str) -> str:
        prompt_format = extract_prompt_format_from_id(test_entry_id)
        return formulate_system_prompt(format_sensitivity_config=prompt_format, functions=function_docs)

    def _evaluate_full_episode(self) -> bool:
        ground_truth_list = self.ground_truth
        model_response_list = self.all_turn_decoded_responses
        if len(ground_truth_list) != len(model_response_list):
            return False
        for single_turn_ground_truth_list, single_turn_model_response_list in zip(ground_truth_list, model_response_list):
            if len(single_turn_ground_truth_list) > 0:
                if not single_turn_model_response_list or is_empty_execute_response(single_turn_model_response_list):
                    return False
        
        if len(self.gt_snapshots) != len(ground_truth_list):
            # 防止预计算出错导致长度不一致
            return False

        for idx, (model_snap, gt_snap) in enumerate(zip(self.model_snapshots, self.gt_snapshots)):
            gt_responses = gt_snap['gt_responses']
            if not gt_responses: continue 

            state_res = state_checker(model_snap['instances'], gt_snap['instances'])
            if not state_res["valid"]: return False
                
            resp_res = response_checker(model_snap['cumulative_results'], gt_snap['exec_results'], turn_index=idx)
            if not resp_res["valid"]: return False
                
        return True

    def _load_next_turn(self) -> bool:
        next_idx = self.turn_idx + 1
        if next_idx >= len(self.all_turns): return False
        
        self.turn_idx = next_idx
        current_turn_msgs = self.all_turns[self.turn_idx]
        
        if str(self.turn_idx) in self.holdout_function:
            missed_funcs = self.holdout_function[str(self.turn_idx)]
            prompt_content = DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(functions=missed_funcs)
            current_turn_msgs = [{"role": "user", "content": prompt_content}]
            
        self.messages.extend(current_turn_msgs)
        return True

    def _execute_func_calls_logic(self, func_call_list: List[str], context: Dict) -> List[str]:
        results = []
        exec_globals = context.copy()
        for func_call in func_call_list:
            func_call = self._process_method_calls(func_call)
            try:
                self._check_safety(func_call)
                res = eval(func_call, exec_globals)
                results.append(self._format_result(res))
            except Exception as e:
                results.append(f"Error: {str(e)}")
        return results
    
    def _precompute_gt_snapshots(self) -> List[Dict]:
        snapshots = []
        gt_instances = {}
        self._init_instances(is_gt=True, target_dict=gt_instances)
        for gt_calls in self.ground_truth:
            gt_results = self._execute_func_calls_logic(gt_calls, gt_instances)
            snapshots.append({
                "instances": {k: copy.deepcopy(v) for k, v in gt_instances.items()},
                "exec_results": copy.deepcopy(gt_results),
                "gt_responses": copy.deepcopy(gt_calls),
            })
        return snapshots

    def _save_model_snapshot(self):
        self.model_snapshots.append({
            "instances": {k: copy.deepcopy(v) for k, v in self.active_instances.items()},
            "cumulative_results": copy.deepcopy(self.all_turn_model_exec_results),
            "current_turn_decoded_responses": self.current_turn_decoded_responses,
            "all_turn_decoded_responses": copy.deepcopy(self.all_turn_decoded_responses),
        })

    def _init_instances(self, is_gt: bool, target_dict: Dict = None):
        if target_dict is None:
            self.active_instances = {}
            target_dict = self.active_instances
        
        if not is_gt: self.class_method_name_mapping = {}

        test_cat = self.test_entry_id.rsplit("_", 1)[0]
        long_ctx = "long_context" in test_cat or "composite" in test_cat
        
        for cls_name in self.involved_classes:
            if cls_name not in CLASS_FILE_PATH_MAPPING: continue
            try:
                mod = importlib.import_module(CLASS_FILE_PATH_MAPPING[cls_name])
                cls = getattr(mod, cls_name)
                instance = cls()
                if cls_name not in STATELESS_CLASSES:
                    cfg = copy.deepcopy(self.initial_config.get(cls_name, {}))
                    instance._load_scenario(cfg, long_context=long_ctx)
                target_dict[cls_name] = instance
                
                if not is_gt:
                    for name, _ in inspect.getmembers(instance, predicate=inspect.ismethod):
                        if not name.startswith("_"):
                            self.class_method_name_mapping[name] = cls_name
            except Exception as e:
                print(f"Init Error ({cls_name}): {e}")

    def _process_method_calls(self, func_call: str) -> str:
        def replace_function(match):
            func_name = match.group(1)
            return f"{self.class_method_name_mapping[func_name]}.{func_name}" if func_name in self.class_method_name_mapping else func_name
        return re.sub(r"\b([a-zA-Z_]\w*)\s*(?=\()", replace_function, func_call)

    def _check_safety(self, func_call: str):
        name = func_call.split("(")[0]
        if "." in name: name = name.split(".")[1]
        if name in ["kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run"]:
            raise ValueError(f"Forbidden: {name}")

    def _format_result(self, result: Any) -> str:
        if isinstance(result, (dict, list)):
            try: return json.dumps(result)
            except: pass
        return str(result)

# =============================================================================
#  HELPER FUNCTION (Client Interaction)
# =============================================================================

def get_action_from_client(client, obs: list):
    """
    Simulate or Real API call.
    """
    messages_for_api = []
    for msg in obs:
        if msg.get("role") == "tool":
            func_name = msg.get("name", "unknown_function")
            content = msg.get("content", "")
            messages_for_api.append({
                "role": "user",
                "content": f"Execution result of function '{func_name}':\n{content}"
            })
        else:
            messages_for_api.append(msg)

    if client is None:
        # Placeholder for testing without API key
        return "print('Hello World')", "Thinking process..."

    response = client.chat.completions.create(
        temperature=1,
        messages=messages_for_api,
        model='doubao-seed-1-6-thinking-250715',
    )
    
    print("Reasoning:", response.choices[0].message.reasoning_content)
    print("Action:\n", response.choices[0].message.content)
        
    return response.choices[0].message.content, response.choices[0].message.reasoning_content

# =============================================================================
#  MAIN EXECUTION (Sequential Loop)
# =============================================================================

if __name__ == "__main__":
    from openai import OpenAI
    from dotenv import load_dotenv

    load_dotenv()

    # Configuration
    result_path = './bfcl_results'
    os.makedirs(result_path, exist_ok=True)
    MAX_STEPS = 30
    
    # Optional: Set up Client
    # client = OpenAI(
    #     base_url="https://ark.cn-beijing.volces.com/api/v3", 
    #     api_key=os.getenv("DOUBAO_API_KEY")
    # )
    client = None # Set to None for dry-run/test

    # 1. 初始化环境 (自动加载所有数据)
    # 你可以在这里指定类别，Env 会自己去 load
    env = BFCLEnv(test_categories=['multi_turn'])

    print(f"Starting Sequential Execution for {len(env)} tests...")

    # 2. 顺序执行循环 (Sequential Loop)
    for i in range(len(env)):
        try:
            # A. Reset Environment
            obs, info = env.reset(idx=i)
            data_id = info["id"]
            
            save_path = os.path.join(result_path, f"{data_id}.json")
            print(f"\n-------------------- Test {i+1}/{len(env)}: {data_id} --------------------")

            # B. Episode Loop
            raw_responses = []
            step_idx = 0
            done = False
            
            while not done and step_idx < MAX_STEPS:
                action, reasoning = get_action_from_client(client, obs)
                raw_responses.append({"action": action, "reasoning_content": reasoning})
                
                obs, reward, done, truncated, info = env.step(action)
                step_idx += 1

            # C. Save Results
            print(f"Finished {data_id}. Success: {info.get('success', False)}")
            
            # 提取 GT 结果用于记录
            gt_results = []
            if hasattr(env, "gt_snapshots"):
                for snapshot in env.gt_snapshots:
                    gt_results.append(snapshot.get("exec_results", []))

            record = {
                "id": data_id,
                "success": info.get("success", False),
                "total_steps_executed": step_idx,
                "messages": env.messages,
                "ground_truth": env.ground_truth,
                "ground_truth_execution_results": gt_results,
                "raw_responses": raw_responses,
                "model_execution_results": env.all_turn_model_exec_results,
                "steps_per_turn": env.step_counts_per_turn,
            }
            
            with open(save_path, "w") as f:
                json.dump(record, f, indent=4)
                
        except Exception as e:
            print(f"!!! Error in Test {i}: {e}")
            import traceback
            traceback.print_exc()
            # Continue to next test case even if one fails
            continue

    print("All tests completed.")