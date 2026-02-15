# Copyright 2025 AxonRL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Shared prompt templates used across GEM training and evaluation.

This module exists to keep inference formatting fully consistent between:
- `gem/examples/train_oat/train_oat_evolve.py` (training)
- `EnvEval/difficulty/vllm_policy.py` (evaluation)
"""


def apply_qwen3_game_template(observation: str) -> str:
    return (
        f"<|im_start|>user\nYou are playing language games. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_no_template(observation: str) -> str:
    return observation


def apply_qwen3_tool_template(question: str) -> str:
    return (
        "<|im_start|>user\n"
        "First think step by step inside <think>...</think>.\n"
        "Then execute exactly one tool call inside <action>...</action>.\n"
        "Tool call format inside <action>: [func_name1(params_name1=params_value1, params_name2=params_value2 ...)].\n"
        "If the function has no parameters, call it as [func_name1()].\n"
        "Example:\n"
        "<action>[func_name1(params_name1=params_value1, params_name2=params_value2)]</action>\n"
        "After you issue the function call, wait for the tool result.\n"
        "If the tool description is unclear, you may invoke it once to inspect the output and adjust your next call.\n\n"
        f"Question: {question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_qwen3_general_template(question: str) -> str:
    return (
        f"<|im_start|>user\nQuestion: {question}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_code_template(question: str) -> str:
    return (
        "You are an expert Python programmer. "
        "You will be given a question (problem specification) and will generate a correct "
        "Python program that matches the specification and passes all tests."
        f"\nQuestion: {question}"
        "\nPlease reason step by step, and write your code in markdown format, e.g., ```python\n# YOUR CODE HERE\n```."
    )


TEMPLATE_FACTORY = {
    "qwen3_game": apply_qwen3_game_template,
    "no": apply_no_template,
    "qwen3_general": apply_qwen3_general_template,
    "code": apply_code_template,
    "qwen3_tool": apply_qwen3_tool_template,
}

