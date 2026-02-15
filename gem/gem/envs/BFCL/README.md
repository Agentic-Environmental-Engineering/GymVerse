# BFCL (GEM Wrapper)

This directory contains a minimal GEM `Env` wrapper around the Berkeley Function-Calling Leaderboard (BFCL) codebase vendored under `gorilla/`.

## Environment

- Env ID: `bfcl:SingleTurn-v0`
- Class: `gem.envs.BFCL.bfcl_env:BFCLEnv`

Each episode samples one BFCL test entry. The agent action is the model output as a string (tool calls optionally wrapped in `\boxed{...}`).

## Usage
```bash
git clone https://github.com/ShishirPatil/gorilla.git
cd gorilla/inference
```
```python
from gem.envs.registration import make

env = make(
    "bfcl:SingleTurn-v0",
    test_category="simple_python",
    allowed_ids=["simple_python_0"],  # optional
)
obs, info = env.reset(seed=0)

action = r"\boxed{calculate_triangle_area(base=10, height=5, unit='units')}"
obs, reward, terminated, truncated, info = env.step(action)
```

Multi-turn categories advance turns explicitly:

```python
env = make("bfcl:SingleTurn-v0", test_category="multi_turn_base")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(r"\boxed{cd(folder='document')}")
obs, reward, terminated, truncated, info = env.step(r"\boxed{DONE}")  # next user turn
```

Agentic categories default to offline tool execution for web search:

```python
env = make("bfcl:SingleTurn-v0", test_category="web_search_base")  # execute_tools defaults to False
```

## Notes

- Supported categories:
  - Single-turn AST: `simple_*`, `parallel`, `multiple`, `parallel_multiple`, `live_*`
  - Format sensitivity: `format_sensitivity` (per-entry return format + optional `<TOOLCALL>...</TOOLCALL>`)
  - Multi-turn executable: `multi_turn_*` (use `\boxed{DONE}` to advance turns)
  - Agentic: `memory_*`, `web_search_*` (final answer checked by substring match)
  - Relevance/irrelevance: `irrelevance`, `live_relevance`, `live_irrelevance`
- BFCL code/data is located at `gem/gem/envs/BFCL/gorilla/berkeley-function-call-leaderboard`.

- memory_vector：依赖 sentence-transformers + faiss，环境里没装齐的话容易炸；目前我默认不开 execute_tools，要跑需要你确认依赖可用。
  - web_search_no_snippet：已能加载，默认离线；如果你要真执行 web search，需要网络 + SerpAPI key，并把 execute_tools=True。
export SERPAPI_API_KEY=d7cd93b94d67013deff48b53cb6ab9070fbff377

## Scoring (BFCL CLI)

If you already have generated result JSONs under `.../result/<model_name>/`, run:

`bash gem/gem/envs/BFCL/eval.sh qwen3-4b-instruct simple_python`

Scores are written to `.../score/` (see `score/data_overall.csv`).
