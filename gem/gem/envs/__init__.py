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

import reasoning_gym as rg

from gem.envs.registration import register

# Register games from our implementation of TextArena and beyond.
# GuessTheNumber
register(
    "game:GuessTheNumber-v0",
    "gem.envs.game_env.guess_the_number:GuessTheNumberEnv",
)
register(
    "game:GuessTheNumber-v0-hard",
    "gem.envs.game_env.guess_the_number:GuessTheNumberEnv",
    min_number=1,
    max_number=50,
    max_turns=7,
)
register(
    "game:GuessTheNumber-v0-easy",
    "gem.envs.game_env.guess_the_number:GuessTheNumberEnv",
    min_number=1,
    max_number=10,
    max_turns=4,
)
register(
    "game:GuessTheNumber-v0-random",
    "gem.envs.game_env.guess_the_number:GuessTheNumberEnv",
    min_number=None,
    max_number=None,
    max_turns=None,
)
# Mastermind
register(
    "game:Mastermind-v0",
    "gem.envs.game_env.mastermind:MastermindEnv",
)
register(
    "game:Mastermind-v0-hard",
    "gem.envs.game_env.mastermind:MastermindEnv",
    code_length=4,
    num_numbers=8,
    max_turns=30,
    duplicate_numbers=False,
)
register(
    "game:Mastermind-v0-random",
    "gem.envs.game_env.mastermind:MastermindEnv",
    code_length=None,
    num_numbers=None,
    max_turns=None,
    duplicate_numbers=False,
)
register(
    "game:Mastermind-v0-easy",
    "gem.envs.game_env.mastermind:MastermindEnv",
    code_length=2,
    num_numbers=6,
    max_turns=10,
    duplicate_numbers=False,
)
# Minesweeper
register(
    "game:Minesweeper-v0",
    "gem.envs.game_env.minesweeper:MinesweeperEnv",
)
register(
    "game:Minesweeper-v0-easy",
    "gem.envs.game_env.minesweeper:MinesweeperEnv",
    rows=5,
    cols=5,
    num_mines=5,
    max_turns=25,
)
register(
    "game:Minesweeper-v0-hard",
    "gem.envs.game_env.minesweeper:MinesweeperEnv",
    rows=8,
    cols=8,
    num_mines=12,
    max_turns=64,
)
register(
    "game:Minesweeper-v0-random",
    "gem.envs.game_env.minesweeper:MinesweeperEnv",
    rows=None,
    cols=None,
    num_mines=None,
    max_turns=None,
)
# Wordle
register(
    "game:Wordle-v0",
    "gem.envs.game_env.wordle:WordleEnv",
)
register(
    "game:Wordle-v0-hard",
    "gem.envs.game_env.wordle:WordleEnv",
    word_length=5,
    only_real_words=True,
    max_turns=25,
)
register(
    "game:Wordle-v0-easy",
    "gem.envs.game_env.wordle:WordleEnv",
    word_length=3,
    only_real_words=True,
    max_turns=15,
)
register(
    "game:Wordle-v0-random",
    "gem.envs.game_env.wordle:WordleEnv",
    word_length=None,
    only_real_words=True,
    max_turns=None,
)
# FifteenPuzzle
register(
    "game:FifteenPuzzle-v0",
    "gem.envs.game_env.fifteen_puzzle:FifteenPuzzleEnv",
)
register(
    "game:FifteenPuzzle-v0-random",
    "gem.envs.game_env.fifteen_puzzle:FifteenPuzzleEnv",
    num_rows=None,
    max_turns=None,
)
register(
    "game:FifteenPuzzle-v0-easy",
    "gem.envs.game_env.fifteen_puzzle:FifteenPuzzleEnv",
    num_rows=2,
    max_turns=10,
)
register(
    "game:FifteenPuzzle-v0-hard",
    "gem.envs.game_env.fifteen_puzzle:FifteenPuzzleEnv",
    num_rows=4,
    max_turns=50,
)
# Hangman
register(
    "game:Hangman-v0",
    "gem.envs.game_env.hangman:HangmanEnv",
)
register(
    "game:Hangman-v0-random",
    "gem.envs.game_env.hangman:HangmanEnv",
    word_length=None,
    hardcore=False,
    max_turns=None,
)
register(
    "game:Hangman-v0-easy",
    "gem.envs.game_env.hangman:HangmanEnv",
    word_length=3,
    hardcore=False,
    max_turns=10,
)
register(
    "game:Hangman-v0-hard",
    "gem.envs.game_env.hangman:HangmanEnv",
    word_length=7,
    hardcore=False,
    max_turns=20,
)
# Sudoku
register(
    "game:Sudoku-v0",
    "gem.envs.game_env.sudoku:SudokuEnv",
)
register(
    "game:Sudoku-v0-easy",
    "gem.envs.game_env.sudoku:SudokuEnv",
    clues=10,
    max_turns=15,
    scale=4,
)
register(
    "game:Sudoku-v0-hard",
    "gem.envs.game_env.sudoku:SudokuEnv",
    clues=50,
    max_turns=50,
    scale=9,
)
register(
    "game:Sudoku-v0-random",
    "gem.envs.game_env.sudoku:SudokuEnv",
    clues=None,
    max_turns=None,
    scale=None,
)
# Tower of Hanoi
register(
    "game:TowerofHanoi-v0",
    "gem.envs.game_env.tower_of_hanoi:TowerofHanoiEnv",
)
register(
    "game:TowerofHanoi-v0-easy",
    "gem.envs.game_env.tower_of_hanoi:TowerofHanoiEnv",
    num_disks=3,
    max_turns=10,
)
register(
    "game:TowerofHanoi-v0-hard",
    "gem.envs.game_env.tower_of_hanoi:TowerofHanoiEnv",
    num_disks=5,
    max_turns=35,
)
register(
    "game:TowerofHanoi-v0-random",
    "gem.envs.game_env.tower_of_hanoi:TowerofHanoiEnv",
    num_disks=None,
    max_turns=None,
)
# Game2048
register(
    "game:Game2048-v0",
    "gem.envs.game_env.game_2048:Game2048Env",
)
register(
    "game:Game2048-v0-easy",
    "gem.envs.game_env.game_2048:Game2048Env",
    target_tile=64,
    max_turns=50,
)
register(
    "game:Game2048-v0-hard",
    "gem.envs.game_env.game_2048:Game2048Env",
    target_tile=512,
    max_turns=50,
)
register(
    "game:Game2048-v0-extreme-hard",
    "gem.envs.game_env.game_2048:Game2048Env",
    target_tile=2048,
    max_turns=100,
)
register(
    "game:Game2048-v0-random",
    "gem.envs.game_env.game_2048:Game2048Env",
    target_tile=None,
    max_turns=None,
)
# Sokoban
register(
    "game:Sokoban-v0",
    "gem.envs.game_env.sokoban:SokobanEnv",
)
register(
    "game:Sokoban-v0-easy",
    "gem.envs.game_env.sokoban:SokobanEnv",
    dim_room=(6, 6),
    num_boxes=2,
    max_turns=20,
)
register(
    "game:Sokoban-v0-hard",
    "gem.envs.game_env.sokoban:SokobanEnv",
    dim_room=(8, 8),
    num_boxes=4,
    max_turns=50,
)
register(
    "game:Sokoban-v0-random",
    "gem.envs.game_env.sokoban:SokobanEnv",
    room_size=None,
    num_boxes=None,
    max_turns=None,
)
# crosswords
register(
    "game:Crosswords-v0",
    "gem.envs.game_env.crosswords:CrosswordsEnv",
)
register(
    "game:Crosswords-v0-easy",
    "gem.envs.game_env.crosswords:CrosswordsEnv",
    hardcore=False,
    max_turns=30,
    num_words=3,
)
register(
    "game:Crosswords-v0-hard",
    "gem.envs.game_env.crosswords:CrosswordsEnv",
    hardcore=True,
    max_turns=40,
    num_words=3,
)
register(
    "game:Crosswords-v0-random",
    "gem.envs.game_env.crosswords:CrosswordsEnv",
    hardcore=None,
    max_turns=None,
    num_words=None,
)
# wordsearch
register(
    "game:WordSearch-v0",
    "gem.envs.game_env.word_search:WordSearchEnv",
)
register(
    "game:WordSearch-v0-easy",
    "gem.envs.game_env.word_search:WordSearchEnv",
    num_words=5,
    max_turns=20,
    hardcore=False,
)
register(
    "game:WordSearch-v0-hard",
    "gem.envs.game_env.word_search:WordSearchEnv",
    num_words=5,
    max_turns=20,
    hardcore=True,
)
register(
    "game:WordSearch-v0-random",
    "gem.envs.game_env.word_search:WordSearchEnv",
    num_words=None,
    max_turns=None,
    hardcore=None,
)

# Register math dataset environments

register(
    "math:ASDiv2K",
    "gem.envs.math_env:MathEnv",
    dataset_name="axon-rl/ASDIV-2k",
    question_key="problem",
    answer_key="answer",
)

register(
    "math:GSM8K",
    "gem.envs.math_env:MathEnv",
    dataset_name="axon-rl/GSM-8k",
    question_key="problem",
    answer_key="answer",
)

register(
    "math:Math12K",
    "gem.envs.math_env:MathEnv",
    dataset_name="axon-rl/MATH-12k",
    question_key="problem",
    answer_key="answer",
)

register(
    "math:Math8K-3to5",
    "gem.envs.math_env:MathEnv",
    dataset_name="axon-rl/MATH-lvl3to5-8k",
    question_key="problem",
    answer_key="answer",
)

register(
    "math:Orz57K",
    "gem.envs.math_env:MathEnv",
    dataset_name="axon-rl/ORZ-57k",
    question_key="problem",
    answer_key="answer",
)

register(
    "math:DeepScaleR40K",
    "gem.envs.math_env:MathEnv",
    dataset_name="axon-rl/DeepScaleR-40K",
    question_key="problem",
    answer_key="answer",
)

register(
    "math:Geometry3K",
    "gem.envs.math_visual_env:MathVisualEnv",
    dataset_name="axon-rl/geometry3k",
    split="train",
    image_key="images",
    question_key="problem",
    answer_key="answer",
)


# Register code dataset environments

register(
    "code:CodeContest",
    "gem.envs.code_env:CodeEnv",
    dataset_name="axon-rl/CodeContest",
    split="train",
    question_key="problem",
    test_key="tests",
)

register(
    "code:Taco8k",
    "gem.envs.code_env:CodeEnv",
    dataset_name="axon-rl/TACO-8k",
    split="train",
    question_key="problem",
    test_key="tests",
)

register(
    "code:PrimeIntellect15k",
    "gem.envs.code_env:CodeEnv",
    dataset_name="axon-rl/PrimeIntellect-15k",
    split="train",
    question_key="problem",
    test_key="tests",
)

# Register qa dataset environments

for i in [0, 1, 2, 3, 5]:
    register(
        f"logic:RuleTaker-d{i}",
        "gem.envs.qa_env:QaEnv",
        dataset_name=f"axon-rl/RuleTaker-d{i}-70k",
        split="train",
        extract_boxed=True,
        question_key="question",
        answer_key="answer",
    )

register(
    "qa:NaturalQuestions",
    "gem.envs.qa_env:QaEnv",
    dataset_name="axon-rl/NaturalQuestions",
    split="train",
    question_key="problem",
    answer_key="answer",
)

register(
    "qa:HotpotQA",
    "gem.envs.qa_env:QaEnv",
    dataset_name="axon-rl/HotpotQA",
    split="train",
    question_key="problem",
    answer_key="answer",
)

# Register datasets from ReasoningGym

for name in rg.factory.DATASETS.keys():
    register(
        f"rg:{name}",
        "gem.envs.reasoning_gym:ReasoningGymEnv",
        name=name,
        size=500,
        seed=42,
    )

# Register evaluation datasets

## MATH500
register(
    "eval:MATH500",
    "gem.envs.math_env:MathEnv",
    dataset_name="axon-rl/math-eval",
    split="math",
    question_key="problem",
    answer_key="answer",
)

## AMC
register(
    "eval:AMC",
    "gem.envs.math_env:MathEnv",
    dataset_name="axon-rl/math-eval",
    split="amc",
    question_key="problem",
    answer_key="answer",
)

## OlympiadBench
register(
    "eval:OlympiadBench",
    "gem.envs.math_env:MathEnv",
    dataset_name="axon-rl/math-eval",
    split="olympiad_bench",
    question_key="problem",
    answer_key="answer",
)

## Minerva
register(
    "eval:Minerva",
    "gem.envs.math_env:MathEnv",
    dataset_name="axon-rl/math-eval",
    split="minerva",
    question_key="problem",
    answer_key="answer",
)

## AIME24
register(
    "eval:AIME24",
    "gem.envs.math_env:MathEnv",
    dataset_name="axon-rl/math-eval",
    split="aime24",
    question_key="problem",
    answer_key="answer",
)

## The test split of deepmind/code_contests, with merged test cases.
register(
    "eval:CodeContest",
    "gem.envs.code_env:CodeEnv",
    dataset_name="axon-rl/CodeContest",
    split="test",
    question_key="problem",
    test_key="tests",
)

## QaOpen
register(
    "eval:QaOpen",
    "gem.envs.qa_env:QaEnv",
    dataset_name="google-research-datasets/nq_open",
    split="validation",
    question_key="question",
    answer_key="answer",
)

## The test split used in DeepResearcher, 512 questions per data_source
#   data_source: 2wiki, popqa, tq, hotpotqa, Bamboogle, nq, musique
data_names = [
    ("2wiki", "2Wiki"),
    ("popqa", "PopQA"),
    ("tq", "TriviaQA"),
    ("hotpotqa", "HotpotQA"),
    ("bamboogle", "Bamboogle"),
    ("nq", "NaturalQuestions"),
    ("musique", "Musique"),
]
for name, env_name in data_names:
    register(
        f"eval:{env_name}",
        "gem.envs.qa_env:QaEnv",
        dataset_name="axon-rl/search-eval",
        split=name,
        question_key="question",
        answer_key="answer",
    )

# Webshop
for obs_mode in ["html", "text", "text_rich"]:
    for split in ["train", "test"]:
        register(
            f"webshop:{split}-{obs_mode}",
            "gem.envs.webshop.webshop:WebshopEnv",
            observation_mode=obs_mode,
            max_turns=15,
            split=split,
            error_tolerance=15,
            format_error_reward=-0.1,
        )

# KORGym


register(
    "game:NumeralBricks",
    "gem.envs.KORGym.numeral_bricks:NumeralBricksEnv",
)

register(
    "game:PipeGame",
    "gem.envs.KORGym.pipe_game:PipeGameEnv",
)

register(
    "game:TowerOfHanoi",
    "gem.envs.KORGym.tower_of_hanoi:TowerOfHanoiEnv",
)

register(
    "game:Pvz",
    "gem.envs.KORGym.pvz:PVZEnv",
)

register(
    "game:Game2048",
    "gem.envs.KORGym.game_2048:KORGym2048Env",
)

register(
    "game:Minesweeper",
    "gem.envs.KORGym.minesweeper:MinesweeperEnv",
)

register(
    "game:JigsawPuzzle",
    "gem.envs.KORGym.jigsaw_puzzle:JigsawPuzzleEnv",
)

register(
    "game:CircleTheCat",
    "gem.envs.KORGym.circle_the_cat:CircleTheCatEnv",
)

register(
    "game:Jiafa",
    "gem.envs.KORGym.jiafa:JiafaEnv",
)

register(
    "game:ConstructionCompany",
    "gem.envs.KORGym.construction_company:ConstructionCompanyEnv",
)

register(
    "game:ArrowPathway",
    "gem.envs.KORGym.arrow_pathway:ArrowPathwayEnv",
)

register(
    "game:LongCat",
    "gem.envs.KORGym.long_cat:LongCatEnv",
)

register(
    "game:PlayLines",
    "gem.envs.KORGym.play_lines:PlayLinesEnv",
)

register(
    "game:PartyTime",
    "gem.envs.KORGym.party_time:PartyTimeEnv",
)

register(
    "game:Anagramania",
    "gem.envs.KORGym.anagramania:AnagramaniaEnv",
)

register(
    "game:OneTouchDrawing",
    "gem.envs.KORGym.one_touch_drawing:OneTouchDrawingEnv",
)

register(
    "game:PuzzleGame",
    "gem.envs.KORGym.puzzle_game:PuzzleGameEnv",
)

register(
    "game:CryptoWord",
    "gem.envs.KORGym.crypto_word:CryptoWordEnv",
)

register(
    "game:BlackWhiteCopy",
    "gem.envs.KORGym.black_white_copy:BlackWhiteCopyEnv",
)

register(
    "game:FillGame",
    "gem.envs.KORGym.fill_game:FillGameEnv",
)

register(
    "game:Maze",
    "gem.envs.KORGym.maze:MazeEnv",
)

register(
    "game:Tetris",
    "gem.envs.KORGym.tetris:TetrisEnv",
)

register(
    "game:GuessWord",
    "gem.envs.KORGym.guess_word:GuessWordEnv",
)

register(
    "game:City",
    "gem.envs.KORGym.city:CityEnv",
)

register(
    "game:MapPositionSimulation",
    "gem.envs.KORGym.map_position_simulation:MapPositionSimulationEnv",
)

register(
    "game:FreeTheKey",
    "gem.envs.KORGym.free_the_key:FreeTheKeyEnv",
)

register(
    "game:EmojiConnect",
    "gem.envs.KORGym.emoji_connect:EmojiConnectEnv",
)

register(
    "game:Sudoku",
    "gem.envs.KORGym.sudoku:SudokuEnv",
)

register(
    "game:BallArrange",
    "gem.envs.KORGym.ball_arrange:BallArrangeEnv",
)

register(
    "game:Snake",
    "gem.envs.KORGym.snake:SnakeEnv",
)

register(
    "game:Alien",
    "gem.envs.KORGym.alien:AlienEnv",
)

register(
    "game:AlphabeticalSorting",
    "gem.envs.KORGym.alphabetical_sorting:AlphabeticalSortingEnv",
)

register(
    "game:CityPath",
    "gem.envs.KORGym.city_path:CityPathEnv",
)

register(
    "game:MinigridNav",
    "gem.envs.KORGym.minigrid_nav:MiniGridEnv",
)

register(
    "game:DateCount",
    "gem.envs.KORGym.date_count:DateCountEnv",
)

register(
    "game:LightsOut",
    "gem.envs.KORGym.lights_out:LightsOutEnv",
)

register(
    "game:WordPuzzle",
    "gem.envs.KORGym.word_puzzle:WordPuzzleEnv",
)

register(
    "game:Wordle",
    "gem.envs.KORGym.wordle:WordleEnv",
)

register(
    "game:TrustEvolution",
    "gem.envs.KORGym.trust_evolution:TrustEvolutionEnv",
)

register(
    "game:Nullify",
    "gem.envs.KORGym.nullify:NullifyEnv",
)

register(
    "game:NpointPlus",
    "gem.envs.KORGym.npoint_plus:NpointPlusEnv",
)

register(
    "game:SpiderSolitaire",
    "gem.envs.KORGym.spider_solitaire:SpiderSolitaireEnv",
)

register(
    "game:WordEncryption",
    "gem.envs.KORGym.word_encryption:WordEncryptionEnv",
)

register(
    "game:DiagramColoring",
    "gem.envs.KORGym.diagram_coloring:DiagramColoringEnv",
)

# CodeGym

# RLVE

# synthesized_game_env




# example
register(
    "example:AlgorithmCost",
    "gem.envs.example.algorithm_AlgorithmCost_MaxLapsEnv_GEM_env:AlgorithmCostEnv",
)

register(
    "example:AlgorithmDAG",
    "gem.envs.example.algorithm_AlgorithmDAG_SpeciesCollectionEnv_GEM_env:AlgorithmDAGEnv",
)

register(
    "example:AlgorithmDebug",
    "gem.envs.example.algorithm_AlgorithmDebug_palindrome_partition_counting_env_env:AlgorithmDebugEnv",
)

register(
    "example:AlgorithmGraphComponents",
    "gem.envs.example.algorithm_AlgorithmGraphComponents_MinimumEffortPathEnv_GEM_env:AlgorithmGraphComponentsEnv",
)

register(
    "example:AlgorithmIdentification",
    "gem.envs.example.algorithm_AlgorithmIdentification_bit_equation_counting_env_env:AlgorithmIdentificationEnv",
)

register(
    "example:AlgorithmInversion",
    "gem.envs.example.algorithm_AlgorithmInversion_MaxContiguousLightAreaEnv_GEM_env:AlgorithmInversionEnv",
)

register(
    "example:AlgorithmOpsPlanner",
    "gem.envs.example.algorithm_AlgorithmOpsPlanner_max_nonadjacent_k_element_sum_env_env:AlgorithmOpsPlannerEnv",
)

register(
    "example:AlgorithmPipeline",
    "gem.envs.example.algorithm_AlgorithmPipeline_fixed_one_edge_num_spanning_tree_env_env:AlgorithmPipelineEnv",
)

register(
    "example:AlgorithmPlanner",
    "gem.envs.example.algorithm_AlgorithmPlanner_task_arrangement_env_env:AlgorithmPlannerEnv",
)

register(
    "example:AlgorithmPortfolio",
    "gem.envs.example.algorithm_AlgorithmPortfolio_fractional_programming_env_env:AlgorithmPortfolioEnv",
)

register(
    "example:AlgorithmRepair",
    "gem.envs.example.algorithm_AlgorithmRepair_circulating_grid_env_env:AlgorithmRepairEnv",
)

register(
    "example:AlgorithmRuntime",
    "gem.envs.example.algorithm_AlgorithmRuntime_SpeciesCollectionEnv_GEM_env:AlgorithmRuntimeEnv",
)

register(
    "example:AlgorithmSATDecision",
    "gem.envs.example.algorithm_AlgorithmSATDecision_WorkshopRoomAssignmentEnv_GEM_env:AlgorithmSATDecisionEnv",
)

register(
    "example:AlgorithmScheduling",
    "gem.envs.example.algorithm_AlgorithmScheduling_bounded_mean_subarray_counting_env_env:AlgorithmSchedulingEnv",
)

register(
    "example:AlgorithmSelection",
    "gem.envs.example.algorithm_AlgorithmSelection_grid_bfs_env_env:AlgorithmSelectionEnv",
)

register(
    "example:AlgorithmSortingLab",
    "gem.envs.example.algorithm_AlgorithmSortingLab_path_no_going_back_counting_env_env:AlgorithmSortingLabEnv",
)

register(
    "example:AlgorithmSorting",
    "gem.envs.example.algorithm_AlgorithmSorting_salad_bar_env_env:AlgorithmSortingEnv",
)

register(
    "example:AlgorithmSynthesis",
    "gem.envs.example.algorithm_AlgorithmSynthesis_MoveZeroesEnv_GEM_env:AlgorithmSynthesisEnv",
)

register(
    "example:AlgorithmTrace",
    "gem.envs.example.algorithm_AlgorithmTrace_grid_local_minimum_counting_env_env:AlgorithmTraceEnv",
)

register(
    "example:AlgorithmWorkbench",
    "gem.envs.example.algorithm_AlgorithmWorkbench_delta_min_popcount_env_env:AlgorithmWorkbenchEnv",
)

register(
    "example:AlgorithmicSorting",
    "gem.envs.example.algorithm_AlgorithmicSorting_pipeline_arrangement_env_env:AlgorithmicSortingEnv",
)

register(
    "example:BSTAudit",
    "gem.envs.example.algorithm_BSTAudit_KingdomConnectivityEnv_GEM_env:BSTAuditEnv",
)

register(
    "example:BooleanCircuitSynthesis",
    "gem.envs.example.algorithm_BooleanCircuitSynthesis_range_four_sequence_construction_env_env:BooleanCircuitSynthesisEnv",
)

register(
    "example:GraphColoringAlgorithm",
    "gem.envs.example.algorithm_GraphColoringAlgorithm_matrix_permutation_main_diagonal_one_env_env:GraphColoringAlgorithmEnv",
)

register(
    "example:GraphCycleStreaming",
    "gem.envs.example.algorithm_GraphCycleStreaming_EmployeeSchedulingEnv_GEM_env:GraphCycleStreamingEnv",
)

register(
    "example:MonotoneSearch",
    "gem.envs.example.algorithm_MonotoneSearch_ShortestPathEnv_GEM_env:MonotoneSearchEnv",
)

register(
    "example:SortingStrategy",
    "gem.envs.example.algorithm_SortingStrategy_polya_model_env_env:SortingStrategyEnv",
)

register(
    "example:WeightedCompletionTimeScheduling",
    "gem.envs.example.algorithm_WeightedCompletionTimeScheduling_weighted_binarytree_env_env:WeightedCompletionTimeSchedulingEnv",
)

register(
    "example:CodeAudit",
    "gem.envs.example.code_CodeAudit_AdCampaignEnv_GEM_env:CodeAuditEnv",
)

register(
    "example:CodeAudit",
    "gem.envs.example.code_CodeAudit_MaxContiguousLightAreaEnv_GEM_env:CodeAuditEnv",
)

register(
    "example:CodeAudit",
    "gem.envs.example.code_CodeAudit_MinimumEffortPathEnv_GEM_env:CodeAuditEnv",
)

register(
    "example:CodeBugAudit",
    "gem.envs.example.code_CodeBugAudit_EmployeeSchedulingEnv_GEM_env:CodeBugAuditEnv",
)

register(
    "example:CodeBugFix",
    "gem.envs.example.code_CodeBugFix_GreaterElementsCountEnv_GEM_env:CodeBugFixEnv",
)

register(
    "example:CodeBugTriage",
    "gem.envs.example.code_CodeBugTriage_GreaterElementsCountEnv_GEM_env:CodeBugTriageEnv",
)

register(
    "example:CodeComplexity",
    "gem.envs.example.code_CodeComplexity_EmployeeSchedulingEnv_GEM_env:CodeComplexityEnv",
)

register(
    "example:CodeComplexity",
    "gem.envs.example.code_CodeComplexity_MinimumEffortPathEnv_GEM_env:CodeComplexityEnv",
)

register(
    "example:CodeComplexity",
    "gem.envs.example.code_CodeComplexity_SpeciesCollectionEnv_GEM_env:CodeComplexityEnv",
)

register(
    "example:CodeCoveragePlanner",
    "gem.envs.example.code_CodeCoveragePlanner_ShortestPathEnv_GEM_env:CodeCoveragePlannerEnv",
)

register(
    "example:CodeDebug",
    "gem.envs.example.code_CodeDebug_LongestAlphabeticalSubstringEnv_GEM_env:CodeDebugEnv",
)

register(
    "example:CodeDependencySCC",
    "gem.envs.example.code_CodeDependencySCC_MinimumEffortPathEnv_GEM_env:CodeDependencySCCEnv",
)

register(
    "example:CodeDependency",
    "gem.envs.example.code_CodeDependency_EmployeeSchedulingEnv_GEM_env:CodeDependencyEnv",
)

register(
    "example:CodeFunctionEval",
    "gem.envs.example.code_CodeFunctionEval_MaxLapsEnv_GEM_env:CodeFunctionEvalEnv",
)

register(
    "example:CodeFunctionInference",
    "gem.envs.example.code_CodeFunctionInference_MinOperationsEnv_GEM_env:CodeFunctionInferenceEnv",
)

register(
    "example:CodeFunctionInference",
    "gem.envs.example.code_CodeFunctionInference_ShortestPathEnv_GEM_env:CodeFunctionInferenceEnv",
)

register(
    "example:CodeGraphRecursion",
    "gem.envs.example.code_CodeGraphRecursion_EmployeeSchedulingEnv_GEM_env:CodeGraphRecursionEnv",
)

register(
    "example:CodeQualityAudit",
    "gem.envs.example.code_CodeQualityAudit_KingdomConnectivityEnv_GEM_env:CodeQualityAuditEnv",
)

register(
    "example:CodeRepair",
    "gem.envs.example.code_CodeRepair_GridRouteCountEnv_GEM_env:CodeRepairEnv",
)

register(
    "example:CodeReview",
    "gem.envs.example.code_CodeReview_EqualSubsetPartitionEnv_GEM_env:CodeReviewEnv",
)

register(
    "example:CodeSynthesis",
    "gem.envs.example.code_CodeSynthesis_GridRouteCountEnv_GEM_env:CodeSynthesisEnv",
)

register(
    "example:CodeTrace",
    "gem.envs.example.code_CodeTrace_FlightBookingEnv_GEM_env:CodeTraceEnv",
)

register(
    "example:CodebaseMetrics",
    "gem.envs.example.code_CodebaseMetrics_MaxLapsEnv_GEM_env:CodebaseMetricsEnv",
)

register(
    "example:CodebaseSummary",
    "gem.envs.example.code_CodebaseSummary_SpeciesCollectionEnv_GEM_env:CodebaseSummaryEnv",
)

register(
    "example:CodebaseTodoAudit",
    "gem.envs.example.code_CodebaseTodoAudit_MaxContiguousLightAreaEnv_GEM_env:CodebaseTodoAuditEnv",
)

register(
    "example:CodebaseTodoAudit",
    "gem.envs.example.code_CodebaseTodoAudit_SpeciesCollectionEnv_GEM_env:CodebaseTodoAuditEnv",
)

register(
    "example:LayeredDependency",
    "gem.envs.example.code_LayeredDependency_KingdomConnectivityEnv_GEM_env:LayeredDependencyEnv",
)

register(
    "example:PackageResolver",
    "gem.envs.example.code_PackageResolver_ThreeSumEnv_GEM_env:PackageResolverEnv",
)

register(
    "example:RepoRiskIndex",
    "gem.envs.example.code_RepoRiskIndex_MinPairDiffSumEnv_GEM_env:RepoRiskIndexEnv",
)

register(
    "example:AlchemyCraft",
    "gem.envs.example.game_AlchemyCraft_pipeline_arrangement_env_env:AlchemyCraftEnv",
)

register(
    "example:ArcaneBrewery",
    "gem.envs.example.game_ArcaneBrewery_game_2048_env:ArcaneBreweryEnv",
)

register(
    "example:ArenaStrike",
    "gem.envs.example.game_ArenaStrike_guess_word_env:ArenaStrikeEnv",
)

register(
    "example:BattleStamina",
    "gem.envs.example.game_BattleStamina_MaxSubarraySumEnv_GEM_env:BattleStaminaEnv",
)

register(
    "example:CardComboGame",
    "gem.envs.example.game_CardComboGame_StringOperationsEnv_GEM_env:CardComboGameEnv",
)

register(
    "example:CipherLockGame",
    "gem.envs.example.game_CipherLockGame_MinPairDiffSumEnv_GEM_env:CipherLockGameEnv",
)

register(
    "example:CipherLockGame",
    "gem.envs.example.game_CipherLockGame_SmallestPrimeGteEnv_GEM_env:CipherLockGameEnv",
)

register(
    "example:CombatPlannerGame",
    "gem.envs.example.game_CombatPlannerGame_SubarraySumPrimeEnv_GEM_env:CombatPlannerGameEnv",
)

register(
    "example:DetectiveDeductionGame",
    "gem.envs.example.game_DetectiveDeductionGame_EmployeeSchedulingEnv_GEM_env:DetectiveDeductionGameEnv",
)

register(
    "example:DetectiveMystery",
    "gem.envs.example.game_DetectiveMystery_UniqueElementsFindingEnv_GEM_env:DetectiveMysteryEnv",
)

register(
    "example:DungeonHeist",
    "gem.envs.example.game_DungeonHeist_xor_equation_counting_env_env:DungeonHeistEnv",
)

register(
    "example:DungeonPartyBuilder",
    "gem.envs.example.game_DungeonPartyBuilder_max_nonadjacent_k_element_sum_env_env:DungeonPartyBuilderEnv",
)

register(
    "example:DungeonPartyPlanning",
    "gem.envs.example.game_DungeonPartyPlanning_PresentationSchedulingEnv_GEM_env:DungeonPartyPlanningEnv",
)

register(
    "example:DungeonRaidPlanning",
    "gem.envs.example.game_DungeonRaidPlanning_WorkshopRoomAssignmentEnv_GEM_env:DungeonRaidPlanningEnv",
)

register(
    "example:DungeonScout",
    "gem.envs.example.game_DungeonScout_party_time_env:DungeonScoutEnv",
)

register(
    "example:GauntletMaxDamage",
    "gem.envs.example.game_GauntletMaxDamage_MaxContiguousLightAreaEnv_GEM_env:GauntletMaxDamageEnv",
)

register(
    "example:GuildRegionsGame",
    "gem.envs.example.game_GuildRegionsGame_ConstellationCountingEnv_GEM_env:GuildRegionsGameEnv",
)

register(
    "example:GuildSplitGame",
    "gem.envs.example.game_GuildSplitGame_tree_coloring_env_env:GuildSplitGameEnv",
)

register(
    "example:MetroidMapGame",
    "gem.envs.example.game_MetroidMapGame_SpeciesCollectionEnv_GEM_env:MetroidMapGameEnv",
)

register(
    "example:MysteryMansionGame",
    "gem.envs.example.game_MysteryMansionGame_DistinctShiftedArraysEnv_GEM_env:MysteryMansionGameEnv",
)

register(
    "example:NetworkPlanningGame",
    "gem.envs.example.game_NetworkPlanningGame_game_2048_env:NetworkPlanningGameEnv",
)

register(
    "example:DungeonScout",
    "gem.envs.example.game_DungeonScout_party_time_env:DungeonScoutEnv",
)
register(
    "example:RaidPartyPlanner",
    "gem.envs.example.game_RaidPartyPlanner_difference_constraint_system_dag_env:RaidPartyPlannerEnv",
)
register(
    "example:RhythmSequencerGame",
    "gem.envs.example.game_RhythmSequencerGame_ZigzagPatternEnv_GEM_env:RhythmSequencerGameEnv",
)

register(
    "example:RuneLockDeduction",
    "gem.envs.example.game_RuneLockDeduction_ShortestPathEnv_GEM_env:RuneLockDeductionEnv",
)

register(
    "example:RuneRingPuzzle",
    "gem.envs.example.game_RuneRingPuzzle_black_white_copy_env:RuneRingPuzzleEnv",
)

register(
    "example:ArenaStrike",
    "gem.envs.example.game_ArenaStrike_guess_word_env:ArenaStrikeEnv",
)

# 0102 add：
register(
    "example:ArtisanToolroom",
    "gem.envs.example.tool_use_ArtisanToolroom_minimum_dominating_interval_env_env:ArtisanToolroomEnv",
)
register(
    "example:DataTinkerBench",
    "gem.envs.example.tool_use_DataTinkerBench_MaxContiguousLightAreaEnv_GEM_env:DataTinkerBenchEnv",
)
register(
    "example:ForgeBenchOrchestrator",
    "gem.envs.example.tool_use_ForgeBenchOrchestrator_tree_change_one_edge_diameter_env_env:ForgeBenchOrchestratorEnv",
)
register(
    "example:InstrumentChainCrafter",
    "gem.envs.example.tool_use_InstrumentChainCrafter_bucket_sorting_env_env:InstrumentChainCrafterEnv",
)
register(
    "example:MacroCrafter",
    "gem.envs.example.tool_use_MacroCrafter_triumphal_arch_env_env:MacroCrafterEnv",
)
register(
    "example:OperatorOrchestra",
    "gem.envs.example.tool_use_OperatorOrchestra_EmployeeSchedulingEnv_GEM_env:OperatorOrchestraEnv",
)
register(
    "example:OpsForgeOrchestrator",
    "gem.envs.example.tool_use_OpsForgeOrchestrator_myj_env_env:OpsForgeOrchestratorEnv",
)
register(
    "example:OpsKitComposer",
    "gem.envs.example.tool_use_OpsKitComposer_squ_squarks_env_env:OpsKitComposerEnv",
)
register(
    "example:OpsLoom",
    "gem.envs.example.tool_use_OpsLoom_MinPairDiffSumEnv_GEM_env:OpsLoomEnv",
)
register(
    "example:OpsMaestro",
    "gem.envs.example.tool_use_OpsMaestro_two_sat_env_env:OpsMaestroEnv",
)
register(
    "example:PipelineConductor",
    "gem.envs.example.tool_use_PipelineConductor_sum_lcm_env_env:PipelineConductorEnv",
)
register(
    "example:SubroutineBlacksmithForge",
    "gem.envs.example.SubroutineBlacksmithForge_env:SubroutineBlacksmithForgeEnv",
)
register(
    "example:ToolchestMaestro",
    "gem.envs.example.tool_use_ToolchestMaestro_KingdomConnectivityEnv_GEM_env:ToolchestMaestroEnv",
)
register(
    "example:ToolCrafterRelay",
    "gem.envs.example.tool_use_ToolCrafterRelay_josephus_env_env:ToolCrafterRelayEnv",
)
register(
    "example:ToolLattice",
    "gem.envs.example.tool_use_ToolLattice_prefixuffix_env_env:ToolLatticeEnv",
)
register(
    "example:TraceTriageFoundry",
    "gem.envs.example.tool_use_TraceTriageFoundry_MinimumEffortPathEnv_GEM_env:TraceTriageFoundryEnv",
)
register(
    "example:UtilityLoom",
    "gem.envs.example.tool_use_UtilityLoom_SpeciesCollectionEnv_GEM_env:UtilityLoomEnv",
)
register(
    "example:WorkbenchConductor",
    "gem.envs.example.tool_use_WorkbenchConductor_no_adjacent_girl_counting_env_env:WorkbenchConductorEnv",
)
register(
    "example:WorkbenchMaestro",
    "gem.envs.example.tool_use_WorkbenchMaestro_StringPartitionEnv_GEM_env:WorkbenchMaestroEnv",
)
register(
    "example:WorkbenchOrchestrator",
    "gem.envs.example.tool_use_WorkbenchOrchestrator_matrix_rmq_counting_env_env:WorkbenchOrchestratorEnv",
)
register(
    "example:WorkflowAnvil",
    "gem.envs.example.tool_use_WorkflowAnvil_stunt_flying_env_env:WorkflowAnvilEnv",
)

# 0104 add:
register(
    "example:ToolchainComposer",
    "gem.envs.example.tool_use_ToolchainComposer_pol_polarization_env_env:ToolchainComposerEnv",
)
register(
    "example:ToolchainConductor",
    "gem.envs.example.tool_use_ToolchainConductor_jug_puzzle_env_env:ToolchainConductorEnv",
)
register(
    "example:TavernLineage",
    "gem.envs.example.game_TavernLineage_MazeEscapeEnv_GEM_env:TavernLineageEnv",
)
register(
    "example:AlgoWorkbench",
    "gem.envs.example.algorithm_AlgoWorkbench_SubmatrixSumEnv_GEM_env:AlgoWorkbenchEnv",
)
register(
    "example:SignalHarnessMapper",
    "gem.envs.example.tool_use_SignalHarnessMapper_subgraph_isomorphism_env_env:SignalHarnessMapperEnv",
)
register(
    "example:TelemetrySmith",
    "gem.envs.example.tool_use_TelemetrySmith_WorkshopRoomAssignmentEnv_GEM_env:TelemetrySmithEnv",
)
register(
    "example:RumorRingSifter",
    "gem.envs.example.game_RumorRingSifter_ArithmeticTripletEnv_GEM_env:RumorRingSifterEnv",
)
register(
    "example:RecurrenceCrafter",
    "gem.envs.example.dynamicprogramming_algorithm_RecurrenceCrafter_BambooBundlingEnv_GEM_env:RecurrenceCrafterEnv",
)
register(
    "example:AquaConflux",
    "gem.envs.example.puzzle_game_AquaConflux_city_path_env:AquaConfluxEnv",
)
register(
    "example:PeriodSeeker",
    "gem.envs.example.algorithm_PeriodSeeker_MaxSubgridSumEnv_GEM_env:PeriodSeekerEnv",
)
register(
    "example:AlgoReliquary",
    "gem.envs.example.algorithm_AlgoReliquary_LargestSquareEnv_GEM_env:AlgoReliquaryEnv",
)
register(
    "example:GlyphConclave",
    "gem.envs.example.puzzle_game_GlyphConclave_party_time_env:GlyphConclaveEnv",
)
register(
    "example:ClueCrafters",
    "gem.envs.example.puzzle_game_ClueCrafters_fractional_programming_env_env:ClueCraftersEnv",
)
register(
    "example:RecurrenceCartographer",
    "gem.envs.example.dynamicprogramming_algorithm_RecurrenceCartographer_OceanViewBuildingsEnv_GEM_env:RecurrenceCartographerEnv",
)


# exp1：
register(
    "example:CodeBugTriageNoFeedback",
    "gem.envs.example.code_CodeBugTriageNoFeedback_GreaterElementsCountEnv_GEM_env:CodeBugTriageNoFeedbackEnv",
)

register(
    "example:CodeBugTriageProcessReward",
    "gem.envs.example.code_CodeBugTriageProcessReward_GreaterElementsCountEnv_GEM_env:CodeBugTriageProcessRewardEnv",
)


# BFCL
register(
    "bfcl:SingleTurn-v0",
    "gem.envs.BFCL.bfcl_env:BFCLEnv",
)

register(
    "example:RunicLockGame",
    "gem.envs.example.game_RunicLockGame_MinJobDurationEnv_GEM_env:RunicLockGameEnv",
)

register(
    "example:RunicPotionGame",
    "gem.envs.example.game_RunicPotionGame_LongestTwoDistinctSubstringEnv_GEM_env:RunicPotionGameEnv",
)

register(
    "example:StealthHeist",
    "gem.envs.example.game_StealthHeist_tree_coloring_env_env:StealthHeistEnv",
)

register(
    "example:TacticsTeamBuilder",
    "gem.envs.example.game_TacticsTeamBuilder_StringPartitionEnv_GEM_env:TacticsTeamBuilderEnv",
)

register(
    "example:VaultCodeQuest",
    "gem.envs.example.game_VaultCodeQuest_MaxLapsEnv_GEM_env:VaultCodeQuestEnv",
)

register(
    "example:ModularRecoveryMath",
    "gem.envs.example.math_ModularRecoveryMath_guess_word_env:ModularRecoveryMathEnv",
)

# new example

register(
    "example:ArmadaTally",
    "gem.envs.new_example.game_ArmadaTally_city_env:ArmadaTallyEnv",
)

register(
    "example:BeastSignTracker",
    "gem.envs.new_example.game_BeastSignTracker_alphabetical_sorting_env:BeastSignTrackerEnv",
)

register(
    "example:ChronoWeaver",
    "gem.envs.new_example.game_ChronoWeaver_arrow_pathway_env:ChronoWeaverEnv",
)

register(
    "example:CitadelRosterForge",
    "gem.envs.new_example.game_CitadelRosterForge_tree_coloring_env_env:CitadelRosterForgeEnv",
)

register(
    "example:ComboChoreographer",
    "gem.envs.new_example.ggame_ComboChoreographer_pipeline_arrangement_env_env:ComboChoreographerEnv",
)

register(
    "example:FestivalCensusMarshal",
    "gem.envs.new_example.game_FestivalCensusMarshal_party_time_env:FestivalCensusMarshalEnv",
)

register(
    "example:HexDuelLoadoutOracle",
    "gem.envs.new_example.game_HexDuelLoadoutOracle_graph_contain_tree_counting_env_env:HexDuelLoadoutOracleEnv",
)

register(
    "example:AetherForge",
    "gem.envs.new_example.game_AetherForge_range_four_sequence_construction_env_env:AetherForgeEnv",
)

# 0108 add：

register(
    "example:AbacusCascade",
    "gem.envs.example.calculationgame_AbacusCascade_map_position_simulation_env:AbacusCascadeEnv",
)

register(
    "example:FormulaFoundry",
    "gem.envs.example.calculationgame_FormulaFoundry_one_touch_drawing_env:FormulaFoundryEnv",
)

register(
    "example:SumSieveArithmetica",
    "gem.envs.example.calculationgame_SumSieveArithmetica_city_env:SumSieveArithmeticaEnv",
)

register(
    "example:BellmanVerifier",
    "gem.envs.example.dynamicprogramming_algorithm_BellmanVerifier_SymmetricPaintingEnv_GEM_env:BellmanVerifierEnv",
)

register(
    "example:RecurrenceRanger",
    "gem.envs.example.dynamicprogramming_algorithm_RecurrenceRanger_LearningMethodGroupingEnv_GEM_env:RecurrenceRangerEnv",
)

register(
    "example:RecurrenceScribe",
    "gem.envs.example.dynamicprogramming_algorithm_RecurrenceScribe_LongestIncreasingSubarrayEnv_GEM_env:RecurrenceScribeEnv",
)

register(
    "example:LexemeCipher",
    "gem.envs.example.language_game_LexemeCipher_double_palindromic_string_counting_env_env:LexemeCipherEnv",
)

register(
    "example:RhymeMorphMinimax",
    "gem.envs.example.language_game_RhymeMorphMinimax_min_path_cover_dag_env_env:RhymeMorphMinimaxEnv",
)

register(
    "example:AxiomOracleDeduction",
    "gem.envs.example.logic_game_AxiomOracleDeduction_trust_evolution_env:AxiomOracleDeductionEnv",
)

register(
    "example:AxiomSnare",
    "gem.envs.example.logic_game_AxiomSnare_circle_the_cat_env:AxiomSnareEnv",
)

register(
    "example:ClauseForge",
    "gem.envs.example.logic_game_ClauseForge_map_position_simulation_env:ClauseForgeEnv",
)

register(
    "example:DeductionLedger",
    "gem.envs.example.logic_game_DeductionLedger_diagram_coloring_env:DeductionLedgerEnv",
)

register(
    "example:ParityOracleDeduction",
    "gem.envs.example.logic_game_ParityOracleDeduction_one_touch_drawing_env:ParityOracleDeductionEnv",
)

register(
    "example:ParityParlor",
    "gem.envs.example.logic_game_ParityParlor_play_lines_env:ParityParlorEnv",
)

register(
    "example:TruthKnot",
    "gem.envs.example.logic_game_TruthKnot_minesweeper_env:TruthKnotEnv",
)

register(
    "example:InferenceOrbit",
    "gem.envs.example.logic_InferenceOrbit_city_env:InferenceOrbitEnv",
)

register(
    "example:SequentSmith",
    "gem.envs.example.logic_SequentSmith_word_encryption_env:SequentSmithEnv",
)

register(
    "example:AlgebraCanonForge",
    "gem.envs.example.math_AlgebraCanonForge_grid_component_env_env:AlgebraCanonForgeEnv",
)

register(
    "example:AlgebraFlow",
    "gem.envs.example.math_AlgebraFlow_pol_polarization_env_env:AlgebraFlowEnv",
)

register(
    "example:AlgebraPathfinder",
    "gem.envs.example.math_AlgebraPathfinder_prefixuffix_env_env:AlgebraPathfinderEnv",
)

register(
    "example:CoprimeChooser",
    "gem.envs.example.math_CoprimeChooser_imp_party_env_env:CoprimeChooserEnv",
)

register(
    "example:PrimeWeaveMatching",
    "gem.envs.example.math_PrimeWeaveMatching_different_color_pairing_env_env:PrimeWeaveMatchingEnv",
)

register(
    "example:RationalCraft",
    "gem.envs.example.math_RationalCraft_polya_model_env_env:RationalCraftEnv",
)

register(
    "example:TheoremCrafter",
    "gem.envs.example.math_TheoremCrafter_coin_square_game_env_env:TheoremCrafterEnv",
)

register(
    "example:TheoremTactician",
    "gem.envs.example.math_TheoremTactician_sum_divisor_num_env_env:TheoremTacticianEnv",
)

register(
    "example:TheoremTinker",
    "gem.envs.example.math_TheoremTinker_sum_gcd_env_env:TheoremTinkerEnv",
)

register(
    "example:LabyrinthCourier",
    "gem.envs.example.maze_game_LabyrinthCourier_sorting_env_env:LabyrinthCourierEnv",
)

register(
    "example:ClueConclave",
    "gem.envs.example.reasoning_game_ClueConclave_tree_center_env_env:ClueConclaveEnv",
)

register(
    "example:MonotoneOracleDeduction",
    "gem.envs.example.reasoning_game_MonotoneOracleDeduction_bucket_sorting_env_env:MonotoneOracleDeductionEnv",
)

register(
    "example:BeaconCorridorNavigator",
    "gem.envs.example.spatial_game_BeaconCorridorNavigator_free_the_key_env:BeaconCorridorNavigatorEnv",
)

register(
    "example:StarwayNavigator",
    "gem.envs.example.spatial_game_StarwayNavigator_city_path_env:StarwayNavigatorEnv",
)

# 0112 add scaleDiff 8envs：
register(
    "example:RunicPotionGameScalediff",
    "gem.envs.example.game_RunicPotionGameScalediff_LongestTwoDistinctSubstringEnv_GEM_env:RunicPotionGameScalediffEnv",
)

register(
    "example:RuneRingPuzzleScalediff",
    "gem.envs.example.game_RuneRingPuzzleScalediff_black_white_copy_env:RuneRingPuzzleScalediffEnv",
)

register(
    "example:AlgorithmCostScalediff",
    "gem.envs.example.algorithm_AlgorithmCostScalediff_MaxLapsEnv_GEM_env:AlgorithmCostScalediffEnv",
)

register(
    "example:AlgorithmRuntimeScalediff",
    "gem.envs.example.algorithm_AlgorithmRuntimeScalediff_SpeciesCollectionEnv_GEM_env:AlgorithmRuntimeScalediffEnv",
)

register(
    "example:AlgorithmSATDecisionScalediff",
    "gem.envs.example.algorithm_AlgorithmSATDecisionScalediff_WorkshopRoomAssignmentEnv_GEM_env:AlgorithmSATDecisionScalediffEnv",
)

register(
    "example:AlgorithmSchedulingScalediff",
    "gem.envs.example.algorithm_AlgorithmSchedulingScalediff_bounded_mean_subarray_counting_env_env:AlgorithmSchedulingScalediffEnv",
)

register(
    "example:BSTAuditScalediff",
    "gem.envs.example.algorithm_BSTAuditScalediff_KingdomConnectivityEnv_GEM_env:BSTAuditScalediffEnv",
)

register(
    "example:DataTinkerBenchScalediff",
    "gem.envs.example.tool_use_DataTinkerBenchScalediff_MaxContiguousLightAreaEnv_GEM_env:DataTinkerBenchScalediffEnv",
)

# 0112 add scaleDiff v2 8envs：
register(
    "example:RunicPotionGameScalediffv2",
    "gem.envs.example.game_RunicPotionGameScalediffv2_LongestTwoDistinctSubstringEnv_GEM_env:RunicPotionGameScalediffv2Env",
)

register(
    "example:RuneRingPuzzleScalediffv2",
    "gem.envs.example.game_RuneRingPuzzleScalediffv2_black_white_copy_env:RuneRingPuzzleScalediffv2Env",
)

register(
    "example:AlgorithmCostScalediffv2",
    "gem.envs.example.algorithm_AlgorithmCostScalediffv2_MaxLapsEnv_GEM_env:AlgorithmCostScalediffv2Env",
)

register(
    "example:AlgorithmRuntimeScalediffv2",
    "gem.envs.example.algorithm_AlgorithmRuntimeScalediffv2_SpeciesCollectionEnv_GEM_env:AlgorithmRuntimeScalediffv2Env",
)

register(
    "example:AlgorithmSATDecisionScalediffv2",
    "gem.envs.example.algorithm_AlgorithmSATDecisionScalediffv2_WorkshopRoomAssignmentEnv_GEM_env:AlgorithmSATDecisionScalediffv2Env",
)

register(
    "example:AlgorithmSchedulingScalediffv2",
    "gem.envs.example.algorithm_AlgorithmSchedulingScalediffv2_bounded_mean_subarray_counting_env_env:AlgorithmSchedulingScalediffv2Env",
)

register(
    "example:BSTAuditScalediffv2",
    "gem.envs.example.algorithm_BSTAuditScalediffv2_KingdomConnectivityEnv_GEM_env:BSTAuditScalediffv2Env",
)

register(
    "example:DataTinkerBenchScalediffv2",
    "gem.envs.example.tool_use_DataTinkerBenchScalediffv2_MaxContiguousLightAreaEnv_GEM_env:DataTinkerBenchScalediffv2Env",
)


# 0112 add codegym4test：
register(
    "codegym:BalancedBracketsEnv",
    "gem.envs.CodeGym.BalancedBracketsEnv_GEM:BalancedBracketsEnvGEM",
)
register(
    "codegym:ConnectedComponentsEnv",
    "gem.envs.CodeGym.ConnectedComponentsEnv_GEM:ConnectedComponentsEnvGEM",
)
register(
    "codegym:DailyTemperaturesEnv",
    "gem.envs.CodeGym.DailyTemperaturesEnv_GEM:DailyTemperaturesEnvGEM",
)
register(
    "codegym:DigitalRootEnv",
    "gem.envs.CodeGym.DigitalRootEnv_GEM:DigitalRootEnvGEM",
)
register(
    "codegym:EncryptionVerificationEnv",
    "gem.envs.CodeGym.EncryptionVerificationEnv_GEM:EncryptionVerificationEnvGEM",
)
register(
    "codegym:LongestEqualSubstringEnv",
    "gem.envs.CodeGym.LongestEqualSubstringEnv_GEM:LongestEqualSubstringEnvGEM",
)
register(
    "codegym:LongestPalindromicSubsequenceEnv",
    "gem.envs.CodeGym.LongestPalindromicSubsequenceEnv_GEM:LongestPalindromicSubsequenceEnvGEM",
)
register(
    "codegym:MaxNonAdjacentSumEnv",
    "gem.envs.CodeGym.MaxNonAdjacentSumEnv_GEM:MaxNonAdjacentSumEnvGEM",
)
register(
    "codegym:MaxSubarraySumEnv",
    "gem.envs.CodeGym.MaxSubarraySumEnv_GEM:MaxSubarraySumEnvGEM",
)
register(
    "codegym:MaxSubsegmentSumEnv",
    "gem.envs.CodeGym.MaxSubsegmentSumEnv_GEM:MaxSubsegmentSumEnvGEM",
)
register(
    "codegym:MinWordsToFormTargetEnv",
    "gem.envs.CodeGym.MinWordsToFormTargetEnv_GEM:MinWordsToFormTargetEnvGEM",
)
register(
    "codegym:NearbyAlmostDuplicateEnv",
    "gem.envs.CodeGym.NearbyAlmostDuplicateEnv_GEM:NearbyAlmostDuplicateEnvGEM",
)
register(
    "codegym:StampArrangementEnv",
    "gem.envs.CodeGym.StampArrangementEnv_GEM:StampArrangementEnvGEM",
)
register(
    "codegym:StringSplitCheckerEnv",
    "gem.envs.CodeGym.StringSplitCheckerEnv_GEM:StringSplitCheckerEnvGEM",
)
register(
    "codegym:SubstringIndicesEnv",
    "gem.envs.CodeGym.SubstringIndicesEnv_GEM:SubstringIndicesEnvGEM",
)
register(
    "codegym:TrailingZerosEnv",
    "gem.envs.CodeGym.TrailingZerosEnv_GEM:TrailingZerosEnvGEM",
)

# 0114 add for difficulty
register(
    "difficulty:AutomationLab",
    "gem.envs.difficulty.tool_use_AutomationLab_env.tool_use_AutomationLab_env:AutomationLabEnv",
)
register(
    "difficulty:AutomationLabWideGap",
    "gem.envs.difficulty.tool_use_AutomationLab_env.tool_use_AutomationLab_env_widegap:AutomationLabWideGapEnv",
)
register(
    "difficulty:AutomationLabHugeGap",
    "gem.envs.difficulty.tool_use_AutomationLab_env.tool_use_AutomationLab_env_hugegap:AutomationLabHugeGapEnv",
)
register(
    "difficulty:ChoreCanvas",
    "gem.envs.difficulty.tool_use_ChoreCanvas_env.tool_use_ChoreCanvas_env:ChoreCanvasEnv",
)
register(
    "difficulty:ChoreCanvasWideGap",
    "gem.envs.difficulty.tool_use_ChoreCanvas_env.tool_use_ChoreCanvas_env_widegap:ChoreCanvasWideGapEnv",
)
register(
    "difficulty:ChoreCanvasHugeGap",
    "gem.envs.difficulty.tool_use_ChoreCanvas_env.tool_use_ChoreCanvas_env_hugegap:ChoreCanvasHugeGapEnv",
)
register(
    "difficulty:CoursePlanner",
    "gem.envs.difficulty.tool_use_CoursePlanner_env.tool_use_CoursePlanner_env:CoursePlannerEnv",
)
register(
    "difficulty:CoursePlannerWideGap",
    "gem.envs.difficulty.tool_use_CoursePlanner_env.tool_use_CoursePlanner_env_widegap:CoursePlannerWideGapEnv",
)
register(
    "difficulty:CoursePlannerHugeGap",
    "gem.envs.difficulty.tool_use_CoursePlanner_env.tool_use_CoursePlanner_env_hugegap:CoursePlannerHugeGapEnv",
)
register(
    "difficulty:EmergencyOps",
    "gem.envs.difficulty.tool_use_EmergencyOps_env.tool_use_EmergencyOps_env:EmergencyOpsEnv",
)
register(
    "difficulty:EmergencyOpsWideGap",
    "gem.envs.difficulty.tool_use_EmergencyOps_env.tool_use_EmergencyOps_env_widegap:EmergencyOpsWideGapEnv",
)
register(
    "difficulty:EmergencyOpsHugeGap",
    "gem.envs.difficulty.tool_use_EmergencyOps_env.tool_use_EmergencyOps_env_hugegap:EmergencyOpsHugeGapEnv",
)
register(
    "difficulty:EnergyGrid",
    "gem.envs.difficulty.tool_use_EnergyGrid_env.tool_use_EnergyGrid_env:EnergyGridEnv",
)
register(
    "difficulty:EnergyGridWideGap",
    "gem.envs.difficulty.tool_use_EnergyGrid_env.tool_use_EnergyGrid_env_widegap:EnergyGridWideGapEnv",
)
register(
    "difficulty:EnergyGridHugeGap",
    "gem.envs.difficulty.tool_use_EnergyGrid_env.tool_use_EnergyGrid_env_hugegap:EnergyGridHugeGapEnv",
)
register(
    "difficulty:EventPlanner",
    "gem.envs.difficulty.tool_use_EventPlanner_env.tool_use_EventPlanner_env:EventPlannerEnv",
)
register(
    "difficulty:EventPlannerWideGap",
    "gem.envs.difficulty.tool_use_EventPlanner_env.tool_use_EventPlanner_env_widegap:EventPlannerWideGapEnv",
)
register(
    "difficulty:EventPlannerHugeGap",
    "gem.envs.difficulty.tool_use_EventPlanner_env.tool_use_EventPlanner_env_hugegap:EventPlannerHugeGapEnv",
)
register(
    "difficulty:FinanceOps",
    "gem.envs.difficulty.tool_use_FinanceOps_env.tool_use_FinanceOps_env:FinanceOpsEnv",
)
register(
    "difficulty:FinanceOpsWideGap",
    "gem.envs.difficulty.tool_use_FinanceOps_env.tool_use_FinanceOps_env_widegap:FinanceOpsWideGapEnv",
)
register(
    "difficulty:FinanceOpsHugeGap",
    "gem.envs.difficulty.tool_use_FinanceOps_env.tool_use_FinanceOps_env_hugegap:FinanceOpsHugeGapEnv",
)
register(
    "difficulty:FoodDelivery",
    "gem.envs.difficulty.tool_use_FoodDelivery_env.tool_use_FoodDelivery_env:FoodDeliveryEnv",
)
register(
    "difficulty:FoodDeliveryWideGap",
    "gem.envs.difficulty.tool_use_FoodDelivery_env.tool_use_FoodDelivery_env_widegap:FoodDeliveryWideGapEnv",
)
register(
    "difficulty:FoodDeliveryHugeGap",
    "gem.envs.difficulty.tool_use_FoodDelivery_env.tool_use_FoodDelivery_env_hugegap:FoodDeliveryHugeGapEnv",
)
register(
    "difficulty:HospitalOps",
    "gem.envs.difficulty.tool_use_HospitalOps_env.tool_use_HospitalOps_env:HospitalOpsEnv",
)
register(
    "difficulty:HospitalOpsWideGap",
    "gem.envs.difficulty.tool_use_HospitalOps_env.tool_use_HospitalOps_env_widegap:HospitalOpsWideGapEnv",
)
register(
    "difficulty:HospitalOpsHugeGap",
    "gem.envs.difficulty.tool_use_HospitalOps_env.tool_use_HospitalOps_env_hugegap:HospitalOpsHugeGapEnv",
)
register(
    "difficulty:RenoPlanner",
    "gem.envs.difficulty.tool_use_RenoPlanner_env.tool_use_RenoPlanner_env:RenoPlannerEnv",
)
register(
    "difficulty:RenoPlannerWideGap",
    "gem.envs.difficulty.tool_use_RenoPlanner_env.tool_use_RenoPlanner_env_widegap:RenoPlannerWideGapEnv",
)
register(
    "difficulty:RenoPlannerHugeGap",
    "gem.envs.difficulty.tool_use_RenoPlanner_env.tool_use_RenoPlanner_env_hugegap:RenoPlannerHugeGapEnv",
)
register(
    "difficulty:ShoppingMall",
    "gem.envs.difficulty.tool_use_ShoppingMall_env.tool_use_ShoppingMall_env:ShoppingMallEnv",
)
register(
    "difficulty:ShoppingMallWideGap",
    "gem.envs.difficulty.tool_use_ShoppingMall_env.tool_use_ShoppingMall_env_widegap:ShoppingMallWideGapEnv",
)
register(
    "difficulty:ShoppingMallHugeGap",
    "gem.envs.difficulty.tool_use_ShoppingMall_env.tool_use_ShoppingMall_env_hugegap:ShoppingMallHugeGapEnv",
)
register(
    "difficulty:SupplyChain",
    "gem.envs.difficulty.tool_use_SupplyChain_env.tool_use_SupplyChain_env:SupplyChainEnv",
)
register(
    "difficulty:SupplyChainWideGap",
    "gem.envs.difficulty.tool_use_SupplyChain_env.tool_use_SupplyChain_env_widegap:SupplyChainWideGapEnv",
)
register(
    "difficulty:SupplyChainHugeGap",
    "gem.envs.difficulty.tool_use_SupplyChain_env.tool_use_SupplyChain_env_hugegap:SupplyChainHugeGapEnv",
)
register(
    "difficulty:TerminalOps",
    "gem.envs.difficulty.tool_use_TerminalOps_env.tool_use_TerminalOps_env:TerminalOpsEnv",
)
register(
    "difficulty:TerminalOpsWideGap",
    "gem.envs.difficulty.tool_use_TerminalOps_env.tool_use_TerminalOps_env_widegap:TerminalOpsWideGapEnv",
)
register(
    "difficulty:TerminalOpsHugeGap",
    "gem.envs.difficulty.tool_use_TerminalOps_env.tool_use_TerminalOps_env_hugegap:TerminalOpsHugeGapEnv",
)
register(
    "difficulty:TransitOps",
    "gem.envs.difficulty.tool_use_TransitOps_env.tool_use_TransitOps_env:TransitOpsEnv",
)
register(
    "difficulty:TransitOpsWideGap",
    "gem.envs.difficulty.tool_use_TransitOps_env.tool_use_TransitOps_env_widegap:TransitOpsWideGapEnv",
)
register(
    "difficulty:TransitOpsHugeGap",
    "gem.envs.difficulty.tool_use_TransitOps_env.tool_use_TransitOps_env_hugegap:TransitOpsHugeGapEnv",
)
register(
    "difficulty:TravelDesk",
    "gem.envs.difficulty.tool_use_TravelDesk_env.tool_use_TravelDesk_env:TravelDeskEnv",
)
register(
    "difficulty:TravelDeskWideGap",
    "gem.envs.difficulty.tool_use_TravelDesk_env.tool_use_TravelDesk_env_widegap:TravelDeskWideGapEnv",
)
register(
    "difficulty:TravelDeskHugeGap",
    "gem.envs.difficulty.tool_use_TravelDesk_env.tool_use_TravelDesk_env_hugegap:TravelDeskHugeGapEnv",
)
register(
    "difficulty:WebShop",
    "gem.envs.difficulty.tool_use_WebShop_env.tool_use_WebShop_env:WebShopEnv",
)
register(
    "difficulty:WebShopWideGap",
    "gem.envs.difficulty.tool_use_WebShop_env.tool_use_WebShop_env_widegap:WebShopWideGapEnv",
)
register(
    "difficulty:WebShopHugeGap",
    "gem.envs.difficulty.tool_use_WebShop_env.tool_use_WebShop_env_hugegap:WebShopHugeGapEnv",
)

# 0116 add eval—-env:

register(
    "example:ApplianceWard",
    "gem.envs.example.tool_use_ApplianceWard_env:ApplianceWardEnv",
)

register(
    "example:CampusWeave",
    "gem.envs.example.tool_use_CampusWeave_env:CampusWeaveEnv",
)

register(
    "example:ErrandLedger",
    "gem.envs.example.tool_use_ErrandLedger_env:ErrandLedgerEnv",
)

register(
    "example:EventHarbor",
    "gem.envs.example.tool_use_EventHarbor_env:EventHarborEnv",
)

register(
    "example:LabRunStride",
    "gem.envs.example.tool_use_LabRunStride_env:LabRunStrideEnv",
)

register(
    "example:LogAtlasStride",
    "gem.envs.example.tool_use_LogAtlasStride_env:LogAtlasStrideEnv",
)

register(
    "example:MarketMosaic",
    "gem.envs.example.tool_use_MarketMosaic_env:MarketMosaicEnv",
)

register(
    "example:MediCircuit",
    "gem.envs.example.tool_use_MediCircuit_env:MediCircuitEnv",
)

register(
    "example:PortfolioForgeStride",
    "gem.envs.example.tool_use_PortfolioForgeStride_env:PortfolioForgeStrideEnv",
)

register(
    "example:ShoppingScout",
    "gem.envs.example.tool_use_ShoppingScout_env:ShoppingScoutEnv",
)

register(
    "example:SupplyWeaveStride",
    "gem.envs.example.tool_use_SupplyWeaveStride_env:SupplyWeaveStrideEnv",
)

register(
    "example:WebScout",
    "gem.envs.example.tool_use_WebScout_env:WebScoutEnv",
)

register(
    "example:WebShopPulse",
    "gem.envs.example.tool_use_WebShopPulse_env:WebShopPulseEnv",
)

register(
    "example:WorkbenchStrideOrchestrator",
    "gem.envs.example.tool_use_WorkbenchStrideOrchestrator_env:WorkbenchStrideOrchestratorEnv",
)

# 0116 add logic 4 eval:

register(
    "example:AxiomDelver",
    "gem.envs.example.logic_AxiomDelver_ArraySearchEnv_GEM_env:AxiomDelverEnv",
)

register(
    "example:ClauseWarden",
    "gem.envs.example.logic_ClauseWarden_IncreasingPathCountEnv_GEM_env:ClauseWardenEnv",
)

register(
    "example:ClauseWitnessQuest",
    "gem.envs.example.logic_ClauseWitnessQuest_AccommodationCheckEnv_GEM_env:ClauseWitnessQuestEnv",
)

register(
    "example:EntailmentSeeker",
    "gem.envs.example.logic_EntailmentSeeker_SuperPalindromicEnv_GEM_env:EntailmentSeekerEnv",
)

register(
    "example:EntailmentSleuth",
    "gem.envs.example.logic_EntailmentSleuth_HappyGroupsCountEnv_GEM_env:EntailmentSleuthEnv",
)

register(
    "example:HornChisel",
    "gem.envs.example.logic_HornChisel_ProblemCountingEnv_GEM_env:HornChiselEnv",
)

register(
    "example:HornTrailEntailer",
    "gem.envs.example.logic_HornTrailEntailer_ElementCountEnv_GEM_env:HornTrailEntailerEnv",
)

register(
    "example:LexMinModelFinder",
    "gem.envs.example.logic_LexMinModelFinder_LowHighPairsEnv_GEM_env:LexMinModelFinderEnv",
)

register(
    "example:LogicGlyphDeductor",
    "gem.envs.example.logic_LogicGlyphDeductor_MarathonTimeEnv_GEM_env:LogicGlyphDeductorEnv",
)

register(
    "example:ParityModelCrafter",
    "gem.envs.example.logic_ParityModelCrafter_diagram_coloring_env:ParityModelCrafterEnv",
)

register(
    "example:ProofCrafter",
    "gem.envs.example.logic_ProofCrafter_tower_of_hanoi_env:ProofCrafterEnv",
)

register(
    "example:QuantifierWitnessForge",
    "gem.envs.example.logic_QuantifierWitnessForge_date_count_env:QuantifierWitnessForgeEnv",
)

register(
    "example:SequentSentinel",
    "gem.envs.example.logic_SequentSentinel_AnagramCheckerEnv_GEM_env:SequentSentinelEnv",
)

register(
    "example:TableauVoyager",
    "gem.envs.example.logic_TableauVoyager_BrokenKeyboardEnv_GEM_env:TableauVoyagerEnv",
)

register(
    "example:TruthSmith",
    "gem.envs.example.logic_TruthSmith_emoji_connect_env:TruthSmithEnv",
)


# 0119 add codegym 4 eval 
register(
    "codegym:AnagramTransformationEnv",
    "gem.envs.CodeGym.AnagramTransformationEnv_GEM:AnagramTransformationEnvGEM",
)
register(
    "codegym:ArithmeticSequenceCheckEnv",
    "gem.envs.CodeGym.ArithmeticSequenceCheckEnv_GEM:ArithmeticSequenceCheckEnvGEM",
)
register(
    "codegym:BalloonBurstingEnv",
    "gem.envs.CodeGym.BalloonBurstingEnv_GEM:BalloonBurstingEnvGEM",
)
register(
    "codegym:BipartiteCheckEnv",
    "gem.envs.CodeGym.BipartiteCheckEnv_GEM:BipartiteCheckEnvGEM",
)
register(
    "codegym:CargoDeliveryEnv",
    "gem.envs.CodeGym.CargoDeliveryEnv_GEM:CargoDeliveryEnvGEM",
)
register(
    "codegym:CircleOverlapEnv",
    "gem.envs.CodeGym.CircleOverlapEnv_GEM:CircleOverlapEnvGEM",
)
register(
    "codegym:CoordinateTransformationEnv",
    "gem.envs.CodeGym.CoordinateTransformationEnv_GEM:CoordinateTransformationEnvGEM",
)
register(
    "codegym:DiagonalCountingEnv",
    "gem.envs.CodeGym.DiagonalCountingEnv_GEM:DiagonalCountingEnvGEM",
)
register(
    "codegym:DistinctElementsCountEnv",
    "gem.envs.CodeGym.DistinctElementsCountEnv_GEM:DistinctElementsCountEnvGEM",
)
register(
    "codegym:EnergyDifferenceMinimizingEnv",
    "gem.envs.CodeGym.EnergyDifferenceMinimizingEnv_GEM:EnergyDifferenceMinimizingEnvGEM",
)
register(
    "codegym:GoldCollectionEnv",
    "gem.envs.CodeGym.GoldCollectionEnv_GEM:GoldCollectionEnvGEM",
)
register(
    "codegym:GridPathCountEnv",
    "gem.envs.CodeGym.GridPathCountEnv_GEM:GridPathCountEnvGEM",
)
register(
    "codegym:GridShortestPathEnv",
    "gem.envs.CodeGym.GridShortestPathEnv_GEM:GridShortestPathEnvGEM",
)
register(
    "codegym:GridSumEnv",
    "gem.envs.CodeGym.GridSumEnv_GEM:GridSumEnvGEM",
)
register(
    "codegym:HamiltonianCycleEnv",
    "gem.envs.CodeGym.HamiltonianCycleEnv_GEM:HamiltonianCycleEnvGEM",
)
register(
    "codegym:HeapSortEnv",
    "gem.envs.CodeGym.HeapSortEnv_GEM:HeapSortEnvGEM",
)
register(
    "codegym:HistogramMaxAreaEnv",
    "gem.envs.CodeGym.HistogramMaxAreaEnv_GEM:HistogramMaxAreaEnvGEM",
)
register(
    "codegym:HouseRobberEnv",
    "gem.envs.CodeGym.HouseRobberEnv_GEM:HouseRobberEnvGEM",
)
register(
    "codegym:KnapsackEnv",
    "gem.envs.CodeGym.KnapsackEnv_GEM:KnapsackEnvGEM",
)
register(
    "codegym:LargestEmptySquareEnv",
    "gem.envs.CodeGym.LargestEmptySquareEnv_GEM:LargestEmptySquareEnvGEM",
)
register(
    "codegym:LargestHarmonicSubsetEnv",
    "gem.envs.CodeGym.LargestHarmonicSubsetEnv_GEM:LargestHarmonicSubsetEnvGEM",
)
register(
    "codegym:LargestRectangleEnv",
    "gem.envs.CodeGym.LargestRectangleEnv_GEM:LargestRectangleEnvGEM",
)
register(
    "codegym:LargestSquareEnv",
    "gem.envs.CodeGym.LargestSquareEnv_GEM:LargestSquareEnvGEM",
)
register(
    "codegym:LongestCommonSubsequenceEnv",
    "gem.envs.CodeGym.LongestCommonSubsequenceEnv_GEM:LongestCommonSubsequenceEnvGEM",
)
register(
    "codegym:LongestConsecutiveOnesEnv",
    "gem.envs.CodeGym.LongestConsecutiveOnesEnv_GEM:LongestConsecutiveOnesEnvGEM",
)
register(
    "codegym:LongestConsecutiveSubsequenceEnv",
    "gem.envs.CodeGym.LongestConsecutiveSubsequenceEnv_GEM:LongestConsecutiveSubsequenceEnvGEM",
)
register(
    "codegym:LongestFibSubseqEnv",
    "gem.envs.CodeGym.LongestFibSubseqEnv_GEM:LongestFibSubseqEnvGEM",
)
register(
    "codegym:LongestIncreasingSubarrayEnv",
    "gem.envs.CodeGym.LongestIncreasingSubarrayEnv_GEM:LongestIncreasingSubarrayEnvGEM",
)
register(
    "codegym:LongestSubsequenceWordEnv",
    "gem.envs.CodeGym.LongestSubsequenceWordEnv_GEM:LongestSubsequenceWordEnvGEM",
)
register(
    "codegym:LongestTwoColorSubarrayEnv",
    "gem.envs.CodeGym.LongestTwoColorSubarrayEnv_GEM:LongestTwoColorSubarrayEnvGEM",
)
register(
    "codegym:MajorityElementEnv",
    "gem.envs.CodeGym.MajorityElementEnv_GEM:MajorityElementEnvGEM",
)
register(
    "codegym:MarathonStationsEnv",
    "gem.envs.CodeGym.MarathonStationsEnv_GEM:MarathonStationsEnvGEM",
)
register(
    "codegym:MatrixCreationEnv",
    "gem.envs.CodeGym.MatrixCreationEnv_GEM:MatrixCreationEnvGEM",
)
register(
    "codegym:MaxApplesEnv",
    "gem.envs.CodeGym.MaxApplesEnv_GEM:MaxApplesEnvGEM",
)
register(
    "codegym:MaxCutTreesEnv",
    "gem.envs.CodeGym.MaxCutTreesEnv_GEM:MaxCutTreesEnvGEM",
)
register(
    "codegym:MaxFlowersEnv",
    "gem.envs.CodeGym.MaxFlowersEnv_GEM:MaxFlowersEnvGEM",
)
register(
    "codegym:MaxGoldCoinsEnv",
    "gem.envs.CodeGym.MaxGoldCoinsEnv_GEM:MaxGoldCoinsEnvGEM",
)
register(
    "codegym:MaxIncreasingSubarraySumEnv",
    "gem.envs.CodeGym.MaxIncreasingSubarraySumEnv_GEM:MaxIncreasingSubarraySumEnvGEM",
)
register(
    "codegym:MaxNonBlockingTowersEnv",
    "gem.envs.CodeGym.MaxNonBlockingTowersEnv_GEM:MaxNonBlockingTowersEnvGEM",
)
register(
    "codegym:MaxNonOverlappingProjectsEnv",
    "gem.envs.CodeGym.MaxNonOverlappingProjectsEnv_GEM:MaxNonOverlappingProjectsEnvGEM",
)
register(
    "codegym:MaxWaterContainerEnv",
    "gem.envs.CodeGym.MaxWaterContainerEnv_GEM:MaxWaterContainerEnvGEM",
)
register(
    "codegym:MaximumSpanningTreeEnv",
    "gem.envs.CodeGym.MaximumSpanningTreeEnv_GEM:MaximumSpanningTreeEnvGEM",
)
register(
    "codegym:MaximumSumSubgridEnv",
    "gem.envs.CodeGym.MaximumSumSubgridEnv_GEM:MaximumSumSubgridEnvGEM",
)
register(
    "codegym:MinContiguousSubarrayEnv",
    "gem.envs.CodeGym.MinContiguousSubarrayEnv_GEM:MinContiguousSubarrayEnvGEM",
)
register(
    "codegym:MinEnergyCombiningEnv",
    "gem.envs.CodeGym.MinEnergyCombiningEnv_GEM:MinEnergyCombiningEnvGEM",
)
register(
    "codegym:MinProductSegmentationEnv",
    "gem.envs.CodeGym.MinProductSegmentationEnv_GEM:MinProductSegmentationEnvGEM",
)
register(
    "codegym:MinSubarrayLenEnv",
    "gem.envs.CodeGym.MinSubarrayLenEnv_GEM:MinSubarrayLenEnvGEM",
)
register(
    "codegym:MinSubsetSumDiffEnv",
    "gem.envs.CodeGym.MinSubsetSumDiffEnv_GEM:MinSubsetSumDiffEnvGEM",
)
register(
    "codegym:MinSwapsToSortEnv",
    "gem.envs.CodeGym.MinSwapsToSortEnv_GEM:MinSwapsToSortEnvGEM",
)
register(
    "codegym:MinimizeMaxSubarraySumEnv",
    "gem.envs.CodeGym.MinimizeMaxSubarraySumEnv_GEM:MinimizeMaxSubarraySumEnvGEM",
)
register(
    "codegym:MinimizeMaxTimeEnv",
    "gem.envs.CodeGym.MinimizeMaxTimeEnv_GEM:MinimizeMaxTimeEnvGEM",
)
register(
    "codegym:MinimumPossibleSumEnv",
    "gem.envs.CodeGym.MinimumPossibleSumEnv_GEM:MinimumPossibleSumEnvGEM",
)
register(
    "codegym:MissingRangesEnv",
    "gem.envs.CodeGym.MissingRangesEnv_GEM:MissingRangesEnvGEM",
)
register(
    "codegym:MostFrequentBirdEnv",
    "gem.envs.CodeGym.MostFrequentBirdEnv_GEM:MostFrequentBirdEnvGEM",
)
register(
    "codegym:NextPalindromeEnv",
    "gem.envs.CodeGym.NextPalindromeEnv_GEM:NextPalindromeEnvGEM",
)
register(
    "codegym:OddOccurrenceFinderEnv",
    "gem.envs.CodeGym.OddOccurrenceFinderEnv_GEM:OddOccurrenceFinderEnvGEM",
)
register(
    "codegym:PalindromeVerificationEnv",
    "gem.envs.CodeGym.PalindromeVerificationEnv_GEM:PalindromeVerificationEnvGEM",
)
register(
    "codegym:ParitySortingEnv",
    "gem.envs.CodeGym.ParitySortingEnv_GEM:ParitySortingEnvGEM",
)
register(
    "codegym:PathCountingEnv",
    "gem.envs.CodeGym.PathCountingEnv_GEM:PathCountingEnvGEM",
)
register(
    "codegym:PerfectSquareSequenceEnv",
    "gem.envs.CodeGym.PerfectSquareSequenceEnv_GEM:PerfectSquareSequenceEnvGEM",
)
register(
    "codegym:PrefixPalindromeEnv",
    "gem.envs.CodeGym.PrefixPalindromeEnv_GEM:PrefixPalindromeEnvGEM",
)
register(
    "codegym:PrimeFilteringEnv",
    "gem.envs.CodeGym.PrimeFilteringEnv_GEM:PrimeFilteringEnvGEM",
)
register(
    "codegym:ProblemCountingEnv",
    "gem.envs.CodeGym.ProblemCountingEnv_GEM:ProblemCountingEnvGEM",
)
register(
    "codegym:RainWaterTrapEnv",
    "gem.envs.CodeGym.RainWaterTrapEnv_GEM:RainWaterTrapEnvGEM",
)
register(
    "codegym:RainwaterCollectionEnv",
    "gem.envs.CodeGym.RainwaterCollectionEnv_GEM:RainwaterCollectionEnvGEM",
)
register(
    "codegym:RainwaterTrapEnv",
    "gem.envs.CodeGym.RainwaterTrapEnv_GEM:RainwaterTrapEnvGEM",
)
register(
    "codegym:RamanujanNumberEnv",
    "gem.envs.CodeGym.RamanujanNumberEnv_GEM:RamanujanNumberEnvGEM",
)
register(
    "codegym:RemoveDuplicatesEnv",
    "gem.envs.CodeGym.RemoveDuplicatesEnv_GEM:RemoveDuplicatesEnvGEM",
)
register(
    "codegym:ResourceCombiningEnv",
    "gem.envs.CodeGym.ResourceCombiningEnv_GEM:ResourceCombiningEnvGEM",
)
register(
    "codegym:RotatedArrayMinEnv",
    "gem.envs.CodeGym.RotatedArrayMinEnv_GEM:RotatedArrayMinEnvGEM",
)
register(
    "codegym:SharedProblemPairsEnv",
    "gem.envs.CodeGym.SharedProblemPairsEnv_GEM:SharedProblemPairsEnvGEM",
)
register(
    "codegym:ShortestPathEnv",
    "gem.envs.CodeGym.ShortestPathEnv_GEM:ShortestPathEnvGEM",
)
register(
    "codegym:SmallestRangeEnv",
    "gem.envs.CodeGym.SmallestRangeEnv_GEM:SmallestRangeEnvGEM",
)
register(
    "codegym:SmallestRectangleEnv",
    "gem.envs.CodeGym.SmallestRectangleEnv_GEM:SmallestRectangleEnvGEM",
)
register(
    "codegym:SmallestSubarrayEnv",
    "gem.envs.CodeGym.SmallestSubarrayEnv_GEM:SmallestSubarrayEnvGEM",
)
register(
    "codegym:StringReorderEnv",
    "gem.envs.CodeGym.StringReorderEnv_GEM:StringReorderEnvGEM",
)
register(
    "codegym:StringSwapEnv",
    "gem.envs.CodeGym.StringSwapEnv_GEM:StringSwapEnvGEM",
)
register(
    "codegym:SubgridBeautyEnv",
    "gem.envs.CodeGym.SubgridBeautyEnv_GEM:SubgridBeautyEnvGEM",
)
register(
    "codegym:SudokuValidationEnv",
    "gem.envs.CodeGym.SudokuValidationEnv_GEM:SudokuValidationEnvGEM",
)
register(
    "codegym:SumArrayConstructionEnv",
    "gem.envs.CodeGym.SumArrayConstructionEnv_GEM:SumArrayConstructionEnvGEM",
)
register(
    "codegym:SymmetricGridEnv",
    "gem.envs.CodeGym.SymmetricGridEnv_GEM:SymmetricGridEnvGEM",
)
register(
    "codegym:TaskManagerEnv",
    "gem.envs.CodeGym.TaskManagerEnv_GEM:TaskManagerEnvGEM",
)
register(
    "codegym:TeamScoreBalancingEnv",
    "gem.envs.CodeGym.TeamScoreBalancingEnv_GEM:TeamScoreBalancingEnvGEM",
)
register(
    "codegym:TicketPriorityEnv",
    "gem.envs.CodeGym.TicketPriorityEnv_GEM:TicketPriorityEnvGEM",
)
register(
    "codegym:ToeplitzMatrixEnv",
    "gem.envs.CodeGym.ToeplitzMatrixEnv_GEM:ToeplitzMatrixEnvGEM",
)
register(
    "codegym:TreasureHuntExpectationEnv",
    "gem.envs.CodeGym.TreasureHuntExpectationEnv_GEM:TreasureHuntExpectationEnvGEM",
)
register(
    "codegym:TreeCheckEnv",
    "gem.envs.CodeGym.TreeCheckEnv_GEM:TreeCheckEnvGEM",
)
register(
    "codegym:TriangularTripletEnv",
    "gem.envs.CodeGym.TriangularTripletEnv_GEM:TriangularTripletEnvGEM",
)
register(
    "codegym:TwoSumEnv",
    "gem.envs.CodeGym.TwoSumEnv_GEM:TwoSumEnvGEM",
)
register(
    "codegym:UniquePathsEnv",
    "gem.envs.CodeGym.UniquePathsEnv_GEM:UniquePathsEnvGEM",
)
register(
    "codegym:UniqueSubstringCounterEnv",
    "gem.envs.CodeGym.UniqueSubstringCounterEnv_GEM:UniqueSubstringCounterEnvGEM",
)
register(
    "codegym:UniqueSubstringsWithOddEnv",
    "gem.envs.CodeGym.UniqueSubstringsWithOddEnv_GEM:UniqueSubstringsWithOddEnvGEM",
)

# 0119 add errorcase：
register(
    "example:AlgorithmCostERROR",
    "gem.envs.example.algorithm_AlgorithmCostERROR_MaxLapsEnv_GEM_env:AlgorithmCostERROREnv",
)

register(
    "example:AlgorithmSATDecisionERROR",
    "gem.envs.example.algorithm_AlgorithmSATDecisionERROR_WorkshopRoomAssignmentEnv_GEM_env:AlgorithmSATDecisionERROREnv",
)

register(
    "example:RuneRingPuzzleERROR",
    "gem.envs.example.game_RuneRingPuzzleERROR_black_white_copy_env:RuneRingPuzzleERROREnv",
)

register(
    "example:DataTinkerBenchERROR",
    "gem.envs.example.tool_use_DataTinkerBenchERROR_MaxContiguousLightAreaEnv_GEM_env:DataTinkerBenchERROREnv",
)

# 0120 add gem_game 4 eval
register(
    "example:GuessTheNumber-v0",
    "gem.envs.example.guess_the_number:GuessTheNumberEnv",
)
register(
    "example:GuessTheNumber-v0-hard",
    "gem.envs.example.guess_the_number:GuessTheNumberEnv",
    min_number=1,
    max_number=50,
    max_turns=7,
)
register(
    "example:GuessTheNumber-v0-easy",
    "gem.envs.example.guess_the_number:GuessTheNumberEnv",
    min_number=1,
    max_number=10,
    max_turns=4,
)
register(
    "example:GuessTheNumber-v0-random",
    "gem.envs.example.guess_the_number:GuessTheNumberEnv",
    min_number=None,
    max_number=None,
    max_turns=None,
)

register(
    "example:Mastermind-v0",
    "gem.envs.example.mastermind:MastermindEnv",
)
register(
    "example:Mastermind-v0-hard",
    "gem.envs.example.mastermind:MastermindEnv",
    code_length=4,
    num_numbers=8,
    max_turns=30,
    duplicate_numbers=False,
)
register(
    "example:Mastermind-v0-random",
    "gem.envs.example.mastermind:MastermindEnv",
    code_length=None,
    num_numbers=None,
    max_turns=None,
    duplicate_numbers=False,
)
register(
    "example:Mastermind-v0-easy",
    "gem.envs.example.mastermind:MastermindEnv",
    code_length=2,
    num_numbers=6,
    max_turns=10,
    duplicate_numbers=False,
)

register(
    "example:Minesweeper-v0",
    "gem.envs.example.minesweeper:MinesweeperEnv",
)
register(
    "example:Minesweeper-v0-easy",
    "gem.envs.example.minesweeper:MinesweeperEnv",
    rows=5,
    cols=5,
    num_mines=5,
    max_turns=25,
)
register(
    "example:Minesweeper-v0-hard",
    "gem.envs.example.minesweeper:MinesweeperEnv",
    rows=8,
    cols=8,
    num_mines=12,
    max_turns=64,
)
register(
    "example:Minesweeper-v0-random",
    "gem.envs.example.minesweeper:MinesweeperEnv",
    rows=None,
    cols=None,
    num_mines=None,
    max_turns=None,
)

register(
    "example:Wordle-v0",
    "gem.envs.example.wordle:WordleEnv",
)
register(
    "example:Wordle-v0-hard",
    "gem.envs.example.wordle:WordleEnv",
    word_length=5,
    only_real_words=True,
    max_turns=25,
)
register(
    "example:Wordle-v0-easy",
    "gem.envs.example.wordle:WordleEnv",
    word_length=3,
    only_real_words=True,
    max_turns=15,
)
register(
    "example:Wordle-v0-random",
    "gem.envs.example.wordle:WordleEnv",
    word_length=None,
    only_real_words=True,
    max_turns=None,
)

register(
    "example:FifteenPuzzle-v0",
    "gem.envs.example.fifteen_puzzle:FifteenPuzzleEnv",
)
register(
    "example:FifteenPuzzle-v0-random",
    "gem.envs.example.fifteen_puzzle:FifteenPuzzleEnv",
    num_rows=None,
    max_turns=None,
)
register(
    "example:FifteenPuzzle-v0-easy",
    "gem.envs.example.fifteen_puzzle:FifteenPuzzleEnv",
    num_rows=2,
    max_turns=10,
)
register(
    "example:FifteenPuzzle-v0-hard",
    "gem.envs.example.fifteen_puzzle:FifteenPuzzleEnv",
    num_rows=4,
    max_turns=50,
)

register(
    "example:Hangman-v0",
    "gem.envs.example.hangman:HangmanEnv",
)
register(
    "example:Hangman-v0-random",
    "gem.envs.example.hangman:HangmanEnv",
    word_length=None,
    hardcore=False,
    max_turns=None,
)
register(
    "example:Hangman-v0-easy",
    "gem.envs.example.hangman:HangmanEnv",
    word_length=3,
    hardcore=False,
    max_turns=10,
)
register(
    "example:Hangman-v0-hard",
    "gem.envs.example.hangman:HangmanEnv",
    word_length=7,
    hardcore=False,
    max_turns=20,
)

register(
    "example:Sudoku-v0",
    "gem.envs.example.sudoku:SudokuEnv",
)
register(
    "example:Sudoku-v0-easy",
    "gem.envs.example.sudoku:SudokuEnv",
    clues=10,
    max_turns=15,
    scale=4,
)
register(
    "example:Sudoku-v0-hard",
    "gem.envs.example.sudoku:SudokuEnv",
    clues=50,
    max_turns=50,
    scale=9,
)
register(
    "example:Sudoku-v0-random",
    "gem.envs.example.sudoku:SudokuEnv",
    clues=None,
    max_turns=None,
    scale=None,
)

register(
    "example:TowerofHanoi-v0",
    "gem.envs.example.tower_of_hanoi:TowerofHanoiEnv",
)
register(
    "example:TowerofHanoi-v0-easy",
    "gem.envs.example.tower_of_hanoi:TowerofHanoiEnv",
    num_disks=3,
    max_turns=10,
)
register(
    "example:TowerofHanoi-v0-hard",
    "gem.envs.example.tower_of_hanoi:TowerofHanoiEnv",
    num_disks=5,
    max_turns=35,
)
register(
    "example:TowerofHanoi-v0-random",
    "gem.envs.example.tower_of_hanoi:TowerofHanoiEnv",
    num_disks=None,
    max_turns=None,
)

register(
    "example:Game2048-v0",
    "gem.envs.example.game_2048:Game2048Env",
)
register(
    "example:Game2048-v0-easy",
    "gem.envs.example.game_2048:Game2048Env",
    target_tile=64,
    max_turns=50,
)
register(
    "example:Game2048-v0-hard",
    "gem.envs.example.game_2048:Game2048Env",
    target_tile=512,
    max_turns=50,
)
register(
    "example:Game2048-v0-extreme-hard",
    "gem.envs.example.game_2048:Game2048Env",
    target_tile=2048,
    max_turns=100,
)
register(
    "example:Game2048-v0-random",
    "gem.envs.example.game_2048:Game2048Env",
    target_tile=None,
    max_turns=None,
)

register(
    "example:WordSearch-v0",
    "gem.envs.example.word_search:WordSearchEnv",
)
register(
    "example:WordSearch-v0-easy",
    "gem.envs.example.word_search:WordSearchEnv",
    num_words=5,
    max_turns=20,
    hardcore=False,
)
register(
    "example:WordSearch-v0-hard",
    "gem.envs.example.word_search:WordSearchEnv",
    num_words=5,
    max_turns=20,
    hardcore=True,
)
register(
    "example:WordSearch-v0-random",
    "gem.envs.example.word_search:WordSearchEnv",
    num_words=None,
    max_turns=None,
    hardcore=None,
)

# 0121 add evolve TEST:
register(
    "example:CoursePlannerTEST",
    "gem.envs.example.tool_use_CoursePlannerTEST_env:CoursePlannerTESTEnv",
)
register(
    "example:EmergencyOpsTEST",
    "gem.envs.example.tool_use_EmergencyOpsTEST_env:EmergencyOpsTESTEnv",
)
register(
    "example:EnergyGridTEST",
    "gem.envs.example.tool_use_EnergyGridTEST_env:EnergyGridTESTEnv",
)
register(
    "example:HospitalOpsTEST",
    "gem.envs.example.tool_use_HospitalOpsTEST_env:HospitalOpsTESTEnv",
)
