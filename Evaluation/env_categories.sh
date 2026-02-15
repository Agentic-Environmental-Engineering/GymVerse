#!/usr/bin/env bash
#
# Environment Categories Configuration
# Define environments for each category: tool-use, code, logic, game
# Total: 32 environments (8 per category)
#

# Tool-use environments (8 total)
TOOL_USE_ENVS=(
    "example:ArtisanToolroom"
    "example:OperatorOrchestra"
    "example:ToolchestMaestro"
    "example:ApplianceWard"
    "example:CampusWeave"
    "example:ErrandLedger"
    "example:ShoppingScout"
    "example:PortfolioForgeStride"
)

# Game environments (8 total)
GAME_ENVS=(
    "example:AlchemyCraft"
    "example:BeaconCorridorNavigator"
    "example:DetectiveDeductionGame"
    "example:DungeonPartyPlanning"
    "example:GuildRegionsGame"
    "example:LexemeCipher"
    "example:MysteryMansionGame"
    "example:RaidPartyPlanner"
)

# Code/Algorithm environments (8 total)
CODE_ENVS=(
    "example:CodeBugFix"
    "example:CodeCoveragePlanner"
    "example:CodeGraphRecursion"
    "example:AlgorithmDAG"
    "example:AlgorithmWorkbench"
    "example:BooleanCircuitSynthesis"
    "example:GraphCycleStreaming"
    "example:SortingStrategy"
)

# Logic environments (8 total)
LOGIC_ENVS=(
    "example:AxiomOracleDeduction"
    "example:InferenceOrbit"
    "example:SequentSmith"
    "example:HornTrailEntailer"
    "example:EntailmentSeeker"
    "example:ClauseWitnessQuest"
    "example:AxiomDelver"
    "example:ParityModelCrafter"
)

# Function to get environments for a category
get_envs_for_category() {
    local category="$1"
    case "$category" in
        tool-use)
            echo "${TOOL_USE_ENVS[@]}"
            ;;
        code)
            echo "${CODE_ENVS[@]}"
            ;;
        logic)
            echo "${LOGIC_ENVS[@]}"
            ;;
        game)
            echo "${GAME_ENVS[@]}"
            ;;
        *)
            echo "ERROR: Unknown category: $category" >&2
            return 1
            ;;
    esac
}

# Export function for use in other scripts
export -f get_envs_for_category
