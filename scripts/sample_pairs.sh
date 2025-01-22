#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/.. # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                     # Do not modify

# ************************** Customizable Arguments ************************************

CONFIG_PATH=$PROJECT_ROOT/configs/sample_pairs.yml

# --------------------------------------------------------------------------------------

# CACHE_DIR=~/.cache
# OUTPUT_DIR=$DATA_ROOT
#
# NUM_PROC=4
#

# **************************************************************************************

cmd=(python3 "$PROJECT_ROOT/sample_pairs.py"
    --config_path "$CONFIG_PATH")

if [[ -v CACHE_DIR ]]; then
    cmd+=(--cache_dir "$CACHE_DIR")
fi

if [[ -v OUTPUT_DIR ]]; then
    cmd+=(--output_dir "$OUTPUT_DIR")
fi

if [[ -v NUM_PROC ]]; then
    cmd+=(--num_proc "$NUM_PROC")
fi

"${cmd[@]}"
