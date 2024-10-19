#!/bin/sh

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..    # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                        # Do not modify

# ************************** Customizable Arguments ************************************

CONFIG_PATH=$PROJECT_ROOT/configs/preprocess.yml
METADATA_PATH=$DATA_ROOT/metadata.json

# --------------------------------------------------------------------------------------

# DO_CONVERT_RESPONSES=true
# RESPONSES_DIR=$DATA_ROOT/responses/
#
# NUM_PROC=4
#
PUSH_TO_HUB=true

# **************************************************************************************

cmd=( python3 "$PROJECT_ROOT/preprocess.py" \
  --config_path "$CONFIG_PATH" \
  --metadata_path "$METADATA_PATH" )

if [[ -v DO_CONVERT_RESPONSES ]]; then
  cmd+=( --do_convert_responses \
    --responses_dir "$RESPONSES_DIR" )
fi

if [[ -v NUM_PROC ]]; then
  cmd+=( --num_proc "$NUM_PROC" )
fi

if [[ -v PUSH_TO_HUB ]]; then
  cmd+=( --push_to_hub )
fi

"${cmd[@]}"
