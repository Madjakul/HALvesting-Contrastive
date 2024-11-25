#!/bin/sh

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..    # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                        # Do not modify

# ************************** Customizable Arguments ************************************

CONFIG_PATH=$PROJECT_ROOT/configs/preprocess.yml
METADATA_PATH=$DATA_ROOT/metadata.json

# --------------------------------------------------------------------------------------

# RESPONSES_DIR=$DATA_ROOT/responses/
#
# NUM_PROC=4
#

# **************************************************************************************

cmd=( python3 "$PROJECT_ROOT/preprocess.py" \
  --config_path "$CONFIG_PATH" \
  --metadata_path "$METADATA_PATH" )

if [[ -v RESPONSES_DIR ]]; then
  cmd+=( --responses_dir "$RESPONSES_DIR" )
fi

if [[ -v NUM_PROC ]]; then
  cmd+=( --num_proc "$NUM_PROC" )
fi

"${cmd[@]}"
