#!/bin/bash
# Resolve a config name (without .yaml extension) to its examples/ subdirectory type.
#
# Usage: resolve_config_type.sh <config_name> [rlinf_root]
#
# Output: subdirectory name (e.g., "embodiment", "agent", "reasoning")
# Exit:   0 if found, 1 if not found
#
# The script searches all config files under examples/*/config/ recursively,
# so it handles direct placement (embodiment/config/), subfolder nesting
# (reasoning/config/math/), and subproject nesting (agent/searchr1/config/)
# without hardcoding any category names.

set -euo pipefail

CONFIG_NAME="${1:?Usage: resolve_config_type.sh <config_name> [rlinf_root]}"
RLINF_ROOT="${2:-${RLINF_ROOT:-$(dirname "$(dirname "$(readlink -f "$0")")")}}"

if [ ! -d "$RLINF_ROOT/examples" ]; then
    echo "ERROR: examples/ directory not found under $RLINF_ROOT" >&2
    exit 1
fi

# Find the config file anywhere under examples/*/config/
# -path "*/config/*" ensures we only match files inside a config/ directory
# (not arbitrary .yaml files elsewhere under examples/)
RESULT=$(find "$RLINF_ROOT/examples" -path "*/config/*" -name "${CONFIG_NAME}.yaml" -type f 2>/dev/null | head -1)

if [ -n "$RESULT" ]; then
    # Extract the type: strip the examples/ prefix, take the first path component
    # e.g. "RLinf/examples/embodiment/config/maniskill_ppo_openvla.yaml" → "embodiment"
    # e.g. "RLinf/examples/agent/searchr1/config/train_qwen2.5.yaml" → "agent"
    echo "$RESULT" | sed "s|$RLINF_ROOT/examples/||" | cut -d/ -f1
    exit 0
fi

echo "unknown" >&2
exit 1