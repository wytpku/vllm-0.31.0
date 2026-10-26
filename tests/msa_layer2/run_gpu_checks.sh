#!/usr/bin/env bash
# Run the source-bound metadata gate with an existing Python environment.
set -euo pipefail

MSA_GATE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MSA_GATE_PYTHON="/jizhicfs/pkuhetu/sca/wyt/msa/msa-v0310/.venv/bin/python"
MSA_GATE_ARGS=()
while (( $# )); do
    case "$1" in
        --python)
            if (( $# < 2 )); then
                echo "--python requires an existing interpreter path" >&2
                exit 2
            fi
            MSA_GATE_PYTHON="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash $0 [--python EXISTING_PYTHON] --output NEW_PERSISTENT_DIR --gpu PHYSICAL_INDEX [--page-sizes 16,32,64]"
            echo "Runs inspect, isolated CPU, normal package/native, imports, kernels, and plan. Stops at the first failure."
            exit 0
            ;;
        *) MSA_GATE_ARGS+=("$1"); shift ;;
    esac
done
if [[ ! -x "$MSA_GATE_PYTHON" ]]; then
    echo "Select an existing executable with --python" >&2
    exit 2
fi
exec "$MSA_GATE_PYTHON" -B "$MSA_GATE_DIR/run_metadata_gate.py" \
    --stage all "${MSA_GATE_ARGS[@]}"
