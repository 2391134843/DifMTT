#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/../log"

declare -A CONFIG
declare -a FLAGS
declare -a EXTRA_ARGS

init_default_config() {
    CONFIG["dataset"]="mimic-iii"
    CONFIG["cuda"]="1"
    CONFIG["model_name"]="DifMTT"
    CONFIG["early_stop"]="55"
    CONFIG["dim"]="256"
    CONFIG["lr"]="2e-5"
    CONFIG["dp"]="0.7"
    CONFIG["target_ddi"]="0.12"
    CONFIG["coef"]="2.5"
    CONFIG["epochs"]="100"
    CONFIG["diff_steps"]="20"
    CONFIG["diff_weight"]="0.2"
    CONFIG["diff_guidance_ddi"]="0.5"
    
    FLAGS=("--test_after_train" "--sbcl" "--diffusion")
    EXTRA_ARGS=("-n" "multi_intent_K16")
}

validate_environment() {
    local python_exec=""
    local required_commands=("python3" "nvidia-smi")
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "${cmd}" &> /dev/null; then
            echo "[WARNING] Command not found: ${cmd}" >&2
        fi
    done
    
    python_exec="$(command -v python3 2>/dev/null || command -v python 2>/dev/null)"
    if [[ -z "${python_exec}" ]]; then
        echo "[ERROR] Python interpreter not found" >&2
        exit 1
    fi
    
    echo "${python_exec}"
}

check_gpu_availability() {
    local cuda_device="${1:-0}"
    
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count
        gpu_count="$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)"
        
        if [[ "${cuda_device}" -ge "${gpu_count}" ]]; then
            echo "[WARNING] CUDA device ${cuda_device} may not be available (found ${gpu_count} GPUs)" >&2
        fi
    fi
}

build_command_args() {
    local args=()
    
    for key in "${!CONFIG[@]}"; do
        args+=("--${key}" "${CONFIG[${key}]}")
    done
    
    for flag in "${FLAGS[@]}"; do
        args+=("${flag}")
    done
    
    for extra in "${EXTRA_ARGS[@]}"; do
        args+=("${extra}")
    done
    
    echo "${args[@]}"
}

print_config_summary() {
    echo "================================================================================"
    echo "                         DifMTT Training Configuration"
    echo "================================================================================"
    echo "  Timestamp      : ${TIMESTAMP}"
    echo "  Working Dir    : ${SCRIPT_DIR}"
    echo "--------------------------------------------------------------------------------"
    echo "  Dataset        : ${CONFIG["dataset"]}"
    echo "  CUDA Device    : ${CONFIG["cuda"]}"
    echo "  Model          : ${CONFIG["model_name"]}"
    echo "  Dimension      : ${CONFIG["dim"]}"
    echo "  Learning Rate  : ${CONFIG["lr"]}"
    echo "  Dropout        : ${CONFIG["dp"]}"
    echo "  Epochs         : ${CONFIG["epochs"]}"
    echo "  Early Stop     : ${CONFIG["early_stop"]}"
    echo "--------------------------------------------------------------------------------"
    echo "  Target DDI     : ${CONFIG["target_ddi"]}"
    echo "  DDI Coef       : ${CONFIG["coef"]}"
    echo "  Diff Steps     : ${CONFIG["diff_steps"]}"
    echo "  Diff Weight    : ${CONFIG["diff_weight"]}"
    echo "  Diff Guidance  : ${CONFIG["diff_guidance_ddi"]}"
    echo "--------------------------------------------------------------------------------"
    echo "  Flags          : ${FLAGS[*]}"
    echo "  Extra Args     : ${EXTRA_ARGS[*]}"
    echo "================================================================================"
}

run_training() {
    local python_exec="${1}"
    local main_script="main_${CONFIG["model_name"]}.py"
    local cmd_args
    
    cmd_args="$(build_command_args)"
    
    if [[ ! -f "${SCRIPT_DIR}/${main_script}" ]]; then
        echo "[ERROR] Main script not found: ${main_script}" >&2
        exit 1
    fi
    
    print_config_summary
    
    echo ""
    echo "[INFO] Starting training at $(date)"
    echo "[INFO] Command: ${python_exec} ${main_script} ${cmd_args}"
    echo ""
    
    cd "${SCRIPT_DIR}"
    exec "${python_exec}" "${main_script}" ${cmd_args}
}

main() {
    init_default_config
    
    local python_exec
    python_exec="$(validate_environment)"
    
    check_gpu_availability "${CONFIG["cuda"]}"
    
    run_training "${python_exec}"
}

main "$@"
