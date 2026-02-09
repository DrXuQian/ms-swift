#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MS_SWIFT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd -- "${MS_SWIFT_ROOT}/.." && pwd)"

CONDA_ENV="modelopt"
MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_TYPE="qwen2"
QUANT_CFG="INT8_DEFAULT_CFG"
TORCH_DTYPE="bfloat16"
MODEL_DEVICE_MAP="cuda:0"
KEEP_ARTIFACTS=0
OUTPUT_DIR=""

print_help() {
    cat <<'EOF'
Smoke test for ModelOpt checkpoint load+train in ms-swift.

Usage:
  scripts/smoke_modelopt_hf.sh [options]

Options:
  --conda-env <name>        Conda env name. Default: modelopt
  --model-id <id_or_path>   HF model id or local model path.
                            Default: Qwen/Qwen2.5-0.5B-Instruct
  --model-type <type>       ms-swift model_type for loading local checkpoint.
                            Default: qwen2
  --quant-cfg <cfg_name>    ModelOpt quant config name.
                            Default: INT8_DEFAULT_CFG
  --torch-dtype <dtype>     torch dtype for load (float16|bfloat16|float32).
                            Default: bfloat16
  --device-map <map>        Device map for ms-swift loading.
                            Default: cuda:0
  --output-dir <dir>        Save quantized checkpoint to this dir.
                            Default: auto temporary directory
  --keep-artifacts          Keep generated checkpoint directory.
  -h, --help                Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --conda-env)
        CONDA_ENV="${2:?missing value for --conda-env}"
        shift 2
        ;;
    --model-id)
        MODEL_ID="${2:?missing value for --model-id}"
        shift 2
        ;;
    --model-type)
        MODEL_TYPE="${2:?missing value for --model-type}"
        shift 2
        ;;
    --quant-cfg)
        QUANT_CFG="${2:?missing value for --quant-cfg}"
        shift 2
        ;;
    --torch-dtype)
        TORCH_DTYPE="${2:?missing value for --torch-dtype}"
        shift 2
        ;;
    --device-map)
        MODEL_DEVICE_MAP="${2:?missing value for --device-map}"
        shift 2
        ;;
    --output-dir)
        OUTPUT_DIR="${2:?missing value for --output-dir}"
        shift 2
        ;;
    --keep-artifacts)
        KEEP_ARTIFACTS=1
        shift
        ;;
    -h | --help)
        print_help
        exit 0
        ;;
    *)
        echo "Unknown argument: $1" >&2
        print_help
        exit 2
        ;;
    esac
done

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not found in PATH." >&2
    exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="$(mktemp -d -t modelopt_swift_smoke_XXXXXX)"
    CREATED_TMP=1
else
    mkdir -p "${OUTPUT_DIR}"
    CREATED_TMP=0
fi

cleanup() {
    if [[ "${KEEP_ARTIFACTS}" -eq 0 && "${CREATED_TMP}" -eq 1 ]]; then
        rm -rf "${OUTPUT_DIR}"
    fi
}
trap cleanup EXIT

echo "Running smoke test with settings:"
echo "  conda_env=${CONDA_ENV}"
echo "  model_id=${MODEL_ID}"
echo "  model_type=${MODEL_TYPE}"
echo "  quant_cfg=${QUANT_CFG}"
echo "  torch_dtype=${TORCH_DTYPE}"
echo "  device_map=${MODEL_DEVICE_MAP}"
echo "  output_dir=${OUTPUT_DIR}"

MS_SWIFT_ROOT="${MS_SWIFT_ROOT}" \
PROJECT_ROOT="${PROJECT_ROOT}" \
MODEL_ID="${MODEL_ID}" \
MODEL_TYPE="${MODEL_TYPE}" \
QUANT_CFG="${QUANT_CFG}" \
TORCH_DTYPE="${TORCH_DTYPE}" \
MODEL_DEVICE_MAP="${MODEL_DEVICE_MAP}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
conda run --no-capture-output -n "${CONDA_ENV}" python - <<'PY'
import os
import sys

import torch

ms_swift_root = os.environ["MS_SWIFT_ROOT"]
project_root = os.environ["PROJECT_ROOT"]

sys.path.insert(0, ms_swift_root)
if os.path.isdir(os.path.join(project_root, "modelopt")):
    sys.path.insert(0, project_root)

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from transformers import AutoModelForCausalLM, AutoTokenizer
from swift.model import get_model_processor


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


model_id = os.environ["MODEL_ID"]
model_type = os.environ["MODEL_TYPE"]
quant_cfg_name = os.environ["QUANT_CFG"]
torch_dtype_name = os.environ["TORCH_DTYPE"]
device_map = os.environ["MODEL_DEVICE_MAP"]
out_dir = os.environ["OUTPUT_DIR"]
torch_dtype = resolve_dtype(torch_dtype_name)

quant_cfg = getattr(mtq, quant_cfg_name, None)
if quant_cfg is None:
    raise ValueError(f"Unknown quant cfg: {quant_cfg_name}")

device = "cuda" if "cuda" in device_map else "cpu"

print(f"[smoke] load base model: {model_id}")
mto.enable_huggingface_checkpointing()
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    device_map=device,
    trust_remote_code=True,
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

sample = tokenizer("modelopt hf smoke calibration", return_tensors="pt").to(device)


def forward_loop(model):
    with torch.no_grad():
        model(**sample)


print(f"[smoke] quantize with {quant_cfg_name}")
mtq.quantize(base_model, quant_cfg, forward_loop=forward_loop)

print(f"[smoke] save checkpoint -> {out_dir}")
base_model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
state_path = os.path.join(out_dir, "modelopt_state.pth")
if not os.path.exists(state_path):
    raise RuntimeError(f"modelopt_state file not found: {state_path}")
print(f"[smoke] modelopt state exists: {state_path}")

del base_model
if device == "cuda":
    torch.cuda.empty_cache()

print("[smoke] load with ms-swift + enable_modelopt_hf")
model, _ = get_model_processor(
    out_dir,
    use_hf=True,
    load_model=True,
    model_type=model_type,
    device_map=device_map,
    torch_dtype=torch_dtype,
    enable_modelopt_hf=True,
)

is_converted = mto.ModeloptStateManager.is_converted(model)
print(f"[smoke] restored modelopt state: {is_converted}")
if not is_converted:
    raise RuntimeError("modelopt state was not restored")

print("[smoke] run one train step")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
train_batch = tokenizer("modelopt hf smoke training step", return_tensors="pt").to(device)
loss = model(**train_batch, labels=train_batch["input_ids"]).loss
loss.backward()
optimizer.step()
optimizer.zero_grad()

print(f"[smoke] train step loss = {float(loss.detach().cpu()):.6f}")
print("SMOKE_TEST_PASS")
PY

echo "Smoke test finished."
if [[ "${KEEP_ARTIFACTS}" -eq 1 || "${CREATED_TMP}" -eq 0 ]]; then
    echo "Artifacts kept at: ${OUTPUT_DIR}"
fi
