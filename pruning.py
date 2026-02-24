import os
import torch
import shutil
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import SparseGPTModifier # SparseGPT 모디파이어

# ##########################################
# # 1. Setting (기존 로직 유지)
# ##########################################
MODEL_ID = "./base_model"     
OUT_DIR  = "./model_sparsegpt"          

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 1024
MAX_SEQUENCE_LENGTH = 512

# SparseGPT Pruning Settings
SPARSITY = 0.5            # 50% 희소성
MASK_STRUCTURE = "2:4"    # vLLM 하드웨어 가속용 2:4 패턴
TARGETS = ["Linear"]

# --- 레이어 제외 로직 (사용자 notebook 로직 반영) ---


IGNORE = ["re:.*embed_tokens", "re:.*lm_head", "re:.*norm", "re:.*rotary_emb"] + ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj", "re:.*o_proj"] + ["re:.*q_norm", "re:.*k_norm"] + ["re:.*post_attention_layrenorm", "re:.*post_feedforward_layrenorm"]

skip_indices = list(range(0, 5)) + list(range(25, 30))
skip_layers0 = [f"re:.*model.layers.{i}.mlp.gate_proj" for i in skip_indices]
skip_layers1 = [f"re:.*model.layers.{i}.mlp.up_proj" for i in skip_indices]
skip_layers2 = [f"re:.*model.layers.{i}.mlp.down_proj" for i in skip_indices]

IGNORE += skip_layers0 + skip_layers1 + skip_layers2

# ##########################################
# # 2. Model & Tokenizer Loads
# ##########################################
print("[INFO] 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# ##########################################
# # 3. Dataset Loads & Preprocess
# ##########################################
print("[INFO] 데이터 전처리 중...")
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False)
    }

ds = ds.map(preprocess)

# ##########################################
# # 4. SparseGPT Pruning 실행
# ##########################################
print(f"[INFO] SparseGPT 프루닝 시작 (sparsity={SPARSITY})...")

recipe = [
    SparseGPTModifier(
        sparsity=SPARSITY,
        mask_structure=MASK_STRUCTURE,
        targets=TARGETS,
        ignore=IGNORE,
        # SparseGPT 전용 옵션 (필요시 조절)
        # block_size=128, 
        # dampening_frac=0.01 
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# ##########################################
# # 5. 저장 및 압축
# ##########################################
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

print(f"[INFO] SparseGPT 완료 및 저장: {OUT_DIR}")