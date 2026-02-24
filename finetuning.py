import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# ==========================================
# 설정
# ==========================================
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

TEACHER_ID = "./base_model"        # 원본 EXAONE
STUDENT_ID = "./model_sparsegpt"   # SparseGPT 프루닝 모델
OUT_DIR = "./model_distilled"

# ==========================================
# 1. 모델 로드
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(
    STUDENT_ID, 
)

student_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

teacher_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

teacher_model.eval()
for p in teacher_model.parameters():
    p.requires_grad = False

# ==========================================
# ⭐ 2. SparseGPT 마스크 저장 (핵심)
# ==========================================
print("=== Saving pruning masks ===")

masks = {}

for name, p in student_model.named_parameters():
    if "mlp" in name:  # 프루닝된 부분만 적용
        zero_mask = (p == 0)
        if zero_mask.any():
            masks[name] = zero_mask
            sparsity = zero_mask.float().mean().item()
            print(f"{name} sparsity = {sparsity:.3f}")

print(f"Total masked tensors: {len(masks)}")

# ==========================================
# 3. 데이터 로드
# ==========================================
dataset = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:8192]")

def preprocess(example):
    text = tokenizer.apply_chat_template(
        example["conversations"],
        add_generation_prompt=True,
        tokenize=False
    )

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=4096,
        return_tensors=None
    )

    text = tokenizer.decode(tokens["input_ids"])

    return {"text": text}

dataset = dataset.map(preprocess)

# ==========================================
# ⭐ 4. KD + Mask 유지 Trainer
# ==========================================
class KDTrainer(SFTTrainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        kd_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean"
        )

        loss = ce_loss + 0.5 * kd_loss
        return (loss, student_outputs) if return_outputs else loss

    def optimizer_step(self, *args, **kwargs):
        # 1. 업데이트 전 그라디언트 마스킹
        for name, p in self.model.named_parameters():
            if name in masks and p.grad is not None:
                p.grad[masks[name]] = 0
                
        super().optimizer_step(*args, **kwargs)

        # 2. 업데이트 후 가중치 0 고정 (2:4 Sparsity 유지)
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in masks:
                    p.data[masks[name]] = 0

# ==========================================
# 5. 학습 설정
# ==========================================
args = TrainingArguments(
    output_dir=OUT_DIR,
    max_steps=5000,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    bf16=True,
    tf32=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=1000,
    report_to="none"
)

trainer = KDTrainer(
    model=student_model,
    train_dataset=dataset,
    args=args,
)
# ==========================================
# 6. 학습
# ==========================================
trainer.train()

# ==========================================
# 7. Sparsity 유지 확인
# ==========================================
print("=== Sparsity after training ===")

for name, p in student_model.named_parameters():
    if name in masks:
        zero_ratio = (p == 0).float().mean().item()
        print(name, zero_ratio)

# ==========================================
# 8. 저장
# ==========================================
student_model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

print(f"[INFO] 학습 및 저장 완료: {OUT_DIR}")