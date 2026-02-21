import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# 1. 환경 설정 및 경로 수정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] {device.upper()} 장치에서 실행을 시작합니다.")

# 사용자 요청 경로로 수정 완료
model_id = "C:/Users/fuco2/Desktop/open/base_model"
quant_model_path = "C:/Users/fuco2/Desktop/open/ign5_qkvinclude/model"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. 모델 로드 (해커톤 환경 최적화: CUDA 12.8 + torch 2.9.0 대응)
# GPU 환경이면 float16을 사용하여 속도(SpeedNorm)를 높입니다.
model_ref = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

model_quant = AutoModelForCausalLM.from_pretrained(
    quant_model_path, 
    device_map="auto"
)

# 의미론적 유사도(PerfNorm용) 모델
sim_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def evaluate_metrics(prompt):
    messages = [{"role": "user", "content": prompt}]
    input_tensor = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    input_len = input_tensor.shape[1]

    # --- [Reference 모델] 추론 및 TPS(Time Per Token) 측정 ---
    start_ref = time.time()
    out_ref = model_ref.generate(input_tensor, max_new_tokens=64, do_sample=False)
    time_ref = time.time() - start_ref
    
    ans_ref = tokenizer.decode(out_ref[0][input_len:], skip_special_tokens=True)
    tokens_ref = len(out_ref[0][input_len:])
    tps_ref = time_ref / tokens_ref if tokens_ref > 0 else 1e-9
    print(tps_ref)

    # --- [Quantized 모델] 추론 및 TPS 측정 ---
    start_quant = time.time()
    out_quant = model_quant.generate(input_tensor, max_new_tokens=64, do_sample=False)
    time_quant = time.time() - start_quant
    
    ans_quant = tokenizer.decode(out_quant[0][input_len:], skip_special_tokens=True)
    tokens_quant = len(out_quant[0][input_len:])
    tps_quant = time_quant / tokens_quant if tokens_quant > 0 else 1e-9
    print(tps_quant)

    # --- [이미지 수식 계산 시작] ---
    
    # 1. PerfNorm (성능 유지 비율): 유사도를 Perf 지표로 활용
    emb_ref = sim_model.encode(ans_ref, convert_to_tensor=True)
    emb_quant = sim_model.encode(ans_quant, convert_to_tensor=True)
    perf_norm = util.pytorch_cos_sim(emb_ref, emb_quant).item()

    # 2. SpeedNorm (추론 시간 감소 비율): 1 - (양자화 TPS / 베이스 TPS)
    speed_norm = tps_ref / tps_quant

    # 3. Final Score: max(0.5 * PerfNorm + 0.5 * SpeedNorm, 0)
    total_score = max(0.5 * perf_norm + 0.5 * speed_norm, 0)

    return {
        "ans_quant": ans_quant,
        "perf_norm": perf_norm,
        "speed_norm": speed_norm,
        "total_score": total_score
    }

# 3. 테스트 실행
test_prompts = [
    "대한민국의 수도는 어디인가요?",
    "양자 역학에 대해 짧게 설명해줘.",
    "사과 3개 중 2개를 먹으면 몇 개가 남지?"
]

print(f"\n{'Prompt':<15} | {'PerfNorm':<10} | {'SpeedNorm':<10} | {'Total Score':<12}")
print("-" * 65)

for p in test_prompts:
    res = evaluate_metrics(p)
    print(f"{p[:10]}... | {res['perf_norm']:.4f} | {res['speed_norm']:.4f} | {res['total_score']:.4f}")
    print(f"  └─ [Quant]: {res['ans_quant'][:50]}...\n")