import numpy as np
import os, json, math, random, time, pathlib, glob, re
from typing import List, Dict, Any

# Optional: openai API imports
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    OpenAI = None

# Huggingface transformers imports
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError:
    AutoModelForCausalLM = None

def make_openai_client():
    if OpenAI is None:
        raise RuntimeError("openai package not installed.")
    key = os.getenv("HF_TOKEN")
    if not key:
        raise RuntimeError("HF_TOKEN env variable is required.")
    client = OpenAI(api_key=key, base_url=os.getenv("BASE_URL"))
    return client

def sample_generations_api(prompt: str, n: int = 20, max_tokens: int = 96, temperature: float = 0.8, top_p: float = 0.95, system: str = "You ask concise, specific clarifying questions.") -> list:
    if OpenAI is None:
        raise RuntimeError("openai package not installed.")
    client = make_openai_client()
    out = []
    for i in range(n):
        resp = client.chat.completions.create(
            model=os.getenv("MODEL_NAME"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=False
        )
        out.append(resp.choices[0].message.content.strip())
        time.sleep(0.05)
    uniq = []
    seen = set()
    for t in out:
        t1 = t.split("\n")[0].strip()
        if t1 and t1 not in seen:
            seen.add(t1); uniq.append(t1)
    return uniq[:n]

# Huggingface local LLM generation
def load_hf_llm(model_name: str = "google/gemma-2b-it", device: str = None):
    """
    Load a Huggingface causal LM and tokenizer.
    """
    if AutoModelForCausalLM is None:
        raise RuntimeError("transformers package not installed.")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    return model, tokenizer, device

def sample_generations_hf(prompt: str, n: int = 20, max_tokens: int = 96, temperature: float = 0.8, top_p: float = 0.95,
                          model_name: str = "google/gemma-2b-it", system: str = None, device: str = None) -> list:
    """
    Generate samples using a local Huggingface LLM.
    """
    model, tokenizer, device = load_hf_llm(model_name, device)
    input_text = prompt if system is None else f"{system}\n{prompt}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    out = []
    for _ in range(n):
        gen_ids = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        gen_text = tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        out.append(gen_text)
        time.sleep(0.01)
    uniq = []
    seen = set()
    for t in out:
        t1 = t.split("\n")[0].strip()
        if t1 and t1 not in seen:
            seen.add(t1); uniq.append(t1)
    return uniq[:n]

def load_clarq_prompts(root="./data/English"):
    print(f"Loading ClarQ prompts from {root} ...")
    prompts = []
    for path in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        # Two possible shapes: list of dicts with fields, or dict with 'tasks'
        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if "tasks" in data and isinstance(data["tasks"], list):
                items = data["tasks"]
            else:
                # try flatten
                items = [v for v in data.values() if isinstance(v, dict)]

        for ex in items:
            # Heuristics: look for English description/background/context fields
            cand = ex.get("description") or ex.get("background") or ex.get("context") or ex.get("instruction")
            if not cand:
                # Sometimes stored under 'system_prompt' or 'seeker_background'
                cand = ex.get("system_prompt") or ex.get("seeker_background")
            if not cand:
                # Try concatenating known fields
                parts = []
                for key in ["goal", "scenario", "task", "role", "items", "skills", "scenes"]:
                    if key in ex and isinstance(ex[key], (str, list)):
                        parts.append(ex[key] if isinstance(ex[key], str) else "\n".join(ex[key]))
                cand = "\n".join(parts) if parts else None
            if cand and isinstance(cand, str) and len(cand.strip())>0:
                # Shorten super-long descriptions
                prompts.append(cand.strip())
    seen=set(); uniq=[]
    for p in prompts:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

def build_kernel(P: np.ndarray, beta: float = 0.5, eps: float = 1e-6) -> np.ndarray:
    S = 0.5 * (P + P.T)  # Symmetric similarity matrix
    K = beta * np.exp(-(1 - S)**2 / 2)  # Gaussian kernel
    np.fill_diagonal(K, 1.0)  # Set diagonal to 1
    K = 0.5 * (K + K.T) + eps * np.eye(len(S))  # Ensure symmetry and add jitter
    return K

def set_function(S_idx: list, K: np.ndarray, q: np.ndarray | None = None, delta: float = 1e-6) -> float:
    if len(S_idx) == 0:
        return 0.0
    Ks = K[np.ix_(S_idx, S_idx)]
    sign, logdet = np.linalg.slogdet(Ks + delta*np.eye(len(S_idx)))
    val = float(logdet)
    if q is not None:
        val += float(np.sum(q[S_idx]))
    return val

def shapley_values(n: int, v_fn, M: int = 1000, seed: int = 0) -> np.ndarray:
    rng = random.Random(seed)
    order = list(range(n))
    phi = np.zeros(n, dtype=np.float64)
    for _ in range(M):
        rng.shuffle(order)
        S = []
        v_S = 0.0
        for j in order:
            v_Sj = v_fn(S + [j])
            phi[j] += (v_Sj - v_S)
            v_S = v_Sj
            S.append(j)
    return phi / M
