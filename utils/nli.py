import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from functions import *

device = "cuda" if torch.cuda.is_available() else "cpu"
nli_name = "microsoft/deberta-large-mnli"
nli_tok = AutoTokenizer.from_pretrained(nli_name)
nli = AutoModelForSequenceClassification.from_pretrained(nli_name).to(device).eval()
print(f"NLI model loaded on {device}: {nli_name}")

@torch.no_grad()
def entail_prob_batch(prem_list: list, hyp_list: list, batch_size: int = 8) -> np.ndarray:
    assert len(prem_list) == len(hyp_list)
    out = []
    for i in range(0, len(prem_list), batch_size):
        prem = prem_list[i:i+batch_size]
        hyp  = hyp_list[i:i+batch_size]
        enc = nli_tok(prem, hyp, return_tensors="pt", truncation=True, padding=True, max_length=384).to(device)
        logits = nli(**enc).logits.float()
        probs = torch.softmax(logits, dim=-1)[:, 2]  # entailment index
        out.append(probs.detach().cpu().numpy())
    return np.concatenate(out, axis=0)

def pairwise_entailment_matrix(samples: list) -> np.ndarray:
    n = len(samples)
    P = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        prem = [samples[i]] * n
        hyp  = list(samples)
        P[i,:] = entail_prob_batch(prem, hyp)
    return P

def shapley_uncertainty_for_prompt(prompt: str, gold: str | None = None, n_samples: int = 20, M: int = 1000,
                                   sigma: float = 0.25, delta: float = 1e-6,
                                   temperature: float = 0.8, top_p: float = 0.95, max_tokens: int = 96,
                                   system_msg: str = "You ask concise, specific clarifying questions.") -> dict:
    samples = sample_generations_api(prompt, n=n_samples, max_tokens=max_tokens, temperature=temperature, top_p=top_p, system=system_msg)
    if len(samples) < 2:
        return {"uncertainty": 0.0, "phi": np.zeros(len(samples)), "samples": samples}
    P = pairwise_entailment_matrix(samples)
    K = build_kernel(P, sigma=sigma)
    q = None
    if gold:
        ent1 = entail_prob_batch([gold]*len(samples), samples)
        ent2 = entail_prob_batch(samples, [gold]*len(samples))
        q = (ent1 + ent2)/2.0
    def v(S):
        return set_function(S, K, q=q, delta=delta)
    phi = shapley_values(len(samples), v, M=M)
    # normalize into a probability-like vector (shift to non-neg)
    w = np.maximum(phi - phi.min(), 0)
    if w.sum() <= 0:
        entropy = 0.0; confgap = 1.0
    else:
        p = w / w.sum()
        entropy = -float(np.sum(p * np.log(p + 1e-12))) / np.log(len(p))
        confgap = 1.0 - float(p.max())
    return {"total_uncertainty": np.sum(phi), "uncertainty": float(entropy), "confidence_gap": float(confgap), "phi": phi, "samples": samples, "kernel": K, "P": P}