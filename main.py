from utils.functions import *
from utils.nli import *

clarq_prompts = load_clarq_prompts()
print(f"Loaded {len(clarq_prompts)} ClarQ prompts. Example:\n{clarq_prompts[0]}")

TEST_SAMPLES = 4
TEST_PERMS = 500
model = NLIModel()
demo_prompt = clarq_prompts[0]
res = model.shapley_uncertainty_for_prompt(demo_prompt, n_samples=TEST_SAMPLES, M=TEST_PERMS)
print("Total uncertainty:", res["total_uncertainty"])
print("Uncertainty entropy:", res["uncertainty"]) 
print("Confidence gap:", res["confidence_gap"]) 
print("#samples:", len(res["samples"]))
print("Top-3 generations by Shapley contribution:")
idx = list(range(len(res["phi"])))
idx.sort(key=lambda i: res["phi"][i], reverse=True)
for i in idx[:3]:
    print(f"[{i}] phi={res['phi'][i]:.4f} :: {res['samples'][i]}")