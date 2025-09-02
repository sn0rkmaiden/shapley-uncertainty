import pandas as pd
from pathlib import Path
from utils.functions import *
from utils.nli import NLIModel
import numpy as np
import matplotlib.pyplot as plt

# Load AmbiK dataset
def load_ambik(path="./data/ambik_dataset/ambik_test_400.csv"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"AmbiK file not found at {path}. Inspect the repo to find the correct file.")
    data = pd.read_csv(path)
    return data

# Create prompt for AmbiK sample
def create_ambik_prompt(data, index):
    row = data.iloc[index]
    env = row['environment_full']
    ambig = row['ambiguous_task']
    prompt = f"Environment: {env}\nInstruction: {ambig}\nWhat should you do next? Ask a question if needed."
    return prompt

# Main batch evaluation
def batch_shapley_eval(n_samples=5, M=500):
    import json
    from tqdm import tqdm
    df = load_ambik()
    model = NLIModel(model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    results = {'preferences': [], 'common_sense_knowledge': []}
    stats = {}
    for cat in ['preferences', 'common_sense_knowledge']:
        subset = df[df['ambiguity_type'] == cat].reset_index(drop=True)
        sample_indices = np.random.choice(len(subset), min(n_samples, len(subset)), replace=False)
        sample_results = []
        uncertainties = []
        print(f"Evaluating {cat} samples...")
        for idx in tqdm(sample_indices, desc=f"{cat}"):
            prompt = create_ambik_prompt(subset, idx)
            res = model.shapley_uncertainty_for_prompt(prompt, n_samples=8, M=M)
            sample_results.append({
                'prompt': prompt,
                'total_uncertainty': res.get('total_uncertainty'),
                'uncertainty': res.get('uncertainty'),
                'confidence_gap': res.get('confidence_gap'),
                'phi': res.get('phi').tolist() if hasattr(res.get('phi'), 'tolist') else res.get('phi'),
                'samples': res.get('samples')
            })
            uncertainties.append(res.get('uncertainty'))
        results[cat] = sample_results
        stats[cat] = {
            'mean': float(np.mean(uncertainties)),
            'std': float(np.std(uncertainties)),
            'min': float(np.min(uncertainties)),
            'max': float(np.max(uncertainties)),
            'all': uncertainties
        }
    # Save summary statistics for plotting
    pd.DataFrame({k: [stats[k]['mean'], stats[k]['std'], stats[k]['min'], stats[k]['max']] for k in stats.keys()},
                 index=['mean', 'std', 'min', 'max']).to_csv('ambik_shapley_uncertainty_stats.csv')
    # Save full results for later analysis
    with open('ambik_shapley_uncertainty_full.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('Statistics:', stats)
    return stats

def plot_stats(stats):
    x = range(1, len(stats['preferences']['all']) + 1)
    plt.plot(x, stats['preferences']['all'], label='Preferences')
    plt.plot(x, stats['common_sense_knowledge']['all'], label='Common Sense Knowledge')
    plt.legend()
    plt.ylabel("Uncertainty (entropy)")
    plt.xlabel("Samples")
    plt.show()

if __name__ == "__main__":
    stats = batch_shapley_eval(n_samples=20, M=1000)
    plot_stats(stats)
