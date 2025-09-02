import pandas as pd
from pathlib import Path
from utils.functions import *
from utils.nli import NLIModel
import numpy as np
import matplotlib.pyplot as plt

def load_iglu(path='./data/iglu/clarifying_questions_train.csv'):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"IGLU file not found at {path}. Inspect the repo to find the correct file.")
    data = pd.read_csv(path)
    return data

def create_iglu_prompt(data, index):
    row = data.iloc[index]
    return row

def batch_shapley_eval(n_samples=5, M=500):
    import json
    from tqdm import tqdm
    df = load_iglu()

    model = NLIModel(model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    results = {'clear_instruction': [], 'ambiguous_instruction': []}
    stats = {}
    for cat in ['Yes', 'No']:
        subset = df[df['IsInstructionClear'] == cat].reset_index(drop=True)
        sample_indices = np.random.choice(len(subset), min(n_samples, len(subset)), replace=False)
        sample_results = []
        uncertainties = []
        print(f"Evaluating {'clear_instruction' if cat == 'Yes' else 'ambiguous_instruction'} samples...")
        for idx in tqdm(sample_indices, desc=f"{'clear_instruction' if cat == 'Yes' else 'ambiguous_instruction'}"):
            prompt = create_iglu_prompt(subset, idx)
            res = model.shapley_uncertainty_for_prompt(prompt, n_samples=n_samples, M=M)
            sample_results.append({
                'prompt': prompt,
                'total_uncertainty': res.get('total_uncertainty'),
                'uncertainty': res.get('uncertainty'),
                'confidence_gap': res.get('confidence_gap'),
                'phi': res.get('phi').tolist() if hasattr(res.get('phi'), 'tolist') else res.get('phi'),
                'samples': res.get('samples')
            })
            uncertainties.append(res.get('uncertainty'))
        results['clear_instruction' if cat == 'Yes' else 'ambiguous_instruction'] = sample_results
        stats['clear_instruction' if cat == 'Yes' else 'ambiguous_instruction'] = {
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