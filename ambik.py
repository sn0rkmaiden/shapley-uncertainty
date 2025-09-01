import pandas as pd
from pathlib import Path
from utils.functions import *
from utils.nli import NLIModel

def load_ambik(path="./data/ambik_dataset/ambik_test_400.csv"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"AmbiK file not found at {path}. Inspect the repo to find the correct file.")
    data = pd.read_csv(path)
    return data

def create_ambik_prompt(data, index):
    row = data.iloc[index]
    env = row['environment_full']
    ambig = row['ambiguous_task']
    unamb = row['unambiguous_direct']
    q_true = row['question']
    a_true = row['answer']
    plan_true = row['plan_for_clear_task']

    prompt = f"Environment: {env}\nInstruction: {ambig}\nWhat should you do next? Ask a question if needed."

    return prompt

df = load_ambik()
print(f"Loaded {len(df)} records. Columns:", list(df.columns))
model = NLIModel()

prefs = df[df['ambiguity_type'] == 'preferences']
commons = df[df['ambiguity_type'] == 'common_sense_knowledge']

prompt_prefs = create_ambik_prompt(prefs, 0)
prompt_commons = create_ambik_prompt(commons, 0)

res1 = model.shapley_uncertainty_for_prompt(prompt_prefs, n_samples=8, M=1000)
res2 = model.shapley_uncertainty_for_prompt(prompt_commons, n_samples=8, M=1000)

print(res1['uncertainty'])
print(res2['uncertainty'])
