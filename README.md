# Shapley Uncertainty
My implementation of the concept described in the paper "Shapley Uncertainty in Natural Language Generation" 

Differential entropy:

![Differential entropy](https://github.com/sn0rkmaiden/shapley-uncertainty/blob/main/images/differential_entropy.png?raw=true)

Shapley value:

![Shapley values](https://github.com/sn0rkmaiden/shapley-uncertainty/blob/main/images/shapley_values.jpg?raw=true)

Total uncertainty:

![Total uncertainty](https://github.com/sn0rkmaiden/shapley-uncertainty/blob/main/images/total_uncertainty.jpg?raw=true)

# Getting started

Create Python virtual environment: `python -m venv venv`

Install all required dependencies: `pip install -r requirements.txt`

Create file `.env` and add following fields: "HF_TOKEN", "BASE_URL", "MODEL_NAME".
___

- `test.py` - Shapley uncertainty for the example from the paper.
- `main.py` - Shapley uncertainty for examples from [ClarQ-LLM](https://github.com/ygan/ClarQ-LLM) dataset
- `ambik.py` - calculates Shapley uncertainty for two examples from AmbiK
- `ambik_batch_eval.py` - calculates Shapley uncertainty for a batch from AmbiK


