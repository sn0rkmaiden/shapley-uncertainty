from lm_polygraph.utils.model import BlackboxModel
from lm_polygraph.estimators import CoCoA
from lm_polygraph import estimate_uncertainty
import os

HUGGINGFACE_API_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_NAME")
model = BlackboxModel.from_huggingface(hf_api_token=HUGGINGFACE_API_TOKEN, hf_model_id=MODEL_ID, openai_api_key=None, openai_model_path=None)

ue_method = CoCoA()
input_text = 'How many floors are in the Empire State Building? Just answer the question.'
estimate_uncertainty(model, ue_method, input_text=input_text)