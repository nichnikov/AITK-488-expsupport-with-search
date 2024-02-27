import os
import torch
from sentence_transformers import SentenceTransformer

st_model = SentenceTransformer(
            str(os.path.join("models", "all_sys_paraphrase.transformers")),
            device="cuda" if torch.cuda.is_available() else "cpu")

tokens_str = ""
emdgs = st_model.encode(tokens_str, batch_size=64, show_progress_bar=False)
