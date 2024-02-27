import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from texts_processing import TextsTokenizer
from readers import ReaderRanker, Validator


tokenizer = TextsTokenizer()

stopwords = []
for fn in ["greetings.csv", "stopwords.csv"]:
    stw_df = pd.read_csv(os.path.join("data", fn), sep="\t")
    stopwords += stw_df["stopwords"].to_list()

tokenizer.add_stopwords(stopwords)

"""ранжируем моделями SentenceTransformer"""
st_model = SentenceTransformer(os.path.join("models", "all_sys_paraphrase.transformers"))
# st_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
reader = ReaderRanker(st_model)

"""ранжируем  моделью e5"""
# e5_tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
# e5_model = AutoModel.from_pretrained('intfloat/multilingual-e5-large').to('cuda')
# reader = ReaderRanker(e5_model, e5_tokenizer)

t5_tokenizer = T5Tokenizer.from_pretrained('ai-forever/ruT5-large')
t5_model = T5ForConditionalGeneration.from_pretrained(os.path.join("data", 'models_bss')).to("cuda")

validator = Validator(t5_model, t5_tokenizer)