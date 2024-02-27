import re
from collections import namedtuple
from torch import Tensor
from transformers import XLMRobertaModel, XLMRobertaTokenizerFast
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import torch

Candidate = namedtuple("Candidate", "Rank, DocName, LemInputText, InputText")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class ReaderRanker:
        """_summary_
        """
        def __init__(self, model, tokenizer=None) -> None:
               self.model = model
               self.tokenizer = tokenizer
        
        def e5_txs2enbs(self, texts: list[str]):
                """переводит тексты в векторы для модели е5"""
                txts_chunks = chunks(texts, 5)
                vectors = []
                for txs in  txts_chunks:
                        batch_dict = self.tokenizer(txs, max_length=512, padding=True, 
                                                        truncation=True, return_tensors='pt').to('cuda')
                        outputs = self.model(**batch_dict)
                        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                        embeddings = F.normalize(embeddings, p=2, dim=1)
                        vectors += [torch.tensor(emb, device='cpu') for emb in  embeddings]
                return vectors
        
        def ranking(self, lem_searched_text: str, candidates: list[Candidate], k: int):
                ranks, doc_names, lm_search_texts, search_texts = zip(*candidates)

                if isinstance(self.model, SentenceTransformer) and self.tokenizer is None:
                        text_emb = self.model.encode(lem_searched_text)
                        candidates_embs = self.model.encode(lm_search_texts)
                        scores = util.cos_sim(text_emb, candidates_embs)
                        
                elif isinstance(self.model, XLMRobertaModel) and isinstance(self.tokenizer, XLMRobertaTokenizerFast):
                        text_emb = self.e5_txs2enbs([lem_searched_text])
                        candidates_embs = self.e5_txs2enbs(lm_search_texts)
                        scores = cosine_similarity(text_emb, candidates_embs)
                
                scores_list = [score.item() for score in scores[0]]
                the_best_results = sorted(list(zip(ranks, doc_names, lm_search_texts, search_texts, scores_list)),
                                                        key=lambda x: x[4], reverse=True)[:k]
                
                return the_best_results

        def __call__(self, lem_searched_text: str, candidates: list[Candidate], k: int):
               return self.ranking(lem_searched_text, candidates, k)


class Validator:
        """_summary_
        """
        def __init__(self, t5_model, t5_tokenizer) -> None:
                self.device = "cuda"
                self.t5_model = t5_model.to(self.device)
                self.t5_tkz = t5_tokenizer

        
        def t5_validate(self, query: str, answer: str, score: float):
                text = query + " Document: " + answer + " Relevant: "
                input_ids = self.t5_tkz.encode(text,  return_tensors="pt").to(self.device)
                outputs=self.t5_model.generate(input_ids, eos_token_id=self.t5_tkz.eos_token_id, 
                                        max_length=64, early_stopping=True).to(self.device)
                outputs_decode = self.t5_tkz.decode(outputs[0][1:])
                outputs_logits=self.t5_model.generate(input_ids, output_scores=True, return_dict_in_generate=True, 
                                                eos_token_id=self.t5_tkz.eos_token_id, 
                                                max_length=64, early_stopping=True)
                sigmoid_0 = torch.sigmoid(outputs_logits.scores[0][0])
                t5_score = sigmoid_0[2].item()
                val_str = re.sub("</s>", "", outputs_decode)
                #logger.info("t5_validate answer is {} with score = {}".format(val_str, str(t5_score)))
                # if val_str == "Правда" and t5_score >= score:
                return {"Opinion": val_str, "Confidence": t5_score}
        
        def __call__(self, query: str, answer: str, score: float):
               return self.t5_validate(query, answer, score)
       