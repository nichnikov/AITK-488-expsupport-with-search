import os
import time
import pandas as pd
from contextlib import suppress
from collections import namedtuple
from start import tokenizer, reader, validator
from retrivers import bm25retriver

q_df = pd.read_csv(os.path.join("data", "queries.csv"), sep="\t")
queries_dcts = q_df.to_dict(orient="records")

FoundDoc = namedtuple("FoundDoc", "Rank, DocName, LemFounText, FounText, Score")

search_results = []
for num, InputDict in enumerate(queries_dcts):
    with suppress(KeyError, ValueError):
        print(num, "/", len(queries_dcts))
        lm_query = " ".join(tokenizer([InputDict["Query"]])[0])
        seaching_results = bm25retriver(lm_query, 30)

        """adding lem searching_text in searching results dictionaries"""
        lm_searching_texts = [" ".join(lm_tx) for lm_tx in tokenizer([d["searching_text"] for d in seaching_results])]
        seaching_results_with_lm = [{**d, **{"lem_searching_text": lm_tx}} for lm_tx, d in zip(lm_searching_texts, seaching_results)]


        candidates = [(d["id"], d["doc_name"], d["lem_searching_text"], d["searching_text"]) for d in seaching_results_with_lm]
        the_best_result = [FoundDoc(*x) for x in reader(lm_query, candidates, 1)][0]
        ResDict = the_best_result._asdict()
        
        ValDict = validator(lm_query, the_best_result.FounText, 0.0)
       
        search_results.append({**InputDict, **ResDict, **ValDict})
        # print(search_results)
        
        search_results_df = pd.DataFrame(search_results)
        search_results_df.to_csv(os.path.join("results", "search_results.csv"), index=False, sep="\t")
        time.sleep(0.5)
    
