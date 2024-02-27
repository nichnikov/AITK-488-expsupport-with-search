import os
import time
import pandas as pd
from contextlib import suppress
from start import tokenizer, reader, validator
from retrivers import bm25retriver

q_df = pd.read_csv(os.path.join("data", "search_queries.csv"), sep="\t")
queries = q_df["Query"].to_list()

result_dfs = []
for num, query in enumerate(queries):
    with suppress(KeyError):
        print(num, "/", len(queries))
        lm_query = " ".join(tokenizer([query])[0])
        seaching_results = bm25retriver(query, 30)

        """adding lem searching_text in searching results dictionaries"""
        lm_searching_texts = [" ".join(lm_tx) for lm_tx in tokenizer([d["searching_text"] for d in seaching_results])]
        seaching_results_with_lm = [{**d, **{"lem_searching_text": lm_tx}} for lm_tx, d in zip(lm_searching_texts, seaching_results)]


        candidates = [(d["id"], d["doc_name"], d["lem_searching_text"], d["searching_text"]) for d in seaching_results_with_lm]
        ranking_results = reader(lm_query, candidates, 1)
        
        validator

        ranking_results_dicts = [{
            "Query": query,
            "search_rank": r,
            "reader_doc_name": dn,
            "reader_text": tx, 
            "reader_score": sc} for r, dn, lm_tx, tx, sc in ranking_results]

        result_compare = [{**rr, **sr} for rr, sr in zip(ranking_results_dicts, seaching_results[:3])]
        result_dfs.append(pd.DataFrame(result_compare))
    
        result_compare_df = pd.concat(result_dfs)
        result_compare_df.to_csv(os.path.join("results", "bm25_e5_compare_box.csv"), index=False, sep="\t")
        time.sleep(2)
    
