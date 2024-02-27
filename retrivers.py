import re
import requests

def bm25retriver(searchString: str, docsCount: int):
    rc_url = "http://release-service-search.rc.search.aservices.tech/api/v2/emsearch"

    template = {
        "snippetSize": 30,
        "docsCount": docsCount,
        "fixedRegionCode": 0,
        "pageNumber": 1,
        "sortBy": "Relevance",
        "newWizardsOnly": True,
        "noWizards": True,
        "noFix" : True,
        "sortOrder": "Desc",
        "searchString": searchString,
        "pubId": 9,
        "pubDivId": 1,
        "searchTagList": [
            
        ],
        "segmentIdList": [
            
        ],
        "areaId": None,
        "aggregationFilter": None,
        "publicationSchemeV2": False
    }

    headers = {'Content-Type': 'application/json', 'charset': 'utf-8', 'x-source': 'ml-ci'}
    # res = requests.post(rc_url, json=template, headers=headers)
    res = requests.post(rc_url, json=template)
    res_dict = res.json()

    parrents = re.compile("</b>|<b>|\\xa0|&#160;|&#34;|&#8211")
    snippets = [parrents.sub(r" ", d["snippet"]) for d in res_dict["items"]]
    docNames = [parrents.sub(r" ", d["docName"]) for d in res_dict["items"]]
    
    return [{"id": num + 1, "doc_name": dn, "searching_text": re.sub(" +", " ", " ".join([dn, sn]))} for num, dn, sn in 
            zip(range(len(docNames)), docNames, snippets)]

    