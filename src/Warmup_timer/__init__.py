import datetime
import logging
import requests

import azure.functions as func

def main(mytimer: func.TimerRequest) -> None:

    myjson = {"data": 'THIS IS A WARMUP CALL'}

    url = "https://AzureFunctions/api/Score_BERT"
    auth_token = "Score_BERT_function_token" 
    
    hed = {'Authorization': 'Bearer ' + auth_token}
    r = requests.post(url, headers=hed, json=myjson) 

    pass # we do not do anything with this result

