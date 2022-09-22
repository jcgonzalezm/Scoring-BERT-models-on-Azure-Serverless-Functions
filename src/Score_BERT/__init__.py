import azure.functions as func
import logging
import os
import sys
import json

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)
import score_bert

def main(req: func.HttpRequest, 
        context: func.Context) -> func.HttpResponse:

    email_data = json.dumps(req.get_json())
        
    try:
        logging.info('calling score_bert.py')
        
        resp = score_bert.run(email_data)
        
        resp = json.dumps(resp)
        logging.info('resp: %s; type %s', resp , type(resp))
    except Exception as er:
        resp = {'resp': 'False'}
        logging.error('problems on score -> %s', er)
        
    return func.HttpResponse(resp)