import requests
import json

VERIFOLIO_URL = r'http://127.0.0.1:5000/extract'

def extract_features(sourcefile):
    with open(sourcefile) as f:
        resp = requests.post(VERIFOLIO_URL, files={'file': f})
    if resp.status_code != '200':
        raise requests.ConnectionError('Could not communicate with verifolio service')
    return json.loads(resp.text)
