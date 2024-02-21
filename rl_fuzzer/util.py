import requests

def get_toxicity(vId):
    r = requests.get(f'http://localhost:5050/perspective/{vId}')
    try:
        return r.json()['toxicity']
    except Exception as e: 
        return None

def is_bug(vId):
    toxicity_value = get_toxicity(vId)
    return toxicity_value is not None and toxicity_value > 0.8