import fasttext, fasttext.util
import demoji
import re
from scipy import spatial

# load model
fasttext.util.download_model('en', if_exists='ignore')  # English
ft = fasttext.load_model('cc.en.300.bin')

# preprocess
def preprocess(x):
    x = demoji.replace(x, '')
    x = re.sub(r'[^a-zA-Z0-9 ]', '', x)
    return x.strip().lower()

def cosine_similarity(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)

def embed(x):
    x = preprocess(x)
    if x == '':
        return None
    return ft.get_sentence_vector(x)