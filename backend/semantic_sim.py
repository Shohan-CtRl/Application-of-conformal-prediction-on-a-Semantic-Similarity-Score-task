from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer, models
from transformers import BertTokenizer, BertModel
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from scipy.spatial import distance
import json
import re
import string


app = Flask(__name__)
CORS(app)
# API Route

@app.route("/sts", methods=["POST"])
def sts():
    data = request.json
    text1 = data.get("input_data1")
    text2 = data.get("input_data2")
    
    sim = fetchSTS(text1, text2)
    cp = fetchCP(text1, text2)
    print(cp)
    
    return jsonify({"result": sim, "CP": cp})




class STSBertModel(torch.nn.Module):

    def __init__(self):

        super(STSBertModel, self).__init__()

        word_embedding_model = models.Transformer('sentence-transformers/stsb-bert-base', max_seq_length=128)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.sts_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def forward(self, input_data):

        output = self.sts_model(input_data)
        
        return output
def fetchSTS(text1, text2):   
    model = STSBertModel()
    model = torch.load("./backend/trained_modelFinalv2.pth", map_location=torch.device('cpu'))

    model.to('cpu')
    model.eval()
    texts = [text1,text2]
    # Lowercase and remove punctuation and special characters
    texts = [text.lower() for text in texts]
    texts = [text.translate(str.maketrans('', '', string.punctuation)) for text in texts]
    texts = [re.sub(r'[^a-zA-Z0-9\s]', '', text) for text in texts]

    # Set the tokenizer
    tokenizer = BertTokenizer.from_pretrained('sentence-transformers/stsb-bert-base')
    test_input = tokenizer(texts, padding='max_length', max_length = 128, truncation=True, return_tensors="pt")
    test_input['input_ids'] = test_input['input_ids']
    test_input['attention_mask'] = test_input['attention_mask']
    del test_input['token_type_ids']


    test_output = model(test_input)['sentence_embedding']
    sim = torch.nn.functional.cosine_similarity(test_output[0], test_output[1], dim=0).item()
    sim = round(sim, 3)
    return sim
def fetchCP(s1,s2):
    # Open the file in read mode
    with open('./backend/calibration_data.json', 'r') as f:
        # Load the contents of the file into a variable
        data = json.load(f)
    # convert the JSON data to a list of dictionaries
    calibration_set = json.loads(data)

 
    # Normalise the data
    similarity_scores_not_normalised = np.array([d['similarity_score'] for d in calibration_set])
    max_score = max(similarity_scores_not_normalised)
    similarity_scores = [score / max_score for score in similarity_scores_not_normalised]
    # convert similarity_scores to a numpy array
    similarity_scores_np = np.array(similarity_scores)    
    # find the k nearest similarity scores to the input score
    k = 10
    input_score = fetchSTS(s1, s2) 
    nearest_indices = np.argsort(np.abs(similarity_scores_np - input_score))[:k]

    # extract the sentences corresponding to the k nearest similarity scores
    nearest_sentences = [(calibration_set[i]['sentence1'], calibration_set[i]['sentence2'], calibration_set[i]['similarity_score']) for i in nearest_indices]

    nonconformity_scores = []
    similarity_scores_nearest = []

    for sentence in nearest_sentences:
        # calculate the similarity scores for the k nearest sentences using the BERT model
        similarity_score_nearest = fetchSTS(sentence[0], sentence[1]) 
        similarity_scores_nearest.append(similarity_score_nearest)


    for i in range(len(similarity_scores_nearest)): 
        # calculate the nonconformity scores
        nonconformity_score = np.abs(similarity_scores_nearest[i] - (nearest_sentences[i][2]/5))
        nonconformity_scores.append(nonconformity_score)
        

    # calculate the conformal prediction intervals
    alpha = 0.1
    num_samples = 1000
    conformal_intervals = []
    for i in range(k):
        pi_lower = max(0, input_score - np.percentile(nonconformity_scores, (1 - alpha) * 100 / 2))
        pi_upper = min(1, input_score + np.percentile(nonconformity_scores, (1 - alpha) * 100 / 2))
        conformal_intervals.append((pi_lower, pi_upper))
            
    min_lower_bound = min([pi[0] for pi in conformal_intervals])
    max_upper_bound = max([pi[1] for pi in conformal_intervals])
    interval =(min_lower_bound, max_upper_bound)
    print("Single conformal prediction interval:", interval)
    print("Semantic similarity: ", input_score)
    
    return interval

if __name__ == "__main__":
    app.run(debug=True)


