import os
from openai import OpenAI
from typing import List

os.environ["OPENAI_API_KEY"] = "paste your own key"

client = OpenAI(api_key='paste your own key')  
               

def explain_molecule(method, smiles, classification, imgURL):
    question = "You are a scientist analyzing the toxicity of various molecules using Explainable AI (XAI) methods. You are currently using the {} explainer to interpret the contributions of different molecular structures to their overall toxicity. Input SMILES: {}, which you've determined to be {}. Attached Image Analysis: The image displays regions of the molecule highlighted in red and green. Red indicates areas that increase toxicity, while green indicates areas that decrease toxicity. Request: Explain how each part of the SMILES string contributes to the molecule's {} classification? Mention its toxicity and please describe which molecular components (highlighted in red and green) impact the toxicity and detail the process by which the model predicts the molecule as {}. Make it concise Answer in four parts: 1. Analyzing the molecular structure 2. Interpreting the color-coded regions 3. Model Prediction Process.".format(method, smiles, classification, method, classification, classification)
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a scientist analyzing the toxicity of various molecules using Explainable AI (XAI) methods."},
            {"role": "user", "content": [{"type": "text","text": question},{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgURL}"}}]}
        ]
    )
    
    return response.choices[0].message.content

def find_similar_smiles(smiles, imgURL) :
    question="The provided image is the visualized molecule graph of {}. {} is predicted as a toxic molecule and red indicates part increases toxicity. Give two toxic SMILES that indicate the same(similar) red part. ANSWER TEMPLETE: Give me two comma-separated SMILES with no other text. ".format(smiles, smiles)
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a scientist analyzing the toxicity of various molecules using Explainable AI (XAI) methods."},
            {"role": "user", "content": question},
            {"role": "user", "content": f"data:image/jpeg;base64,{imgURL}"}
        ]
    )
    
    return response.choices[0].message.content

    