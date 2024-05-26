import os
from openai import OpenAI
from typing import List

os.environ["OPENAI_API_KEY"] = "sk-proj-1WpTP6hVVMBEX89EdOb8T3BlbkFJUiLsTM0QO8E8yW9D1WDb"

client = OpenAI(api_key='sk-proj-1WpTP6hVVMBEX89EdOb8T3BlbkFJUiLsTM0QO8E8yW9D1WDb')  
               

def explain_molecule(method, smiles, classification, imgURL):
    question = "You are a scientist analyzing the toxicity of various molecules using Explainable AI (XAI) methods. You are currently using the {} explainer to interpret the contributions of different molecular structures to their overall toxicity. Input SMILES: {}, which you've determined to be {}. Attached Image Analysis: The image displays regions of the molecule highlighted in red and green. Red indicates areas that decrease toxicity, while green indicates areas that increase toxicity. Request: Using the {} method, explain how each part of the SMILES string contributes to the molecule's {} classification? Mention its toxicity and please describe which molecular components (highlighted in red and green) impact the toxicity and detail the process by which the model predicts the molecule as {}. Answer in four parts: 1. Understanding the {} method 2. Analyzing the molecular structure 3. Interpreting the color-coded regions 4. Model Prediction Process.".format(method, smiles, classification, method, classification, classification, method)
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a scientist analyzing the toxicity of various molecules using Explainable AI (XAI) methods."},
            {"role": "user", "content": [{"type": "text","text": question},{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgURL}"}}]}
        ]
    )
    
    return response.choices[0].message.content

    