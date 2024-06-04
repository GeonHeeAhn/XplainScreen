# 🔑 XplainScreen : <br/> Unveiling the Black Box of GNN Drug Screening Models with a Unified XAI Framework

To address the inter-pretability issues associated with GNN-based virtual drug screening, we introduce ***XplainScreen***: a unified explanation framework designed to evaluate various explanation methods for GNN-based models. XplainScreen offers a user-friendly, web-based interactive platform that allows for the selection of specific GNN-based drug screening models and multiple cutting-edge Explainable AI methods. It supports both qualitative assessments (through visualization and generative text descriptions) and quantitative evaluations of these methods, utilizing drug molecules inputted in SMILES format. 

This demonstration showcases the utility of XplainScreen through a user study with pharmacological researchers focused on virtual screening tasks based on toxicity, highlighting the framework’s potential to enhance the integrity and trustworthiness of AI-driven virtual drug screening.

<img src="https://github.com/GeonHeeAhn/XplainScreen/blob/main/Images/interface.png"/> 
<br/>   
<br/>

## Installation

### Download Datafiles
Download [Tox21](https://drive.google.com/file/d/1TNn5ft59Xf_d5x5r0qOYFucbadIefpT4/view?usp=share_link) and [ClinTox](https://drive.google.com/file/d/1L-uh85sy6lsF8qhtcKcSxYjFDFysj74u/view?usp=share_link) datafiles from the link and place them in the root directory.

### Create new environment

```
conda create -n dig python=3.8
conda activate dig
```

### Install requirements.txt

```
pip install -r requirements.txt
```
### Create GPT4 api key and paste
Unfortunately, due to OPENAI's policy, uploading the api key is restricted.<br/>
Create your own api key(GPT-4 or higher model required) and paste it in create_gpt.py

### After installation to run the app

```
streamlit run streamlit_app.py
```

## Contact
If you have any questions, please feel free to contact [justamoment@ewhain.net.](mailto:﻿justamoment@ewhain.net)
<br/>
