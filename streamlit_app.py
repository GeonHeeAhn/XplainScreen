from create_explanations import load_model
import streamlit as st
import pickle
from torch_geometric.utils.loop import add_remaining_self_loops
from molecular_graph import Molecule
import matplotlib.pyplot as plt
from vis_utils import relevance_vis_2d_v1
from create_explanations import from_smiles
from data_from_smile import process_out
import math
import base64 
import torch as torch
import pandas as pd
import numpy as np
import uuid
import os
from PIL import Image

from rdkit import Chem, rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO


import streamlit as st
import pickle
from torch_geometric.utils.loop import add_remaining_self_loops

from molecular_graph import Molecule
from dig.xgraph.evaluation import XCollector, ExplanationProcessor

from create_gpt import explain_molecule, find_similar_smiles

def load(data):
    if data == "tox":
        with open('explanations_tox.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        return loaded_dict
    else:
        with open('explanations_clintox.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        return loaded_dict

tox_21_explanations = load(data="tox")
clintox_explanations = load(data="clintox")

def create_relevance_from_data(data, method="deeplift"):
    
    relevances_scale = []
    all_edges = add_remaining_self_loops(data.x)[0]
    if method == "deeplift":    
        try:
            exs = tox_21_explanations[data.smiles[0]]['deep_lift_edge_level_explanations']
            for id in range(all_edges.size()[1]):
                relevances_scale.append(([all_edges[0][id].item(),all_edges[1][id].item()], exs[1][id].item()-exs[0][id].item()) )
        except Exception as e:
            return []
        
    elif method == "gradcam":
        try:
            exs = tox_21_explanations[data.smiles[0]]['gradCam_edge_level_explanations']
            for id in range(all_edges.size()[1]):
                relevances_scale.append(([all_edges[0][id].item(),all_edges[1][id].item()], exs[1][id].item()-exs[0][id].item()) )
        except Exception as e:
            return []
            
    elif method=="gnnexplainer":
        try:
            exs = tox_21_explanations[data.smiles[0]]['gnnExplainer_edge_level_explanations']
            for id in range(all_edges.size()[1]):
                relevances_scale.append(([all_edges[0][id].item(),all_edges[1][id].item()], exs[1][id].item()-exs[0][id].item()) )
        except Exception as e:
            return []

    elif method == "gnnlrp":
        try:
            exs = tox_21_explanations[data.smiles[0]]['gnn_lrp_path_level_explanations']
            #print(exs)
            for id in range(exs['ids'].size()[0]): #scores
                node_list = []
                
                for idx, edge in enumerate(exs['ids'][id].tolist()):
                    if idx==0:
                        node_list.append(all_edges[0][edge].item())
                        node_list.append(all_edges[1][edge].item())
                    else:
                        node_list.append(all_edges[1][edge].item())
                relevances_scale.append((node_list, exs['score'][id][1].item()-exs['score'][id][0].item()) )
        except Exception as e:
            return []
    return relevances_scale

def create_relevance_from_explanation(data, exs, method="deeplift"):
    
    relevances_scale = []
    all_edges = add_remaining_self_loops(data.edge_index)[0]
    if method == "deeplift":    
        try:
            for id in range(all_edges.size()[1]):
                relevances_scale.append(([all_edges[0][id].item(),all_edges[1][id].item()], exs[1][id].item()-exs[0][id].item()) )
        except Exception as e:
            return []
        
    elif method == "gradcam":
        try:
            for id in range(all_edges.size()[1]):
                relevances_scale.append(([all_edges[0][id].item(),all_edges[1][id].item()], exs[1][id].item()-exs[0][id].item()) )
        except Exception as e:
            return []
            
    elif method=="gnnexplainer":
        try:
            for id in range(all_edges.size()[1]):
                relevances_scale.append(([all_edges[0][id].item(),all_edges[1][id].item()], exs[1][id].item()-exs[0][id].item()) )
        except Exception as e:
            return []

    elif method == "gnnlrp":
        try:
            for id in range(exs['ids'].size()[0]): #scores
                node_list = []
                
                for idx, edge in enumerate(exs['ids'][id].tolist()):
                    if idx==0:
                        node_list.append(all_edges[0][edge].item())
                        node_list.append(all_edges[1][edge].item())
                    else:
                        node_list.append(all_edges[1][edge].item())
                relevances_scale.append((node_list, exs['score'][id][1].item()-exs['score'][id][0].item()) )
        except Exception as e:
            return []
    return relevances_scale

def create_relevance_for_smiles(data, exs, method="deeplift"):
    
    relevances_scale = []
    all_edges = add_remaining_self_loops(data)[0]
    if method == "deeplift":    
        try:
            for id in range(all_edges.size()[1]):
                relevances_scale.append(([all_edges[0][id].item(),all_edges[1][id].item()], exs[1][id].item()-exs[0][id].item()) )
        except Exception as e:
            return []
    return relevances_scale

neg_color = (0.0, 1.0, 0.0)
pos_color = (1.0, 0.0, 0.0)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def encode_image2(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded_string}"

def plot_rdkit(smi, relevances_scale, data, label='', fmt='svg', filename='rdkit_temp_image'): #svg
    rdDepictor.SetPreferCoordGen(True)
    mol = Chem.MolFromSmiles(smi)
    mol = Draw.PrepareMolForDrawing(mol)
    
    q = Chem.MolFromSmarts(smi)
    alist = [i for i in range(mol.GetNumAtoms())]
    
    atomList = [i for i in range(mol.GetNumAtoms())]
    if isinstance(data, torch.Tensor):
        print("map: ", data)
        all_edges = add_remaining_self_loops(data)[0]
    else : 
        print("map: ", data.edge_index)
        all_edges = add_remaining_self_loops(data.edge_index)[0]
    edge_rel_map = {}
    for id in range(all_edges.size()[1]):
            edge_rel_map[(all_edges[0][id].item(),all_edges[1][id].item())]= relevances_scale[id]
    bcols = {}
    for ha1 in alist:
        for ha2 in alist:
            if ha1 > ha2:
                b = mol.GetBondBetweenAtoms(ha1, ha2)
                if b:
                    if ha2 not in bcols.keys():
                        bcols[ha2] = edge_rel_map[(ha1, ha2)][1]
                    else:
                        bcols[ha2]+= edge_rel_map[(ha1, ha2)][1]
    
    keys = bcols.keys()
    for key in keys:
        if bcols[key]>0:
            bcols[key] = [pos_color]
        else:
            bcols[key] = [neg_color]
    
    acols = {}
    
    h_rads = {}
    h_lw_mult = {}

    for idx , (walk, rel) in enumerate(relevances_scale):
        for el in walk:
            if el not in acols.keys():
                acols[el] = 0.0
            else:
                acols[el] += rel
    keys = list(acols.keys())   
    for key in keys:
        if acols[key]<0:
            new_color = neg_color[:]
            acols[key] = [neg_color]
        else:
            acols[key] = [pos_color]
    if fmt == 'svg':
        d = rdMolDraw2D.MolDraw2DSVG(500, 400)
        mode = 'w'
    elif fmt == 'png':
        d = rdMolDraw2D.MolDraw2DCairo(400, 400)
        mode = 'wb'
    else:
        print('unknown format {}'.format(fmt))
        return
    
    d.drawOptions().fillHighlights = True
    d.DrawMoleculeWithHighlights(mol, label, acols, bcols, h_rads, h_lw_mult, -1)
    d.FinishDrawing()
    with open(f"{filename}.svg", mode) as f: 
        f.write(d.GetDrawingText())        
        
    pngDraw = rdMolDraw2D.MolDraw2DCairo(400, 400)
    mode = 'wb'
    pngDraw.drawOptions().fillHighlights = True
    pngDraw.DrawMoleculeWithHighlights(mol, label, acols, bcols, h_rads, h_lw_mult, -1)
    pngDraw.FinishDrawing()
    
    with open(f"{filename}.png", mode) as f: 
        f.write(pngDraw.GetDrawingText())
    return atomList

def add_data(f, fi, s, index):
    if 'data' not in st.session_state:
        st.session_state.data = []
        
    new_entry = {
        'fidelity': f,
        'fidelity_inv': fi,
        'sparsity': s,
        'smiles': index
    }
    st.session_state.data.append(new_entry)

def split_text(text, length):
    return '\n'.join([text[i:i+length] for i in range(0, len(text), length)])


def display_images(img_urls, smiles, images_per_row=3):
    for i in range(0, len(img_urls), images_per_row):
        cols = st.columns(images_per_row)
        for j, img_url in enumerate(img_urls[i:i+images_per_row]):
            with cols[j]:
                st.image(img_url, caption=smiles[j])
                
                
#text 적히는 부분
def plot_streamlit(option,relevances_scale, actual = None, x_collector=None, data_source=None, model_type=None, draw_rdkit=True, data=None):
    if 'imgurls' not in st.session_state:
        st.session_state.imgurls = []
        
    if Vis_Style=="Custom":
        draw_rdkit=False
    if draw_rdkit:
        atom = plot_rdkit(option,relevances_scale,data)
        st.image("rdkit_temp_image.svg")
    else:
        molecule = Molecule(option)
        sample, pos_2d, graph = molecule.load_molecule()        

        fig = plt.figure(figsize=(14, 8))
        ax = plt.subplot(1, 1, 1)

        ax = relevance_vis_2d_v1(ax, relevances_scale, sample["_atomic_numbers"][0], pos_2d, graph, shrinking_factor=10)
        plt.axis('off')
        st.pyplot(fig=plt)
    st.markdown("""<div style="display:flex; justify-content: start; gap: 20px;">
                        <h3 style="font-family:sans-serif; color:#C70039; font-size: 20px;">Negative Contributions</h3> 
                        <h3 style="font-family:sans-serif; color:#2ECC71; font-size: 20px;">Positive Contribution</h3>
                    </div>""", unsafe_allow_html=True)
    st.divider()
    r_fid = round(x_collector.fidelity,2)
    r_fidInv = round(x_collector.fidelity_inv,2)
    r_spars = round(x_collector.sparsity,2)
    
    add_data(r_fid, r_fidInv, r_spars, option)

    if x_collector:
        st.write("Fidelity is :", r_fid)
        st.write("Fidelity_inv is :", r_fidInv)
        st.write("Sparsity is :", r_spars)
        
    
    data = process_out(option)[0]
    model = load_model(data=data_source, model_type=model_type)
    out = model(data.x, data.edge_index)

    prediction = ""
    if out[0][0].item()>out[0][1].item():
        if data_source=="tox":
            prediction = "Non Toxic"
        else:
            prediction = "FDA Approved"
    else:
        if data_source=="tox":
            prediction = "Toxic"
        else:
            prediction = "FDA Rejected"
    print(data_source, prediction)
    print(out[0][0].item(),out[0][1].item(), actual)
    
    st.write("Prediction is :", prediction)
    if actual:
        st.write("Truth is :", actual)
    
    df = pd.DataFrame(st.session_state.data)

    if not df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        bar_width = 0.2
        index = np.arange(len(df))
        
        fid_bar = ax.bar(index, df['fidelity'], bar_width, label='Fidelity', color='orange')
        fid_inv_bar = ax.bar(index + bar_width, df['fidelity_inv'], bar_width, label='Fidelity Inverse', color='gold')
        spars_bar = ax.bar(index + 2 * bar_width, df['sparsity'], bar_width, label='Sparsity', color='darkorange')
        
        for i, (fidelity, fidelity_inv, sparsity) in enumerate(zip(df['fidelity'], df['fidelity_inv'], df['sparsity'])):
            ax.text(i, fidelity, f'{fidelity:.2f}', ha='center', va='top',fontsize=9, fontweight='bold', color='black')
            ax.text(i + bar_width, fidelity_inv, f'{fidelity_inv:.2f}', ha='center', va='top',fontsize=9, fontweight='bold', color='black')
            ax.text(i + 2 * bar_width, sparsity, f'{sparsity:.2f}', ha='center', va='top',fontsize=9, fontweight='bold', color='black')

        #ax.set_xlabel('Tries')
        ax.set_ylabel('Value')
        ax.set_title('Comparison Studies')
        if len(df) > 4 :
            df['smiles'] = df['smiles'].apply(lambda x: split_text(x, 10))
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels(df['smiles'], fontsize=8)
        ax.legend()


        st.pyplot(fig)
     
    st.subheader("SMILES Log")   
    imgURL = encode_image("rdkit_temp_image.png") 
    imgURL2 = encode_image2("rdkit_temp_image.png")
    st.session_state.imgurls.append(imgURL2)
    display_images(st.session_state.imgurls, df['smiles'])
    
        
    st.divider()
     
    with st.spinner("Analyzing smiles..."): 
        response = explain_molecule(method, option, prediction, imgURL)
        
    
    st.write(response)

    st.markdown("""<div style="font-weight: bold">Disclaimer</div>""", unsafe_allow_html=True)
    st.write("The interpretation of toxicity based on the XAI method is a complex process that involves analyzing various molecular components and their interactions within a model. While efforts have been made to accurately attribute toxicity to specific structural features using this XAI technique, it's essential to acknowledge the potential for discrepancies between predicted toxicity and real-world outcomes. Factors such as experimental conditions, biological variability, and unforeseen interactions may influence the actual toxicity of a compound differently than predicted by the model. Therefore, the predictions provided should be considered as estimations rather than definitive assessments of toxicity. Researchers and practitioners are encouraged to exercise caution and consult domain experts when making decisions based on these predictions.")



st.markdown("""<div style="display:flex; flex-direction: column; justify-content:center; align-items: center;">
                <div style="font-weight: bold; font-size: 50px;" >🔑 XplainScreen</div>
            </div>""", unsafe_allow_html=True)

st.sidebar.title("Options")

DataSet = st.sidebar.radio(
        "Choose your Dataset",
        ["Tox21", "Clintox"],
        help="Tox21 predicts Nr-AhR probability and Clintox predicts the probability of FDA Approval",horizontal=True
)
ExplainerType = st.sidebar.radio(
        "Choose your explainer",
        ["DeepLift (Default)", "GnnLRP", "GradCam", "GNNExplainer"],
        help="At present, you can choose between 2 models (Flair or DistilBERT) to embed your text. More to come!",horizontal=True
    )
method=None
if ExplainerType=="DeepLift (Default)":
    method="deeplift"
elif ExplainerType=="GnnLRP":
    method="gnnlrp"
elif ExplainerType=="GradCam":
    method="gradcam"
elif ExplainerType=="GNNExplainer":
    method="gnnexplainer"


ModelType = st.sidebar.radio(
        "Choose your Model",
        ["GCN", "GIN"],
        help="GCN: Graph Convolutional Neural Network, GIN: Graph Isomorphic Network",horizontal=True
)

Vis_Style = st.sidebar.radio(
        "Choose your Visualization Style",
        ["Rdkit", "Custom"],
        help="Use cusom for GNNLrp or rdkit otherwise",horizontal=True
)

#smiles 선택하는 부분
#option값이 선택된 스마일스 -> 버튼이 클릭될 때 할당된 option값을 지피티 함수로 넘기기
#지피티 함수 def 해두기
option=None
option = st.sidebar.selectbox('Select from Tox21 Database',tuple(list(tox_21_explanations.keys())[:60]))
if st.sidebar.button("process Selected tox"):
    data = process_out(option)
    exs , x_collector = from_smiles(option, data_in="tox",model_type=ModelType,method=method)
    relevances_scale = create_relevance_from_explanation(data[0],exs ,method = method)

    if len(relevances_scale)!=0:
        actual="Non Toxic"
        if math.isnan(tox_21_explanations[option]['data'].y[0].item()):
            actual == None
        else:
            if int(tox_21_explanations[option]['data'].y[0].item())==1:
                actual = "Toxic"
        plot_streamlit(option, relevances_scale, actual=actual, x_collector = x_collector,data_source="tox",model_type=ModelType, data = data[0])
    else:
        st.markdown('<p style="font-family:sans-serif; color:red; font-size: 20px;">Explanation Not Available</p>', unsafe_allow_html=True)

user_input = st.sidebar.text_input("Provide Custom Smile: ", option)
if st.sidebar.button("process tox"):
    print("relavence from explanation called from Tox")
    data = process_out(user_input)
    if len(data)!=0:
        exs , x_collector = from_smiles(user_input, data_in="tox",model_type=ModelType,method=method)
        relevances_scale = create_relevance_from_explanation(data[0],exs ,method = method)
        if len(relevances_scale)!=0:
            actual="Non Toxic"
            if math.isnan(tox_21_explanations[option]['data'].y[0].item()):
                actual == None
            else:
                if int(tox_21_explanations[option]['data'].y[0].item())==1:
                    actual = "Non Toxic"
            plot_streamlit(user_input, relevances_scale, actual=actual, x_collector = x_collector, data_source="tox",model_type=ModelType, data=data[0])
    else:
        st.markdown('<p style="font-family:sans-serif; color:red; font-size: 50px;">Invalid Smile String</p>', unsafe_allow_html=True)

# clintox
option2 = st.sidebar.selectbox('Select from Clintox Database',tuple(list(clintox_explanations.keys())[:60]))

if st.sidebar.button("process Selected Clintox"):
    data = process_out(option2)
    exs , x_collector = from_smiles(option2,data_in="clintox",model_type=ModelType, method=method)
    relevances_scale = create_relevance_from_explanation(data[0],exs ,method = method)    
    if len(relevances_scale)!=0:
        actual="FDA Rejected"
        if math.isnan(clintox_explanations[option2]['data'].y[0].item()):
            actual == None
        else:
            if int(clintox_explanations[option2]['data'].y[0].item())==1:
                actual = "FDA Approved"
        plot_streamlit(option2, relevances_scale,actual=actual,x_collector = x_collector, data_source="clintox",model_type=ModelType, data=data[0])
    else:
        st.markdown('<p style="font-family:sans-serif; color:red; font-size: 20px;">Explanation Not Available</p>', unsafe_allow_html=True)

user_input = st.sidebar.text_input("Provide Custom Smile: ", option2)
if st.sidebar.button("process clintox"):
    data = process_out(user_input)
    if len(data)!=0:
        exs, x_collector = from_smiles(user_input,data_in="clintox",model_type=ModelType, method=method)
        relevances_scale = create_relevance_from_explanation(data[0],exs ,method = method)
        if len(relevances_scale)!=0:
            actual="FDA Rejected"
            if math.isnan(clintox_explanations[option2]['data'].y[0].item()):
                actual == None
            else:
                if int(clintox_explanations[option2]['data'].y[0].item())==1:
                    actual = "FDA Approved"
            plot_streamlit(user_input, relevances_scale,actual=actual,x_collector = x_collector, data_source="tox",model_type=ModelType, data=data[0])
    else:
        st.markdown('<p style="font-family:sans-serif; color:red; font-size: 50px;">Invalid Smile String</p>', unsafe_allow_html=True)

