# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from turtle import width
from dash import Dash, html, dcc,Input, Output
import dash_bootstrap_components as dbc
import pickle
from torch_geometric.utils.loop import add_remaining_self_loops
import chart_studio.plotly as py
from molecular_graph import Molecule

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vis_utils import relevance_vis_2d_v1
from create_explanations import from_smiles
from data_from_smile import process_out
from create_explanations import load_model
import plotly.tools as tls
import math

def load():
    with open('explanations_tox.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

tox_21_explanations = load()

tox21_chems = list(tox_21_explanations.keys())[:10]

app = Dash(external_stylesheets=[dbc.themes.MINTY])

app.layout = dbc.Container([
                dbc.Row([
                    dbc.Col([
                            html.H2('Choose Model'),
                            dcc.RadioItems(["DeepLift", "GnnLRP", "GradCam", "GNNExplainer"], value='DeepLift',inline=False, id="selected_model"),
                            html.H2('Choose Molecule'),
                            dcc.Dropdown(tox21_chems, tox21_chems[0], id='tox21_dropdown'),
                            html.Div(id='dd-output-container')
                    ],width=3),
                    dbc.Col([
                        # html.Div(id='display')
                        dbc.Spinner(
                            dcc.Graph(id="display"),
                            color="primary",
                        )
                    ],
                    width=True,
                ),
                ]),
],fluid=True)


@app.callback(
    Output('dd-output-container', 'children'),
    Input('tox21_dropdown', 'value')
)
def update_output(value):
    return f'You have selected {value}'



def create_relevance_from_explanation(data, exs, method="deeplift"):
    #print(method, data)
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
            #print("comes to gnnlrp")
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


def create_relevance_from_data(data, method="deeplift"):
    print("Inside relevence from data Function, ", data, method)
    #print(method, data)
    relevances_scale = []
    all_edges = add_remaining_self_loops(data.edge_index)[0]
    print(method)
    if method == "deeplift":    
        print("Inside Deep lift")
        try:
            print("Inside Try")
            exs = tox_21_explanations[data.smiles[0]]['deep_lift_edge_level_explanations']

            for id in range(all_edges.size()[1]):
                relevances_scale.append(([all_edges[0][id].item(),all_edges[1][id].item()], exs[1][id].item()-exs[0][id].item()) )
        except Exception as e:
            print("error ", e)
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


def plot_streamlit(option,relevances_scale, actual = None):

    data = process_out(option)[0]
    model = load_model()
    out = model(data.x, data.edge_index)
    prediction = ""
    if out[0][0].item()>out[0][1].item():
        prediction = "Non Toxic"
    else:
        prediction = "Toxic"
    
    print(out[0][0].item(),out[0][1].item(), actual)
    molecule = Molecule(option)
    sample, pos_2d, graph = molecule.load_molecule()
    mpl_fig = plt.figure()
    #print(pos_2d)
    fig = plt.figure(figsize=(14, 8))
    ax = plt.subplot(1, 1, 1)
    #print(sample["_atomic_numbers"][0])
    ax = relevance_vis_2d_v1(ax, relevances_scale, sample["_atomic_numbers"][0], pos_2d, graph, shrinking_factor=10)
    plt.axis('off')

    plotly_fig = tls.mpl_to_plotly(fig) ## convert 
    return plotly_fig
    #unique_url = py.plot_mpl(mpl_fig, filename="test")
    #plt.show()
    #st.pyplot(fig=plt)
    #st.markdown('<p style="font-family:sans-serif; color:#66ff66; font-size: 20px;">Negative Contribution</p> ', unsafe_allow_html=True)
    #st.markdown('<p style="font-family:sans-serif; color:#ff66ff; font-size: 20px;">Positive Contribution</p>', unsafe_allow_html=True)
        
    #if actual:
    #    st.write("Prediction is :", prediction)
    #    st.write("Truth is :", actual)
        

@app.callback(
    Output(component_id='display', component_property='figure'),
    Input('tox21_dropdown', 'value'),Input('selected_model', 'value')
)
def update_output(tox21_dropdown,selected_model):
    relevances_scale = create_relevance_from_data(tox_21_explanations[tox21_dropdown]['data'], method = selected_model.lower())
    if len(relevances_scale)!=0:
        print("comes3")
        actual="Non Toxic"
        if math.isnan(tox_21_explanations[tox21_dropdown]['data'].y[0].item()):
            actual == None
        else:
            if int(tox_21_explanations[tox21_dropdown]['data'].y[0].item())==1:
                actual = "Toxic"
        print("comes before plot streamlit")
        py_fig = plot_streamlit(tox21_dropdown, relevances_scale, actual=actual)
        print("comes")
        print(py_fig)
        return py_fig
    print("comes4")
    return None

if __name__ == '__main__':
    app.run_server(debug=True)
