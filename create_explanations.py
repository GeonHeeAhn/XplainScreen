from dig.xgraph.dataset import MoleculeDataset
from dig.xgraph.models import GCN_3l, GIN_3l
import torch
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.data.dataset import files_exist
import os.path as osp
import os
from tqdm import tqdm
import traceback

from dig.xgraph.method import DeepLIFT
from dig.xgraph.method import GNN_LRP
from dig.xgraph.method import GNNExplainer
from dig.xgraph.method import GradCAM
from dig.xgraph.utils.compatibility import compatible_state_dict
from dig.xgraph.evaluation import XCollector, ExplanationProcessor

from torch_geometric.utils.loop import add_remaining_self_loops
import pickle

from data_from_smile import process_out 

FILE_NAME="explanations_clintox.pkl"
data_name = "clintox"
model_name="gcn"

def save(dictionary):
    with open(FILE_NAME, 'wb') as f:
        pickle.dump(dictionary, f)

def load():
    with open(FILE_NAME, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

def check_checkpoints(root='./'):
    if osp.exists(osp.join(root, 'checkpoints')):
        return
    url = ('https://github.com/divelab/DIG_storage/raw/main/xgraph/checkpoints.zip')
    path = download_url(url, root)
    print("checkpoint 다운 완료")
    extract_zip(path, root)
    os.unlink(path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def split_dataset(dataset, dataset_split=[0.8, 0.1, 0.1]):
    dataset_len = len(dataset)
    dataset_split = [int(dataset_len * dataset_split[0]),
                     int(dataset_len * dataset_split[1]),
                     0]
    dataset_split[2] = dataset_len - dataset_split[0] - dataset_split[1]
    train_set, val_set, test_set = \
        random_split(dataset, dataset_split)

    return {'train': train_set, 'val': val_set, 'test': test_set}

num_classes = 2

def create_dataset(data="tox"):
    if data=="tox":
        dataset = MoleculeDataset('datasets', 'Tox21')
        dataset.data.x = dataset.data.x.to(torch.float32)
        dataset.data.y = dataset.data.y[:, 2] # the target 2 task.
        return dataset
    else:
        dataset = MoleculeDataset('datasets', 'clintox')
        dataset.data.x = dataset.data.x.to(torch.float32)
        dataset.data.y = dataset.data.y[:, 0] 
        return dataset

dataset = create_dataset(data=data_name)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

print(dataset)



def load_model(data, model_type):
    print("load model 시작")
    if data == "tox":
        if model_type=="GCN":
            model = GCN_3l(model_level='graph', dim_node=dataset.num_node_features, dim_hidden=300, num_classes=num_classes)
            model.to(device)
            check_checkpoints()    
            ckpt_path = osp.join('checkpoints', 'tox21', 'GCN_3l', '2', 'GCN_3l_best.ckpt')
            state_dict = compatible_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'])
            model.load_state_dict(state_dict)
            #model.load_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'])
            return model
        else:
            model = GIN_3l(model_level='graph', dim_node=dataset.num_node_features, dim_hidden=300, num_classes=num_classes)
            model.to(device)
            check_checkpoints()    
            ckpt_path = osp.join('checkpoints', 'tox21', 'GIN_3l', '2', 'GIN_3l_best.ckpt')
            #state_dict = compatible_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'])
            #model.load_state_dict(state_dict)

            model.load_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'])
            return model
        
    else:
        if model_type=="GIN":
            model = GIN_3l(model_level='graph', dim_node=dataset.num_node_features, dim_hidden=300, num_classes=num_classes)
            model.to(device)
            check_checkpoints()
            ckpt_path = osp.join('checkpoints', 'clintox', 'GIN_3l', '0', 'GIN_3l_best.ckpt')
            #state_dict = compatible_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'])
            #model.load_state_dict(state_dict)

            model.load_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'])
            return model
        else:
            model = GCN_3l(model_level='graph', dim_node=dataset.num_node_features, dim_hidden=300, num_classes=num_classes)
            model.to(device)
            check_checkpoints()
            ckpt_path = osp.join('checkpoints', 'clintox', 'GCN_3l', '0', 'GCN_3l_best.ckpt')
            state_dict = compatible_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'])
            model.load_state_dict(state_dict)

            #model.load_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'])
            return model



#explanation_dic = load()

explanation_dic = {}

#splitted_dataset = split_dataset(dataset)


def populate_dic():
    for index, data in enumerate(dataloader):
        explanation_dic[data.smiles[0]] = {}
        explanation_dic[data.smiles[0]]['index']=index
        explanation_dic[data.smiles[0]]['data']=data
    
populate_dic()
#save(explanation_dic)

def from_smiles(smile, data_in,model_type, method="deeplift"):
    x_collector = XCollector(sparsity=0.5)
    dt = process_out(smile)
    
    #dt = process_out("Cc1ncc([N+](=O)[O-])n1CC(C)O")
    #data, slices = collate(dt)
    #print(data)
    dataloader = DataLoader(dt, batch_size=1, shuffle=False)

    if method == "deeplift":
        deep_lift_explainer = DeepLIFT(model=load_model(data=data_in,model_type=model_type), explain_graph=True)
        sparsity = 0.5

        for data in tqdm(dataloader):
            #print(f'explain graph line {dataloader.dataset.indices[index] + 2}')
            data.to(device)
            if torch.isnan(data.y[0].squeeze()):
                continue
            try:
                walks, masks, related_preds = \
                    deep_lift_explainer(data.x, data.edge_index, sparsity=sparsity, num_classes=num_classes)
                x_collector.collect_data(masks, related_preds, data.y[0].squeeze().long().item())
                return walks, x_collector
            except Exception as e:
                return []

    elif method=="gnnlrp":
        lrp_explainer = GNN_LRP(model=load_model(data=data_in,model_type=model_type), explain_graph=True)
        lrp_explainer.num_layers = 3
        cnt = 0
        sparsity = 0.5

        for data in tqdm(dataloader):
            #print(f'explain graph line {dataloader.dataset.indices[index] + 2}')
            data.to(device)

            if torch.isnan(data.y[0].squeeze()):
                continue
            try:
                walks, masks, related_preds = \
                    lrp_explainer(data.x, data.edge_index, sparsity=sparsity, num_classes=num_classes)
                x_collector.collect_data(masks, related_preds, data.y[0].squeeze().long().item())
                return walks, x_collector
            except Exception as e:
                return []

    elif method=="gnnexplainer":
        gnn_explainer = GNNExplainer(model=load_model(data=data_in,model_type=model_type), epochs=100, lr=0.01, explain_graph=True)

        sparsity = 0.5

        for data in tqdm(dataloader):
            #print(f'explain graph line {dataloader.dataset.indices[index] + 2}')
            data.to(device)

            if torch.isnan(data.y[0].squeeze()):
                continue
            try:
                edge_masks, hard_edge_masks, related_preds = \
                    gnn_explainer(data.x, data.edge_index, sparsity=sparsity, num_classes=num_classes)
                x_collector.collect_data(hard_edge_masks, related_preds, data.y[0].squeeze().long().item())
                return edge_masks, x_collector
            except Exception as e:
                return []
    elif method=="gradcam":
        grad_cam_explainer = GradCAM(model=load_model(data=data_in,model_type=model_type), explain_graph=True)
        sparsity = 0.5
        for data in tqdm(dataloader):
            #print(f'explain graph line {dataloader.dataset.indices[index] + 2}')
            data.to(device)
            if torch.isnan(data.y[0].squeeze()):
                continue
            try:
                walks, masks, related_preds = \
                    grad_cam_explainer(data.x, data.edge_index, sparsity=sparsity, num_classes=num_classes)
                x_collector.collect_data(masks, related_preds, data.y[0].squeeze().long().item())
                return walks, x_collector
                
            except Exception as e:
                return []
    return []




def deep_lift_explanation(dataloader):
    deep_lift_explainer = DeepLIFT(model=load_model(data=data_name,model_type=model_name), explain_graph=True)
    sparsity = 0.5
    #for param in model.parameters():
    #    print(param.requires_grad)
    
    for data in tqdm(dataloader):
        #print(f'explain graph line {dataloader.dataset.indices[index] + 2}')
        data.to(device)

        if torch.isnan(data.y[0].squeeze()):
            continue
        try:
            walks, masks, related_preds = \
                deep_lift_explainer(data.x, data.edge_index, sparsity=sparsity, num_classes=num_classes)
            explanation_dic[data.smiles[0]]['deep_lift_edge_level_explanations'] = walks
            #print("comes")
            #return
            #print(explanation_dic[data.smiles[0]])
        except:
            traceback.print_exc()
            deep_lift_explainer = DeepLIFT(model=load_model(data=data_name,model_type=model_name), explain_graph=True)
            continue
        #break



def gnn_lrp_explanation(dataloader, path_length=3):
    lrp_explainer = GNN_LRP(model=load_model(data=data_name,model_type=model_name), explain_graph=True)
    lrp_explainer.num_layers = path_length
    cnt = 0
    sparsity = 0.5

    for data in tqdm(dataloader):
        #print(f'explain graph line {dataloader.dataset.indices[index] + 2}')
        data.to(device)

        if torch.isnan(data.y[0].squeeze()):
            continue
        try:
            walks, masks, related_preds = \
                lrp_explainer(data.x, data.edge_index, sparsity=sparsity, num_classes=num_classes)
            explanation_dic[data.smiles[0]]['gnn_lrp_path_level_explanations'] = walks
            #print(walks)
        except Exception as e:
            continue

def gnnExplainer_explanation(dataloader):
    
    gnn_explainer = GNNExplainer(model=load_model(data=data_name,model_type=model_name), epochs=100, lr=0.01, explain_graph=True)

    sparsity = 0.5

    for data in tqdm(dataloader):
        #print(f'explain graph line {dataloader.dataset.indices[index] + 2}')
        data.to(device)

        if torch.isnan(data.y[0].squeeze()):
            continue
        try:
            edge_masks, hard_edge_masks, related_preds = \
                gnn_explainer(data.x, data.edge_index, sparsity=sparsity, num_classes=num_classes)
            #print(edge_masks)
            explanation_dic[data.smiles[0]]['gnnExplainer_edge_level_explanations'] = edge_masks
        except Exception as e:
            print("gnn_explainer", e)
            continue
        #break


def gradCam_explanation(dataloader):
    grad_cam_explainer = GradCAM(model=load_model(data=data_name,model_type=model_name), explain_graph=True)
    sparsity = 0.5
    for data in tqdm(dataloader):
        #print(f'explain graph line {dataloader.dataset.indices[index] + 2}')
        data.to(device)
        if torch.isnan(data.y[0].squeeze()):
            continue
        try:
            walks, masks, related_preds = \
                grad_cam_explainer(data.x, data.edge_index, sparsity=sparsity, num_classes=num_classes)
            explanation_dic[data.smiles[0]]['gradCam_edge_level_explanations'] = walks
            #print(explanation_dic[data.smiles[0]])
            #print("comes")
        except Exception as e:
            print("grad_cam", str(e))
            continue

    
def verify_accuracy(model,dataloader):
    import math
    all_data = list(dataloader)
    correct, incorrect=0,0
    for dt in tqdm(all_data):
        single_data = dt.to(device)
        out = model(single_data.x, single_data.edge_index)
        pred = 1
        if math.isnan(single_data.y.item()):
            continue
        if out[0][0]>out[0][1]:
            pred = 0
        if int(single_data.y.item())== pred:
            correct+=1
        else:
            incorrect+=1
        
    print(correct+incorrect,correct/(correct+incorrect))


#save(explanation_dic)
#deep_lift_explanation(dataloader=dataloader)
#print(explanation_dic)
#save(explanation_dic)
#gradCam_explanation(dataloader=dataloader)
#print(explanation_dic)

#save(explanation_dic)
#gnnExplainer_explanation(dataloader=dataloader)
#print(explanation_dic)

#save(explanation_dic)
#gnn_lrp_explanation(dataloader=dataloader, path_length=3)
#print(explanation_dic)

#save(explanation_dic)
#for key in list(explanation_dic.keys())[:2]:
#    print(explanation_dic[key])
#print(explanation_dic[list(explanation_dic.keys())[0]])

#save(explanation_dic)