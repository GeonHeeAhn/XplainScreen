import os
import os.path as osp
import torch
from torch.utils.data import random_split
from torch_geometric.data import download_url, extract_zip
from torch_geometric.loader import DataLoader

from dig.xgraph.dataset import MoleculeDataset
from dig.xgraph.models import GCN_3l
from dig.xgraph.utils.compatibility import compatible_state_dict
from dig.xgraph.utils.init import fix_random_seed
from dig.xgraph.method import DeepLIFT
from dig.xgraph.evaluation import XCollector

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def split_dataset(dataset, dataset_split=(0.8, 0.1, 0.1)):
    dataset_len = len(dataset)
    dataset_split = [int(dataset_len * dataset_split[0]),
                     int(dataset_len * dataset_split[1]),
                     0]
    dataset_split[2] = dataset_len - dataset_split[0] - dataset_split[1]
    train_set, val_set, test_set = random_split(dataset, dataset_split)

    return {'train': train_set, 'val': val_set, 'test': test_set}


fix_random_seed(123)
dataset = MoleculeDataset('datasets', 'Tox21')
dataset.data.x = dataset.data.x.to(torch.float32)
dataset.data.y = dataset.data.y[:, 2]  # the target 2 task.
dim_node = dataset.num_node_features
dim_edge = dataset.num_edge_features
num_targets = dataset.num_classes
num_classes = 2

splitted_dataset = split_dataset(dataset)
dataloader = DataLoader(splitted_dataset['test'], batch_size=1, shuffle=False)

def check_checkpoints(root='./'):
    if osp.exists(osp.join(root, 'checkpoints')):
        return
    url = ('https://github.com/divelab/DIG_storage/raw/main/xgraph/checkpoints.zip')
    path = download_url(url, root)
    extract_zip(path, root)
    os.unlink(path)


model = GCN_3l(model_level='graph', dim_node=dim_node, dim_hidden=300, num_classes=num_classes)
model.to(device)
check_checkpoints()
ckpt_path = osp.join('checkpoints', 'tox21', 'GCN_3l', '2', 'GCN_3l_best.ckpt')
state_dict = compatible_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])

params_dict=torch.load(ckpt_path, map_location='cpu')['state_dict']
params_dict["convs.0.lin.weight"] = params_dict["convs.0.weight"].t()
params_dict["conv1.lin.weight"] = params_dict["conv1.weight"].t()

model.load_state_dict(params_dict,strict=False)
#model.load_state_dict(state_dict)

load_result = model.load_state_dict(state_dict)
#if load_result.missing_keys == [] and load_result.unexpected_keys == []:
#    print("<All keys matched successfully>")
data = list(dataloader)[0]
out = model(data.x, data.edge_index)
print(out)

explainer = DeepLIFT(model, explain_graph=True)

# --- Set the Sparsity to 0.5 ---
sparsity = 0.5

# --- Create data collector and explanation processor ---

x_collector = XCollector(sparsity)
# x_processor = ExplanationProcessor(model=model, device=device)

for index, data in enumerate(dataloader):
    print(f'explain graph line {dataloader.dataset.indices[index] + 2}')
    data.to(device)

    if torch.isnan(data.y[0].squeeze()):
        continue

    walks, masks, related_preds = explainer(data.x, data.edge_index, sparsity=sparsity, num_classes=num_classes)

    x_collector.collect_data(masks, related_preds, data.y[0].squeeze().long().item())

    # if you only have the edge masks without related_pred, please feed sparsity controlled mask to
    # obtain the result: x_processor(data, masks, x_collector)

    if index >= 99:
        break
    
print(f'Fidelity: {x_collector.fidelity:.4f}\n'
      f'Fidelity_inv: {x_collector.fidelity_inv:.4f}\n'
      f'Sparsity: {x_collector.sparsity:.4f}')


