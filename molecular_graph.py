import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from batch_utils import batch_defining
import numpy as np


class Molecule:
    """Defines pseudo data loader for a single molecule based on its trivial name. Provided the particular molecule
    is included in our trivial name dictionary, this function generates a conformer for the respective molecule.
    The conformer corresponds to the energy optimized structure of the molecule. The function returns a pseudo-batch
    containing one single conformer of the chosen molecule"""

    def __init__(self, smiles):
        self.rdkmol = self._smiles_to_mol(smiles)
        self.n_nodes = self.rdkmol.GetNumAtoms()

    def _smiles_to_mol(self, smiles):
        # define 3d molecule structure from smiles
        rdkmol = Chem.MolFromSmiles(smiles)
        rdkmol = Chem.AddHs(rdkmol)
        AllChem.EmbedMolecule(rdkmol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(rdkmol)
        return rdkmol

    def _get_2d_embedding(self):
        AllChem.Compute2DCoords(self.rdkmol)
        # compute 2D positions
        pos = []
        n_nodes = self.rdkmol.GetNumAtoms()
        for i in range(n_nodes):
            conformer_pos = self.rdkmol.GetConformer().GetAtomPosition(i)
            pos.append([conformer_pos.x, conformer_pos.y])
        return np.array(pos)

    def _get_molecule_graph(self):
        graph = np.zeros((self.n_nodes, self.n_nodes), dtype=int)
        for bond in self.rdkmol.GetBonds():
            graph[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = 1
            graph[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = 1
        return graph

    def _get_schnet_format(self):
        pos = []
        for i in range(self.n_nodes):
            conformer_pos = self.rdkmol.GetConformer().GetAtomPosition(i)
            pos.append([conformer_pos.x, conformer_pos.y, conformer_pos.z])
        positions_3d = torch.tensor(pos)
        # get atomic numbers
        atomic_numbers = []
        for atom in self.rdkmol.GetAtoms():
            atomic_numbers.append(atom.GetAtomicNum())
        atomic_numbers = torch.tensor(atomic_numbers)
        # define sample (has the shape of batch with size 1)
        sample = batch_defining(positions_3d, atomic_numbers)
        return sample

    def load_molecule(self):
        return self._get_schnet_format(), self._get_2d_embedding(), self._get_molecule_graph()


class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in xrange(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in xrange(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x

        
def get_all_walks(L, lamb, end_id=None, node_id=None, self_loops=True):
    """
    :param node_id: The id of the node we start with.
    :param L: The length of every walk.
    :param lamb: The adjacency matrix of the graph we consider.
    """
    if L == 1:
        return [[idx, idx] for idx in range(lamb.shape[-1])]

    def get_seq_of_nodes(tree):
        node_seq = [tree.id]
        while tree.parent is not None:
            tree = tree.parent
            node_seq.append(tree.id)
        return node_seq[::-1]
    
    def get_neighbors(id, lamb):
        x = torch.zeros(lamb.shape[1],1)
        x[id,0] = 1.
        neighbors = lamb.mm(x).nonzero()[:,0]
        return [int(id) for id in neighbors]

    if node_id is None:
        # Multiple start nodes
        num_of_nodes = lamb.shape[1]
        current_nodes = [None]*num_of_nodes
        for i in range(num_of_nodes): current_nodes[i] = Tree(); current_nodes[i].id = i
    else:
        # Starting in one node
        root = Tree()
        root.id = node_id
        current_nodes = [root]

    for l in range(L-1):
        leaf_nodes = []
        for node in current_nodes:
            for neighbor in get_neighbors(node.id, lamb):
                new_node = Tree()
                new_node.id = neighbor
                node.add_child(new_node)
                leaf_nodes.append(new_node)

        current_nodes = leaf_nodes

    all_walks = []
    for node in leaf_nodes:
        if end_id is None:
            all_walks.append(get_seq_of_nodes(node))
        else:
            if node.id == end_id:
                all_walks.append(get_seq_of_nodes(node))

    # filter out walks that include self loops
    if not self_loops:
        all_walks_filtered = []
        for w in all_walks:
            if len(set(w)) == len(w):
                all_walks_filtered.append(w)
        return all_walks_filtered

    return all_walks
