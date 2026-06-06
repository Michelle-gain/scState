import anndata as ad
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from tqdm import tqdm
import math
import scanpy as sc
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import palantir


def subgraph(graph, seed, n_neighbors, node_sele_prob):
    total_matrix_size = 1 + np.cumprod(n_neighbors).sum()  # Number of nodes in the subgraph
    picked_nodes = {seed}  # One node in the batch
    last_layer_nodes = {seed}

    # Number of nodes selected in each layer. Initially, only the seed node is selected.
    to_pick = 1
    for n_neighbors_current in n_neighbors:  # Current layer neighbors
        to_pick = to_pick * n_neighbors_current
        neighbors = graph[list(last_layer_nodes), :].nonzero()[1]  # Find neighbors of last_layer_nodes

        neighbors_prob = node_sele_prob[list(neighbors)]
        neighbors = list(set(neighbors))  # Make all nodes from the last layer part of the neighbors set
        n_neigbors_real = min(
            to_pick,
            len(neighbors))  # Handle the case where the required number of neighbors is less than the actual number of neighbors
        if len(neighbors_prob) == 0:
            continue
        last_layer_nodes = set(
            np.random.choice(neighbors, n_neigbors_real, replace=False,
                             p=softmax(neighbors_prob)))  # Select non-repeated nodes from neighbors
        picked_nodes |= last_layer_nodes  # Update picked_nodes as last_layer_nodes ∪ picked_nodes
    indices = list(sorted(picked_nodes - {seed}))
    return indices


def batch_select_whole(RNA_matrix, ATAC_matrix, neighbor=[20], cell_size=30):
    print('We are currently in the process of partitioning the data into batches. Kindly wait for a moment, please.')
    node_ids = np.random.choice(RNA_matrix.shape[1], size=RNA_matrix.shape[1], replace=False)
    n_batch = math.ceil(node_ids.shape[0] / cell_size)
    indices_ss = []

    RNA_matrix1 = RNA_matrix
    dic = {}
    for i in tqdm(range(n_batch)):
        gene_indices_all = []
        peak_indices_all = []
        if i < n_batch:
            for index, node in enumerate(node_ids[i * cell_size:(i + 1) * cell_size]):
                rna_ = RNA_matrix1[:, node].todense()
                rna_[rna_ < 5] = 0
                gene_indices = subgraph(RNA_matrix.transpose(), node, neighbor, np.squeeze(np.array(np.log(rna_ + 1))))
                peak_indices = subgraph(ATAC_matrix.transpose(), node, neighbor,
                                        np.squeeze(np.array(np.log(ATAC_matrix[:, node].todense() + 1))))
                dic[node] = {'g': gene_indices, 'p': peak_indices}
                gene_indices_all = gene_indices_all + gene_indices
                peak_indices_all = peak_indices_all + peak_indices
            node_indices_all = node_ids[i * cell_size:(i + 1) * cell_size]
        else:
            for index, node in enumerate(node_ids[i * cell_size:]):
                rna_ = RNA_matrix1[:, node].todense()
                rna_[rna_ < 5] = 0
                gene_indices = subgraph(RNA_matrix.transpose(), node, neighbor,
                                        np.squeeze(np.array(np.log(rna_[:, node].todense() + 1))))
                peak_indices = subgraph(ATAC_matrix.transpose(), node, neighbor,
                                        np.squeeze(np.array(np.log(ATAC_matrix[:, node].todense() + 1))))
                dic[node] = {'g': gene_indices, 'p': peak_indices}
                gene_indices_all = gene_indices_all + gene_indices
                peak_indices_all = peak_indices_all + peak_indices
            node_indices_all = node_ids[i * cell_size:]

        gene_indices_all = list(set(gene_indices_all))
        peak_indices_all = list(set(peak_indices_all))
        h = dict()
        h['gene_index'] = gene_indices_all
        h['peak_index'] = peak_indices_all
        h['cell_index'] = node_indices_all
        indices_ss.append(h)
    return indices_ss, node_ids, dic


def softmax(x):
    return (np.exp(x) / np.exp(x).sum())


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """Constructor for LabelSmoothing module.
        :param smoothing: Label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def initial_clustering(matrix, resolution=0.2, n_neighbors=15, batch_remove=False, custom_n_neighbors=None, n_pcs=40, custom_resolution=None, use_rep=None):
    if batch_remove:
        print('\tUsing batch-free matrix for pre-clustering.')
        adata = sc.AnnData(matrix)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        sc.tl.leiden(adata, resolution=resolution)
    else:
        print('\tUsing original count matrix for pre-clustering.')
        adata = ad.AnnData(matrix.transpose(), dtype='int32')
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        sc.tl.leiden(adata, resolution=resolution)

    return adata.obs['leiden']


def purity_score(y_true, y_pred):
    """Purity score

    Args:
        y_true (np.ndarray): n*1 matrix, true labels
        y_pred (np.ndarray): n*1 matrix, predicted clusters

    Returns:
        float: Purity score
    """
    # Create a matrix to store the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)

    # Sort the labels
    # Some labels might be missing, e.g., a set {0,2} where 1 is missing
    # First, find the unique labels and then map them to an ordered set
    # E.g., {0,2} should be mapped to {0,1}
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    y_true = np.array(y_true, dtype='int64')

    # Update the unique labels
    labels = np.unique(y_true)

    # Set the number of bins to n_classes + 2 so that we can compute the actual
    # class occurrences between two consecutive bins
    # The larger bin is excluded: [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        # Find the most frequent label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    y_true = np.array(y_true, dtype='int8')
    y_voted_labels = np.array(y_voted_labels, dtype='int8')
    return accuracy_score(y_true, y_voted_labels), y_true


def Entropy(pred_label, true_label):
    e = 0
    for k in set(pred_label):
        en = 0
        pred_k = Counter(pred_label)[k]
        index_pred_k = pred_label == k
        for j in set(true_label):
            true_j = Counter(true_label)[j]
            intersection_kj = (true_label[index_pred_k] == j).sum()
            p = np.array(intersection_kj) / np.array(pred_k)
            if p != 0:
                en += np.log(p) * p
        e = e + en * pred_k / true_label.shape[0]
    return abs(e)


def run_palantir_pipeline(
    adata,
    ROI_genes,
    k=50
):
    """
    Run Palantir trajectory inference pipeline.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object (cells × genes).
        
    ROI_genes : list or tuple
        Marker genes used to select the root (early) cell.

    k : int, optional
        Number of nearest neighbors used to construct the diffusion graph.
        
    Returns
    -------
    adata : 
        Processed AnnData with:
        - palantir_pseudotime
        - palantir_entropy
    """
    
    adata = adata.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    # sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
    sc.pp.highly_variable_genes(adata)
    adata = adata[:, adata.var['highly_variable']].copy()
    sc.pp.scale(adata)
    sc.tl.pca(adata)

    dm_res = palantir.utils.run_diffusion_maps(pd.DataFrame(adata.obsm['X_pca']), knn=k)

    ms_data = palantir.utils.determine_multiscale_space(dm_res)
    ms_data.index = adata.obs_names

    available_genes = [g for g in ROI_genes if g in adata.var_names]

    if len(available_genes) == 0:
        raise ValueError("None of the ROI_genes are found in adata.var_names")

    expr = adata[:, available_genes].X
    if not isinstance(expr, np.ndarray):
        expr = expr.toarray()

    avg_expr = expr.mean(axis=1)
    adata.obs['marker'] = avg_expr
    start_cell = adata.obs['marker'].idxmax()
    pr_res = palantir.core.run_palantir(
        ms_data,
        early_cell=start_cell
    )
    adata.obs['palantir_pseudotime'] = pr_res.pseudotime.values
    adata.obs['palantir_entropy'] = pr_res.entropy.values

    return adata

# Adaptively determine the confidence threshold to ensure there are sufficient candidate samples
def adaptive_confidence_threshold(proba_matrix, target_samples_ratio=0.3, min_percentile=20):
    confidence_scores = np.max(proba_matrix, axis=1)
    threshold = np.percentile(confidence_scores, 100 - target_samples_ratio * 100)
    dynamic_min_threshold = np.percentile(confidence_scores, min_percentile)
    return max(threshold, dynamic_min_threshold)

# Diversity Sampling: Using the k-means++ approach to select the most diverse samples
def diverse_sampling(features, high_indices, n_samples, prototype_indices_local=None):
    selected = []
    if prototype_indices_local is not None: 
        selected.append(prototype_indices_local)
        if prototype_indices_local in high_indices:
            high_indices = high_indices[high_indices != prototype_indices_local]
    while len(selected) < n_samples and len(high_indices) != 0:
        distances_to_selected = []
        best_dist = -1
        for i in high_indices:
            if len(selected) == 0:
                min_dist = 1e9
            else:
                min_dist = min(np.linalg.norm(features[i] - features[j]) for j in selected)
            if min_dist > best_dist:
                best_dist = min_dist
                best_indice = i
        selected.append(best_indice)
        high_indices = high_indices[high_indices != best_indice]
    return selected

# High-confidence sampling
def confidence_filtered_diverse_sampling(features, indices, proba_matrix, n_samples, 
                                       confidence_threshold=0.7, prototype_indices_local=None):
    confidence_scores = np.max(proba_matrix, axis=1)
    high_confidence_mask = confidence_scores >= confidence_threshold
    if np.sum(high_confidence_mask) == 0:
        confidence_threshold = np.percentile(confidence_scores, 50)  # 取中位数
        high_confidence_mask = confidence_scores >= confidence_threshold
    high_conf_indices_local = indices[high_confidence_mask]
    
    if len(high_conf_indices_local) <= n_samples:
        return high_conf_indices_local
    else:
        return diverse_sampling(features, high_conf_indices_local, n_samples, prototype_indices_local=prototype_indices_local)

def stage2_initial_clustering(GSVA_matrix, Node_Ids, is_stem_mask, device, 
                             n_samples_per_class=2):
    GSVA_selected = np.transpose(GSVA_matrix)
    GSVA_stem_like = GSVA_selected[is_stem_mask].toarray()
    gmm = GaussianMixture(n_components=2, covariance_type='full')
    gmm.fit(GSVA_stem_like)
    stem_pseudo_labels = gmm.predict(GSVA_stem_like)
    unique, counts = np.unique(stem_pseudo_labels, return_counts=True)
    count_dict = dict(zip(unique, counts))
    total = len(stem_pseudo_labels)
    gmm_ratio = [v / total for _, v in count_dict.items()]
    two_centers = gmm.means_
    proba_matrix = gmm.predict_proba(GSVA_stem_like)
    stem_labels_torch = torch.from_numpy(stem_pseudo_labels).to(device)
    stem_labels_torch[stem_labels_torch == 0] = 105 
    stem_labels_torch[stem_labels_torch == 1] = 106 
    stem_center_indices_local, _ = pairwise_distances_argmin_min(two_centers, GSVA_stem_like)
    prototype_indices = [is_stem_mask[i] for i in stem_center_indices_local]
    labeled_samples_indices = []
    labeled_samples_labels = []
    for class_id in [0, 1]:
        class_mask = (stem_pseudo_labels == class_id)
        class_indices_local = np.where(class_mask)[0]
        if len(class_indices_local) <= n_samples_per_class:
            selected_indices_local = class_indices_local
        else:
            confidence_threshold = adaptive_confidence_threshold(
                proba_matrix[class_mask], target_samples_ratio=0.4
            )
            selected_indices_local = confidence_filtered_diverse_sampling(
                GSVA_stem_like,
                class_indices_local,
                proba_matrix[class_mask],
                n_samples_per_class,
                confidence_threshold,
                stem_center_indices_local[class_id]
            ) 
        proto_local_idx = stem_center_indices_local[class_id] 
        if proto_local_idx not in selected_indices_local:
            selected_indices_local = np.append(selected_indices_local, proto_local_idx)

        original_indices = [is_stem_mask[i] for i in selected_indices_local] 
        labeled_samples_indices.extend(original_indices) 
        target_label = 105 if class_id == 0 else 106
        labeled_samples_labels.extend([target_label] * len(selected_indices_local)) 
    labeled_target_data = {
        'indices': labeled_samples_indices,
        'labels': torch.tensor(labeled_samples_labels, device=device)
    }

    return prototype_indices, labeled_target_data, stem_labels_torch, gmm_ratio

from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min

import torch
import torch.nn as nn
import torch.nn.functional as F
from conv import *
from utils import *
import pandas as pd
from scipy.stats import mannwhitneyu
import seaborn as sns

class ProtoClusteringLoss(nn.Module):
    def __init__(self, device, prototypes, labeled_target_data, stem_pseudo_labels,
                 gmm_ratio=[0.5,0.5],
                 target_distance=6.0,      
                 min_inter_distance=15.0,
                 balance_strength=8.0,
                 T_start=3.0, T_end=0.1, anneal_epochs=200):
        super().__init__()
        self.device = device
        self.proto_A = nn.Parameter(prototypes[0])
        self.proto_B = nn.Parameter(prototypes[1])
        self.labeled_target_data = labeled_target_data
        self.stem_pseudo_labels = stem_pseudo_labels
        self.register_buffer('stem_center', (prototypes[0]+prototypes[1])/2)
        
        self.target_distance = target_distance
        self.balance_strength = balance_strength 
        self.T_start = T_start
        self.T_end = T_end
        self.anneal_epochs = anneal_epochs
        self.min_inter_distance = min_inter_distance

        self.target_ratio = torch.tensor(gmm_ratio, device=self.device)
        self.min_class_ratio = 0.05  
        self.ratio_tolerance = 0.15 

        self.labsm = 0.1
        self.LabSm = LabelSmoothing(self.labsm)

    def get_temp(self, epoch):
        if epoch >= self.anneal_epochs:
            return self.T_end
        ratio = min(1.0, epoch / self.anneal_epochs)
        return self.T_end + 0.5 * (self.T_start - self.T_end) * (1 + math.cos(math.pi * ratio))

    def adaptive_ratio_loss(self, current_probs):
        current_ratio = current_probs.mean(dim=0)
        pred_classes = torch.argmax(current_probs, dim=1)   # shape (N,)
        
        num_class_0 = (pred_classes == 0).sum().item()
        num_class_1 = (pred_classes == 1).sum().item()

        ratio_loss = 0.0
        for i in range(2):
            target_ratio_i = self.target_ratio[i].item()
            target_ratio_i = torch.tensor(target_ratio_i, device=self.device)
            current_ratio_i = current_ratio[i]
            lower_bound = target_ratio_i 
            upper_bound =  1 - target_ratio_i
            if current_ratio_i < lower_bound:
                ratio_loss += F.relu(lower_bound - current_ratio_i).pow(2)
            elif current_ratio_i > upper_bound:
                ratio_loss += F.relu(current_ratio_i - upper_bound).pow(2)
        return ratio_loss
        
    def ce_margin_loss(self, embeddings, cell_indices, proto_A, proto_B, labeled_data, margin=1.0, lambda_margin=0.5):
        global_labeled_indices = labeled_data['indices'] 
        global_labeled_labels = labeled_data['labels']  
    
        index_map = {int(idx): i for i, idx in enumerate(cell_indices.tolist())}
    
        valid_group_positions = []
        valid_labels = []
        for g_idx, g_label in zip(global_labeled_indices, global_labeled_labels):
            g_idx = int(g_idx)
            if g_idx in index_map:
                valid_group_positions.append(index_map[g_idx])
                valid_labels.append(g_label)
    
        if len(valid_group_positions) == 0:
            return torch.tensor(0.0, device=self.device)
    
        valid_group_positions = torch.tensor(valid_group_positions, device=self.device)
        valid_labels = torch.tensor(valid_labels, device=self.device)
    
        labels = (valid_labels == 106).long()
    
        labeled_features = embeddings[valid_group_positions]
    
        dist_A = torch.norm(labeled_features - proto_A, dim=1)
        dist_B = torch.norm(labeled_features - proto_B, dim=1)
    
        logits = torch.stack([-dist_A, -dist_B], dim=1)
        ce_loss = F.cross_entropy(logits, labels)
    
        margin_loss = 0.0
        for feature, lab in zip(labeled_features, labels):
            dA = torch.norm(feature - proto_A)
            dB = torch.norm(feature - proto_B)
    
            if lab == 0:  
                margin_loss += F.relu(dA - dB + margin)
            else:       
                margin_loss += F.relu(dB - dA + margin)
    
        margin_loss = margin_loss / len(labels)
    
        constraint_loss = ce_loss + lambda_margin * margin_loss
    
        return constraint_loss

        
    def forward(self, group_embeddings, group_cell_indices, epoch, stem_embeddings, train_proto):
        T = self.get_temp(epoch)
        warmup_epochs = 20
        lamda_ent = min(1.0, epoch / warmup_epochs) * 0.1
        dist_A = torch.sum((stem_embeddings - self.proto_A.unsqueeze(0))**2, dim=1)
        dist_B = torch.sum((stem_embeddings - self.proto_B.unsqueeze(0))**2, dim=1)
        current_temp = 1
        logits = torch.stack([-dist_A / current_temp, -dist_B / current_temp], dim=1)
        probs = F.softmax(logits, dim=1)
        ratio_loss = self.adaptive_ratio_loss(probs)
        entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1).mean()
        if train_proto:
            entropy_loss = -lamda_ent * entropy
        else:
            entropy_loss = lamda_ent * entropy
            
        constraint_loss = self.ce_margin_loss(group_embeddings, group_cell_indices, self.proto_A, self.proto_B, labeled_data=self.labeled_target_data, margin=1.0, lambda_margin=0.5)
        
        total_loss = entropy_loss + constraint_loss + 10 * ratio_loss
        return total_loss
        
    @torch.no_grad()
    def assign_labels(self, stem_embeddings, label_map=(105,106)):
        stem_embeddings = torch.tensor(stem_embeddings, device=self.proto_A.device, dtype=self.proto_A.dtype)
        dist_to_A = torch.norm(stem_embeddings - self.proto_A.unsqueeze(0), dim=1)
        dist_to_B = torch.norm(stem_embeddings - self.proto_B.unsqueeze(0), dim=1)
        assigned = torch.where(dist_to_A < dist_to_B, label_map[0], label_map[1])
        return assigned

class ReconstructLoss(nn.Module):
    def __init__(self):
        super(ReconstructLoss, self).__init__()

    def forward(self, decoder, node_emb_sub):
        logp_x = F.log_softmax(decoder, dim=-1)
        p_y = F.softmax(node_emb_sub, dim=-1)
        loss_kl = F.kl_div(logp_x, p_y, reduction='mean')
        return loss_kl
