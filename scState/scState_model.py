import torch
import torch.nn as nn
import torch.nn.functional as F
from conv import *
from utils import *
import pandas as pd
from scipy.stats import mannwhitneyu

class GNN_from_raw(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout=0.2, conv_name='hgt',
                 prev_norm=True, last_norm=True):
        super(GNN_from_raw, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.adapt_ws = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        self.embedding1 = nn.ModuleList()

        # Initialize MLP weight matrices
        for ti in range(num_types):
            self.embedding1.append(nn.Linear(in_dim[ti], 256))

        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(256, n_hid))

        # Initialize graph convolution layers
        for l in range(n_layers - 1):
            self.gcs.append(
                GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm=prev_norm))
        self.gcs.append(
            GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm=last_norm))

    def encode(self, x, t_id):
        h1 = F.relu(self.embedding1[t_id](x))
        return h1

    def forward(self, node_feature, node_type, edge_index, edge_type):
        node_embedding = []
        for t_id in range(self.num_types):
            node_embedding += list(self.encode(node_feature[t_id], t_id))

        node_embedding = torch.stack(node_embedding)
        # Initialize result matrix
        res = torch.zeros(node_embedding.size(0), self.n_hid).to(node_feature[0].device)

        # Process each node type
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            # Update result matrix
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_embedding[idx]))

        # Apply dropout to the result matrix
        meta_xs = self.drop(res)
        del res

        # Iterate through graph convolution layers and update result matrix
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type)

        return meta_xs


class Net(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x

class NodeDimensionReduction(nn.Module):
    def __init__(self, RNA_matrix, GSVA_matrix, indices, ini_p1, n_hid, n_heads,
                 n_layers, labsm, lr, wd, device, num_types=3, num_relations=2, epochs=1):
        super(NodeDimensionReduction, self).__init__()
        self.RNA_matrix = RNA_matrix
        self.GSVA_matrix = GSVA_matrix
        self.indices = indices
        self.ini_p1 = ini_p1
        self.in_dim = [RNA_matrix.shape[0], RNA_matrix.shape[1], GSVA_matrix.shape[1]]
        self.n_hid = n_hid
        self.num_types = num_types
        self.num_relations = num_relations
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.labsm = labsm
        self.lr = lr
        self.wd = wd
        self.device = device
        self.epochs = epochs
        self.LabSm = LabelSmoothing(self.labsm)
        self.gnn = GNN_from_raw(in_dim=self.in_dim,
                                n_hid=self.n_hid,
                                num_types=self.num_types,
                                num_relations=self.num_relations,
                                n_heads=self.n_heads,
                                n_layers=self.n_layers,
                                dropout=0.3).to(self.device)
        self.optimizer = torch.optim.AdamW(self.gnn.parameters(), lr=self.lr, weight_decay=self.wd)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=5,
                                                                    verbose=True)

    def train_model(self, n_batch):
        print('The training process for the NodeDimensionReduction model has started. Please wait.')
        for epoch in tqdm(range(self.epochs)):
            embedding = []
            l_pre = []
            # embeddings = []
            for batch_id in np.arange(n_batch):
                gene_index = self.indices[batch_id]['gene_index']
                cell_index = self.indices[batch_id]['cell_index']
                peak_index = self.indices[batch_id]['peak_index']
                gene_feature = self.RNA_matrix[list(gene_index),]
                cell_feature = self.RNA_matrix[:, list(cell_index)].T
                peak_feature = self.GSVA_matrix[list(peak_index),]
                gene_feature = torch.tensor(np.array(gene_feature.todense()), dtype=torch.float32).to(self.device)
                cell_feature = torch.tensor(np.array(cell_feature.todense()), dtype=torch.float32).to(self.device)
                peak_feature = torch.tensor(np.array(peak_feature.todense()), dtype=torch.float32).to(self.device)
                node_feature = [cell_feature, gene_feature, peak_feature]
                gene_cell_sub = self.RNA_matrix[list(gene_index),][:, list(cell_index)]
                peak_cell_sub = self.GSVA_matrix[list(peak_index),][:, list(cell_index)]
                # gene_cell_edge_index = torch.LongTensor([np.nonzero(gene_cell_sub)[0]+gene_cell_sub.shape[1],np.nonzero(gene_cell_sub)[1]]).to(device)
                # peak_cell_edge_index = torch.LongTensor([np.nonzero(peak_cell_sub)[0]+gene_cell_sub.shape[0]+gene_cell_sub.shape[1],np.nonzero(peak_cell_sub)[1]]).to(device)
                gene_cell_edge_index1 = list(np.nonzero(gene_cell_sub)[0] + gene_cell_sub.shape[1]) + list(
                    np.nonzero(gene_cell_sub)[1])
                gene_cell_edge_index2 = list(np.nonzero(gene_cell_sub)[1]) + list(
                    np.nonzero(gene_cell_sub)[0] + gene_cell_sub.shape[1])
                gene_cell_edge_index = torch.LongTensor([gene_cell_edge_index1, gene_cell_edge_index2]).to(self.device)
                peak_cell_edge_index1 = list(
                    np.nonzero(peak_cell_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1]) + list(
                    np.nonzero(peak_cell_sub)[1])
                peak_cell_edge_index2 = list(np.nonzero(peak_cell_sub)[1]) + list(
                    np.nonzero(peak_cell_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1])
                peak_cell_edge_index = torch.LongTensor([peak_cell_edge_index1, peak_cell_edge_index2]).to(self.device)

                edge_index = torch.cat((gene_cell_edge_index, peak_cell_edge_index), dim=1)
                node_type = torch.LongTensor(np.array(
                    list(np.zeros(len(cell_index))) + list(np.ones(len(gene_index))) + list(
                        np.ones(len(peak_index)) * 2))).to(self.device)
                # edge_type = torch.LongTensor(np.array(list(np.zeros(gene_cell_edge_index.shape[1]))+list(np.ones(peak_cell_edge_index.shape[1]) ))).to(device)
                edge_type = torch.LongTensor(np.array(list(np.zeros(np.nonzero(gene_cell_sub)[0].shape[0])) + list(
                    np.ones(np.nonzero(gene_cell_sub)[1].shape[0])) + list(
                    2 * np.ones(np.nonzero(peak_cell_sub)[0].shape[0])) + list(
                    3 * np.ones(np.nonzero(peak_cell_sub)[1].shape[0])))).to(self.device)
                l = torch.LongTensor(np.array(self.ini_p1)[[cell_index]]).to(self.device)
                node_rep = self.gnn.forward(node_feature, node_type,
                                            edge_index,
                                            edge_type).to(self.device)
                cell_emb = node_rep[node_type == 0]
                gene_emb = node_rep[node_type == 1]
                peak_emb = node_rep[node_type == 2]

                decoder1 = torch.mm(gene_emb, cell_emb.t())
                decoder2 = torch.mm(peak_emb, cell_emb.t())
                gene_cell_sub = torch.tensor(np.array(gene_cell_sub.todense()), dtype=torch.float32).to(self.device)
                peak_cell_sub = torch.tensor(np.array(peak_cell_sub.todense()), dtype=torch.float32).to(self.device)
                
                loss_cluster = self.LabSm(cell_emb, l)
                lll = 0
                g = [int(i) for i in l]
                for i in set([int(k) for k in l]):
                    h = cell_emb[[True if i == j else False for j in g]]
                    ll = F.cosine_similarity(h[list(range(h.shape[0])) * h.shape[0],],
                                             h[[v for v in range(h.shape[0]) for i in range(h.shape[0])]]).mean()
                    lll = ll + lll
                loss = 5 * loss_cluster - lll
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                embedding.append(cell_emb)
                cell_pre = list(cell_emb.argmax(dim=1).cpu().detach().numpy())
                l_pre.extend(cell_pre)
                
            cell_embedding = torch.cat(embedding, dim=0)
            cell_clu = np.array(l_pre)
        print('The training for the NodeDimensionReduction model has been completed.')
        return self.gnn, cell_embedding, cell_clu


class scState_stage1(nn.Module):
    def __init__(self, gnn, labsm, Node_Ids, n_hid, n_batch, device, lr, palantir_pseudotime, wd, num_epochs=1):
        super(scState_stage1, self).__init__()
        self.lr = lr
        self.wd = wd
        self.gnn = gnn
        self.n_hid = n_hid
        self.n_batch = n_batch
        self.device = device
        self.num_epochs = num_epochs
        self.gnn_optimizer = torch.optim.AdamW(self.gnn.parameters(), lr=self.lr, weight_decay=self.wd)
        self.labsm = labsm
        self.LabSm = LabelSmoothing(self.labsm)
        self.palantir_pseudotime = palantir_pseudotime
        self.Node_Ids = Node_Ids

    def stem_cell_selector(self, cell_clu, palantir_pseudotime, Node_Ids, alpha=0.05):
        clu_pseudotime = pd.DataFrame(cell_clu, index=Node_Ids)
        clu_pseudotime.columns = ['pre_cluster']
        clu_pseudotime['palantir_pseudotime'] = palantir_pseudotime[Node_Ids]
        per_clu_avg_pseudotime = clu_pseudotime.groupby('pre_cluster')['palantir_pseudotime'].mean()
        clu_pseudotime_sorted = clu_pseudotime.sort_values(by='palantir_pseudotime')
        ranked_clusters = per_clu_avg_pseudotime.sort_values().index.tolist()
        group_1 = clu_pseudotime[clu_pseudotime['pre_cluster'] == ranked_clusters[0]]['palantir_pseudotime']
        other_mask = clu_pseudotime['pre_cluster'] != ranked_clusters[0]
        group_2 = clu_pseudotime[other_mask]['palantir_pseudotime']
        stat, pval = mannwhitneyu(group_1, group_2)
        if pval < alpha: 
            stem_like_clusters = [ranked_clusters[0]]
            cluster_pseudotimes = {
                clu: clu_pseudotime[clu_pseudotime['pre_cluster'] == clu]['palantir_pseudotime']
                for clu in ranked_clusters
            }
            for i in range(1, len(ranked_clusters)):
                curr_clu = ranked_clusters[i]
                is_stem = False
                
                for stem_clu in stem_like_clusters:
                    pt_curr = cluster_pseudotimes[curr_clu]
                    pt_stem = cluster_pseudotimes[stem_clu]
                    
                    stat, pval = mannwhitneyu(pt_curr, pt_stem)
                    
                    if pval > 0.05:
                        is_stem = True
                        break  
                
                if is_stem:
                    stem_like_clusters.append(curr_clu)
                else:
                    break  
            stem_like_mapping = {clu: i + 1 for i, clu in enumerate(stem_like_clusters)}
            clu_pseudotime['stem_like'] = clu_pseudotime['pre_cluster'].map(stem_like_mapping).fillna(0).astype(int)
        else:
            clu_pseudotime['stem_like'] = 0
        return clu_pseudotime


    def forward(self, indices, RNA_matrix, GSVA_matrix, ini_p1):
        ini_p1 = np.array(ini_p1)
        for epoch in range(self.num_epochs):
            print(f'The {epoch} epoch')
            embedding = []
            l_pre = []
            is_stem_mask = []
            recon_loss = 0
            clus_loss = 0
            for batch_id in tqdm(np.arange(self.n_batch)):
                gene_index = indices[batch_id]['gene_index']
                cell_index = indices[batch_id]['cell_index']
                peak_index = indices[batch_id]['peak_index']
                gene_feature = RNA_matrix[list(gene_index),]
                cell_feature = RNA_matrix[:, list(cell_index)].T
                peak_feature = GSVA_matrix[list(peak_index),]
                gene_feature = torch.tensor(np.array(gene_feature.todense()), dtype=torch.float32).to(self.device)
                cell_feature = torch.tensor(np.array(cell_feature.todense()), dtype=torch.float32).to(self.device)
                peak_feature = torch.tensor(np.array(peak_feature.todense()), dtype=torch.float32).to(self.device)
                node_feature = [cell_feature, gene_feature, peak_feature]
                gene_cell_sub = RNA_matrix[list(gene_index),][:, list(cell_index)]
                peak_cell_sub = GSVA_matrix[list(peak_index),][:, list(cell_index)]
                gene_cell_edge_index1 = list(np.nonzero(gene_cell_sub)[0] + gene_cell_sub.shape[1]) + list(
                    np.nonzero(gene_cell_sub)[1])
                gene_cell_edge_index2 = list(np.nonzero(gene_cell_sub)[1]) + list(
                    np.nonzero(gene_cell_sub)[0] + gene_cell_sub.shape[1])
                gene_cell_edge_index = torch.LongTensor([gene_cell_edge_index1, gene_cell_edge_index2]).to(self.device)
                peak_cell_edge_index1 = list(
                    np.nonzero(peak_cell_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1]) + list(
                    np.nonzero(peak_cell_sub)[1])
                peak_cell_edge_index2 = list(np.nonzero(peak_cell_sub)[1]) + list(
                    np.nonzero(peak_cell_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1])
                peak_cell_edge_index = torch.LongTensor([peak_cell_edge_index1, peak_cell_edge_index2]).to(self.device)
                # gene_cell_edge_index = torch.LongTensor([np.nonzero(gene_cell_sub)[0]+gene_cell_sub.shape[1],np.nonzero(gene_cell_sub)[1]]).to(device)
                # peak_cell_edge_index = torch.LongTensor([np.nonzero(peak_cell_sub)[0]+gene_cell_sub.shape[0]+gene_cell_sub.shape[1],np.nonzero(peak_cell_sub)[1]]).to(device)
                edge_index = torch.cat((gene_cell_edge_index, peak_cell_edge_index), dim=1)
                node_type = torch.LongTensor(np.array(
                    list(np.zeros(len(cell_index))) + list(np.ones(len(gene_index))) + list(
                        np.ones(len(peak_index)) * 2))).to(self.device)
                edge_type = torch.LongTensor(np.array(list(np.zeros(np.nonzero(gene_cell_sub)[0].shape[0])) + list(
                    np.ones(np.nonzero(gene_cell_sub)[1].shape[0])) + list(
                    2 * np.ones(np.nonzero(peak_cell_sub)[0].shape[0])) + list(
                    3 * np.ones(np.nonzero(peak_cell_sub)[1].shape[0])))).to(self.device)

                # edge_type = torch.LongTensor(np.array(list(np.zeros(gene_cell_edge_index.shape[1]))+list(np.ones(peak_cell_edge_index.shape[1]) ))).to(device)
                l = torch.LongTensor(np.array(ini_p1)[[cell_index]]).to(self.device)
                
                node_rep = self.gnn.forward(node_feature, node_type,
                                            edge_index,
                                            edge_type).to(self.device)
                cell_emb = node_rep[node_type == 0]
                gene_emb = node_rep[node_type == 1]
                peak_emb = node_rep[node_type == 2]

                decoder1 = torch.mm(gene_emb, cell_emb.t())
                decoder2 = torch.mm(peak_emb, cell_emb.t())
                gene_cell_sub = torch.tensor(np.array(gene_cell_sub.todense()), dtype=torch.float32).to(self.device)
                peak_cell_sub = torch.tensor(np.array(peak_cell_sub.todense()), dtype=torch.float32).to(self.device)

                logp_x1 = F.log_softmax(decoder1, dim=-1)
                p_y1 = F.softmax(gene_cell_sub, dim=-1)
                loss_kl1 = F.kl_div(logp_x1, p_y1, reduction='mean')
                logp_x2 = F.log_softmax(decoder2, dim=-1)
                p_y2 = F.softmax(peak_cell_sub, dim=-1)
                loss_kl2 = F.kl_div(logp_x2, p_y2, reduction='mean')
                loss_kl = loss_kl1 + loss_kl2

                loss_cluster = self.LabSm(cell_emb, l)
                recon_loss += loss_kl
                clus_loss += loss_cluster

                embedding.append(cell_emb)
    
                cell_pre = list(cell_emb.argmax(dim=1).cpu().detach().numpy())
                l_pre.extend(cell_pre)
                
            cell_embedding = torch.cat(embedding, dim=0)
            cell_clu = np.array(l_pre)
            # Calculate the average pseudo-time for each class, compute the degree of variation, and return a list indicating whether each cell is a stem cell
            clu_pseudotime = self.stem_cell_selector(cell_clu=cell_clu, palantir_pseudotime=self.palantir_pseudotime, Node_Ids=self.Node_Ids)
            stem_like_clus = clu_pseudotime[clu_pseudotime['stem_like'] != 0]['stem_like'].unique()
            eps = 1e-8
            loss_stem_pseudotime = 0
        
            for stem_like_clu in stem_like_clus:
                stem_mask = clu_pseudotime.reset_index(drop=True).index[clu_pseudotime['stem_like'] == stem_like_clu].tolist()
                stem_embedding = cell_embedding[stem_mask]
                stem_cell_pt = torch.tensor(clu_pseudotime[clu_pseudotime['stem_like'] == stem_like_clu]['palantir_pseudotime'].to_numpy(), dtype=torch.float32, device=self.device)
                is_stem_mask = is_stem_mask + stem_mask

                stem_probs = F.softmax(stem_embedding, dim=1)
                stem_class_prob = stem_probs[:, stem_like_clu]
                pt_weights = F.softmax(stem_cell_pt * 5, dim=0) 
                weighted_loss = (pt_weights * stem_class_prob).sum()
                loss_stem_pseudotime += weighted_loss

            alpha = 1
            beta = 1
            gamma = 10
            loss_stage1 = alpha * recon_loss + beta * clus_loss + gamma * loss_stem_pseudotime
            self.gnn_optimizer.zero_grad()
            loss_stage1.backward()
            self.gnn_optimizer.step()

        return self.gnn, is_stem_mask, cell_embedding, stem_like_clus

    def train_model(self, indices, RNA_matrix, GSVA_matrix, ini_p1):
        self.train()
        print('The training process for the scState model has started. Please wait.')
        Mars_gnn, is_stem_mask, cell_embedding, stem_like_clus = self.forward(indices, RNA_matrix, GSVA_matrix, ini_p1)
        print('The training for the scState model has been completed.')
        return Mars_gnn, is_stem_mask, cell_embedding, stem_like_clus

class scState_stage2(nn.Module):
    def __init__(self, gnn, labsm, Node_Ids, n_hid, n_batch, device, lr, wd, cell_embedding, prototype_indicies, is_stem_mask, stem_pseudo_labels, labeled_target_data, gmm_ratio, indices, RNA_matrix, GSVA_matrix, num_epochs=1):
        super(scState_stage2, self).__init__()
        self.lr = lr
        self.wd = wd
        self.gnn = gnn
        self.n_hid = n_hid
        self.n_batch = n_batch
        self.device = device
        self.num_epochs = num_epochs
        self.prototype_indicies = prototype_indicies
        self.is_stem_mask = is_stem_mask
        self.labeled_target_data = labeled_target_data
        self.gmm_ratio = gmm_ratio
            
        cell_embedding_all = torch.tensor(cell_embedding, dtype=torch.float32, device=self.device)
        self.old_embeddings = cell_embedding_all.detach()
        self.super_labels_predict = cell_embedding_all.argmax(dim=1)
        self.stem_pseudo_labels = stem_pseudo_labels
            
        prototypes = cell_embedding_all[self.prototype_indicies]
        self.proto_loss_fn = ProtoClusteringLoss(device=self.device, prototypes=prototypes, labeled_target_data=self.labeled_target_data, stem_pseudo_labels=self.stem_pseudo_labels, gmm_ratio=self.gmm_ratio)
        
        self.gnn_optimizer = torch.optim.AdamW(list(self.gnn.parameters()), lr=self.lr, weight_decay=self.wd)
        self.proto_optimizer = torch.optim.AdamW(list(self.proto_loss_fn.parameters()), lr=self.lr, weight_decay=self.wd)
        self.labsm = labsm
        self.LabSm = LabelSmoothing(self.labsm)
        self.Node_Ids = Node_Ids
        
        self.ReConLoss = ReconstructLoss()

    def gnn_node_emb(self, batch_id, indices, RNA_matrix, GSVA_matrix, ini_p1):
        gene_index = indices[batch_id]['gene_index']
        cell_index = indices[batch_id]['cell_index']
        peak_index = indices[batch_id]['peak_index']
        gene_feature = RNA_matrix[list(gene_index),]
        cell_feature = RNA_matrix[:, list(cell_index)].T
        peak_feature = GSVA_matrix[list(peak_index),]
        gene_feature = torch.tensor(np.array(gene_feature.todense()), dtype=torch.float32).to(self.device)
        cell_feature = torch.tensor(np.array(cell_feature.todense()), dtype=torch.float32).to(self.device)
        peak_feature = torch.tensor(np.array(peak_feature.todense()), dtype=torch.float32).to(self.device)
        node_feature = [cell_feature, gene_feature, peak_feature]
        gene_cell_sub = RNA_matrix[list(gene_index),][:, list(cell_index)]
        peak_cell_sub = GSVA_matrix[list(peak_index),][:, list(cell_index)]
        gene_cell_edge_index1 = list(np.nonzero(gene_cell_sub)[0] + gene_cell_sub.shape[1]) + list(
            np.nonzero(gene_cell_sub)[1])
        gene_cell_edge_index2 = list(np.nonzero(gene_cell_sub)[1]) + list(
            np.nonzero(gene_cell_sub)[0] + gene_cell_sub.shape[1])
        gene_cell_edge_index = torch.LongTensor([gene_cell_edge_index1, gene_cell_edge_index2]).to(self.device)
        peak_cell_edge_index1 = list(
            np.nonzero(peak_cell_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1]) + list(
            np.nonzero(peak_cell_sub)[1])
        peak_cell_edge_index2 = list(np.nonzero(peak_cell_sub)[1]) + list(
            np.nonzero(peak_cell_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1])
        peak_cell_edge_index = torch.LongTensor([peak_cell_edge_index1, peak_cell_edge_index2]).to(self.device)
        # gene_cell_edge_index = torch.LongTensor([np.nonzero(gene_cell_sub)[0]+gene_cell_sub.shape[1],np.nonzero(gene_cell_sub)[1]]).to(device)
        # peak_cell_edge_index = torch.LongTensor([np.nonzero(peak_cell_sub)[0]+gene_cell_sub.shape[0]+gene_cell_sub.shape[1],np.nonzero(peak_cell_sub)[1]]).to(device)
        edge_index = torch.cat((gene_cell_edge_index, peak_cell_edge_index), dim=1)
        node_type = torch.LongTensor(np.array(
            list(np.zeros(len(cell_index))) + list(np.ones(len(gene_index))) + list(
                np.ones(len(peak_index)) * 2))).to(self.device)
        edge_type = torch.LongTensor(np.array(list(np.zeros(np.nonzero(gene_cell_sub)[0].shape[0])) + list(
            np.ones(np.nonzero(gene_cell_sub)[1].shape[0])) + list(
            2 * np.ones(np.nonzero(peak_cell_sub)[0].shape[0])) + list(
            3 * np.ones(np.nonzero(peak_cell_sub)[1].shape[0])))).to(self.device)

        l = torch.LongTensor(np.array(ini_p1)[cell_index]).to(self.device)
        
        node_rep = self.gnn.forward(node_feature, node_type,
                                    edge_index,
                                    edge_type).to(self.device)
        cell_emb = node_rep[node_type == 0]
        gene_emb = node_rep[node_type == 1]
        peak_emb = node_rep[node_type == 2]

        gene_cell_sub = torch.tensor(np.array(gene_cell_sub.todense()), dtype=torch.float32).to(self.device)
        peak_cell_sub = torch.tensor(np.array(peak_cell_sub.todense()), dtype=torch.float32).to(self.device)

        return cell_emb, gene_emb, peak_emb, gene_cell_sub, peak_cell_sub, cell_index, l
        

    def forward(self, indices, RNA_matrix, GSVA_matrix, ini_p1, group_size = 200):    
        batch_groups = [
            list(range(i, min(i + group_size, self.n_batch)))
            for i in range(0, self.n_batch, group_size)
        ]    
        
        loss_stage2_list = []
        recon_loss_list = []
        proto_loss_min_list = []
        proto_loss_max_list = []
        cluster_loss_list = []
    
        ini_p1 = np.array(ini_p1)
        super_labels_pseudo = torch.tensor(ini_p1[[self.Node_Ids]]).to(self.device)
        self.is_stem_mask_TF = torch.zeros(len(ini_p1), dtype=torch.bool, device=self.device)
        self.is_stem_mask_TF[self.is_stem_mask] = True
        for epoch in tqdm(range(self.num_epochs)):
            for g, group in enumerate(batch_groups):
                embedding_buffer = []
                index_buffer = [] 
                l_buffer = []
                recon_loss_buffer = 0
    
                for batch_id in group:
                    (cell_emb, gene_emb, peak_emb,
                     gene_cell_sub, peak_cell_sub,
                     cell_index, l) = self.gnn_node_emb(batch_id, indices, RNA_matrix, GSVA_matrix, ini_p1)
                    embedding_buffer.append(cell_emb)
                    index_buffer.append(torch.tensor(cell_index, device=self.device, dtype=torch.long))
                    l_buffer.append(l)
                    decoder1 = torch.mm(gene_emb, cell_emb.t())
                    decoder2 = torch.mm(peak_emb, cell_emb.t())
                    loss_kl1 = self.ReConLoss(decoder1, gene_cell_sub)
                    loss_kl2 = self.ReConLoss(decoder2, peak_cell_sub)
                    recon_loss_buffer += (loss_kl1 + loss_kl2)
    
                group_cell_embedding = torch.cat(embedding_buffer, dim=0)
                group_cell_index = torch.cat(index_buffer, dim=0)
                group_l = torch.cat(l_buffer, dim=0)
                local_mask = self.is_stem_mask_TF[group_cell_index]
                
                group_mask_other = ~local_mask
                cluster_loss = self.LabSm(group_cell_embedding, group_l)
                
                stem_embeddings = group_cell_embedding[local_mask]
                if stem_embeddings.shape[0] != 0:
                    proto_loss_min = self.proto_loss_fn(
                        group_cell_embedding,
                        group_cell_index,
                        epoch,
                        stem_embeddings,  
                        train_proto=False
                    )
                    total_loss = 10 * recon_loss_buffer + proto_loss_min + 20 * cluster_loss
                else:
                    total_loss = 10 * recon_loss_buffer + 20 * cluster_loss
    
                self.gnn_optimizer.zero_grad()
                total_loss.backward()
                self.gnn_optimizer.step()

                group_cell_embedding_proto = group_cell_embedding.clone().detach()
                stem_embeddings_proto = stem_embeddings.clone().detach()
                proto_loss_max = self.proto_loss_fn(
                    group_cell_embedding_proto,
                    group_cell_index,
                    epoch,
                    stem_embeddings_proto,
                    train_proto=True
                ) 
                self.proto_optimizer.zero_grad()
                proto_loss_max.backward()
                self.proto_optimizer.step()

        return self.gnn, list(self.proto_loss_fn.parameters())

    def train_model(self, indices, RNA_matrix, GSVA_matrix, ini_p1):
        self.train()
        print('The training process for the scState model has started. Please wait.')
        Mars_gnn = self.forward(indices, RNA_matrix, GSVA_matrix, ini_p1)
        print('The training for the scState model has been completed.')
        return Mars_gnn


def scState_pred_stage2(RNA_matrix, GSVA_matrix, egrn, scState_gnn, indices, nodes_id, cell_size, device, gene_names,
                gepa_names, stem_clu, prototypes=None):
    # scState_gnn.eval()
    n_batch = math.ceil(nodes_id.shape[0] / cell_size)
    embedding = []
    l_pre = []
    scState_result = {}
    with torch.no_grad():
        for batch_id in range(n_batch):
            gene_index = indices[batch_id]['gene_index']
            cell_index = indices[batch_id]['cell_index']
            peak_index = indices[batch_id]['peak_index']
            gene_feature = RNA_matrix[list(gene_index),]
            cell_feature = RNA_matrix[:, list(cell_index)].T
            peak_feature = GSVA_matrix[list(peak_index),]
            gene_feature = torch.tensor(np.array(gene_feature.todense()), dtype=torch.float32).to(device)
            cell_feature = torch.tensor(np.array(cell_feature.todense()), dtype=torch.float32).to(device)
            peak_feature = torch.tensor(np.array(peak_feature.todense()), dtype=torch.float32).to(device)
            node_feature = [cell_feature, gene_feature, peak_feature]
            gene_cell_sub = RNA_matrix[list(gene_index),][:, list(cell_index)]
            peak_cell_sub = GSVA_matrix[list(peak_index),][:, list(cell_index)]
            gene_cell_edge_index = torch.LongTensor(
                [np.nonzero(gene_cell_sub)[0] + gene_cell_sub.shape[1], np.nonzero(gene_cell_sub)[1]]).to(device)
            peak_cell_edge_index = torch.LongTensor(
                [np.nonzero(peak_cell_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1],
                 np.nonzero(peak_cell_sub)[1]]).to(device)
            edge_index = torch.cat((gene_cell_edge_index, peak_cell_edge_index), dim=1)
            node_type = torch.LongTensor(np.array(
                list(np.zeros(len(cell_index))) + list(np.ones(len(gene_index))) + list(
                    np.ones(len(peak_index)) * 2))).to(device)
            edge_type = torch.LongTensor(np.array(
                list(np.zeros(gene_cell_edge_index.shape[1])) + list(np.ones(peak_cell_edge_index.shape[1])))).to(
                device)
            node_rep = scState_gnn.forward(node_feature, node_type,
                                          edge_index,
                                          edge_type).to(device)
            cell_emb = node_rep[node_type == 0]
            gene_emb = node_rep[node_type == 1]
            peak_emb = node_rep[node_type == 2]

            # If the device is CUDA, copy the tensor to CPU memory
            if device.type == "cuda":
                cell_emb = cell_emb.cpu()
            # It is now safe to convert the tensor to a NumPy array
            embedding.append(cell_emb.detach().numpy())

            cell_pre = list(cell_emb.argmax(dim=1).detach().numpy())
            l_pre.extend(cell_pre)

    cell_embedding = np.vstack(embedding)
    cell_clu = np.array(l_pre)

    # Split the Stem class
    if prototypes is not None:
        if isinstance(prototypes, (list, tuple)) or isinstance(prototypes, torch.nn.ParameterList):
            proto = [p.detach().cpu().numpy() for p in prototypes]
            proto = np.vstack(proto)  # shape (2, d)
        else:
            proto = prototypes.detach().cpu().numpy()

        idx_stem = np.where(np.isin(cell_clu, stem_clu))[0]
        if len(idx_stem) > 0:
            sub_emb = cell_embedding[idx_stem]
            dist_0 = np.linalg.norm(sub_emb - proto[0], axis=1)
            dist_1 = np.linalg.norm(sub_emb - proto[1], axis=1)
            new_labels = np.where(dist_0 < dist_1, 105, 106)
            cell_clu[idx_stem] = new_labels
            
    if egrn:
        final_egrn_df = egrn_calculate(cell_clu, nodes_id, RNA_matrix, GSVA_matrix, gene_names, peak_names)
        scState_result = {'pred_label': cell_clu, 'cell_embedding': cell_embedding, 'egrn': final_egrn_df}
        return scState_result
    else:
        scState_result = {'pred_label': cell_clu, 'cell_embedding': cell_embedding}
        return scState_result