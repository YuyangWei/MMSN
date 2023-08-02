from embedding import *
from collections import OrderedDict
import torch
import json
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import math

class LSTM_attn(nn.Module):
    def __init__(self, embed_size=100, n_hidden=200, out_size=100, layers=1):
        super(LSTM_attn, self).__init__()
        self.embed_size = embed_size
        self.n_hidden = n_hidden
        self.out_size = out_size
        self.layers = layers
        self.lstm = nn.LSTM(self.embed_size*2, self.n_hidden, self.layers, bias=False, bidirectional=False)
        # self.lstm = nn.LSTM(self.embed_size*2, self.n_hidden, self.layers, bidirectional=False)
        self.out = nn.Linear(self.n_hidden*self.layers, self.out_size, bias=False)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden, self.layers)
        attn_weight = torch.bmm(lstm_output, hidden).squeeze(2)
        attn_weight = attn_weight.view(lstm_output.shape[0], -1, self.layers)
        soft_attn_weight = F.softmax(attn_weight, 1)
        context = torch.bmm(lstm_output.transpose(1,2), soft_attn_weight)
        context = context.view(-1, self.n_hidden*self.layers)
        return context

    def forward(self, inputs):
        size = inputs.shape
        inputs = inputs.contiguous().view(size[0], size[1], -1)
        input = inputs.permute(1, 0, 2)
        hidden_state = Variable(torch.zeros(self.layers, size[0], self.n_hidden)).cuda()
        cell_state = Variable(torch.zeros(self.layers, size[0], self.n_hidden)).cuda()
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2)
        attn_output = self.attention_net(output, final_hidden_state)
        outputs = self.out(attn_output)
        return outputs.view(size[0], 1, 1, self.out_size)

class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(4))
        init.constant_(self.weight, 0.15)
        # self.weight = nn.Parameter(torch.FloatTensor(4,1))
        # init.xavier_normal_(self.weight)

    def forward(self, h, t, h_multi, t_multi, r, pos_num):

        score_ss= -torch.norm(h + r - t, 2, -1).squeeze(2)
        score_is = -torch.norm(h_multi + r - t, 2, -1).squeeze(2)
        score_si = -torch.norm(h + r - t_multi, 2, -1).squeeze(2)
        score_ii = -torch.norm(h_multi + r - t_multi, 2, -1).squeeze(2)

        score = self.weight[0] * score_ss + self.weight[1] * score_is + self.weight[2] * score_si + self.weight[3] * score_ii
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score

def save_grad(grad):
    global grad_norm
    grad_norm = grad

class MMSN(nn.Module):
    def __init__(self, dataset, parameter, num_symbols, multi_emb, embed = None):
        super(MMSN, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.abla = parameter['ablation']
        self.rel2id = dataset['rel2id']
        self.relation2id = dataset['relation2id']
        print("self.rel2id",len(self.rel2id))
        self.ent2id = dataset['ent2id']
        self.num_old_rel = len(self.relation2id)
        self.num_rel = len(self.rel2id)
        self.embedding = Embedding(dataset, parameter)
        self.few = parameter['few']
        self.dropout = nn.Dropout(0)
        self.symbol_emb = nn.Embedding(num_symbols + 1, self.embed_dim, padding_idx = num_symbols)

        self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))

        self.h_emb = nn.Embedding(self.num_rel, self.embed_dim)
        init.xavier_uniform_(self.h_emb.weight)

        self.gcn_w = nn.Linear(2*self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))
        self.attn_w = nn.Linear(self.embed_dim, 1)

        self.gate_w = nn.Linear(self.embed_dim, 1)
        self.gate_b = nn.Parameter(torch.FloatTensor(1))

        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        init.xavier_normal_(self.attn_w.weight)

        self.symbol_emb.weight.requires_grad = True
        self.h_norm = None

        if parameter['dataset'] == 'FB-One':
            self.relation_learner = LSTM_attn(embed_size=100, n_hidden=450, out_size=100, layers=1)
        elif parameter['dataset'] == 'DB-One':
            self.relation_learner = LSTM_attn(embed_size=100, n_hidden=450, out_size=100, layers=1)
        self.embedding_learner = EmbeddingLearner()
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()
        self.norm_q_sharing = dict()

        multi_dim = np.size(multi_emb, 1)
        num_entities = np.size(multi_emb, 0)
        self.multi_emb = nn.Embedding(num_entities, multi_dim)
        self.multi_emb.weight.data.copy_(torch.from_numpy(multi_emb))
        self.multi_emb.requires_grad = True
        self.multi_map_w = nn.Linear(multi_dim, self.embed_dim)
        self.multi_map_b = nn.Parameter(torch.FloatTensor(self.embed_dim))

        init.xavier_normal_(self.multi_map_w.weight)
        init.constant_(self.multi_map_b, 0)

        self.beta_w = nn.Linear(2 * self.embed_dim, 1)
        self.beta_b = nn.Parameter(torch.FloatTensor(1))
        init.xavier_normal_(self.beta_w.weight)
        init.constant_(self.beta_b, 0)

        self.fuse_w = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.fuse_b = nn.Parameter(torch.FloatTensor(self.embed_dim))
        self.fuse_attn_w = nn.Linear(self.embed_dim, 1)
        init.xavier_normal_(self.fuse_w.weight)
        init.constant_(self.fuse_b, 0)
        init.xavier_normal_(self.fuse_attn_w.weight)

        self.s_w = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.s_b = nn.Parameter(torch.FloatTensor(self.embed_dim))
        self.s_attn_w = nn.Linear(self.embed_dim, 1)
        init.xavier_normal_(self.s_w.weight)
        init.constant_(self.s_b, 0)
        init.xavier_normal_(self.s_attn_w.weight)

        self.attention = selfAttention(1, 400, 200)

    def multi_map(self, multi_emb):
        mapped_multi = self.multi_map_w(multi_emb) #+ self.multi_map_b
        return mapped_multi.tanh()

    def neighbot_multi(self, connections, istest):
        relations = connections[:,:,1].squeeze(-1)
        entities = connections[:,:,2].squeeze(-1)
        entself = connections[:,0,0].squeeze(-1)
        rel_embeds = self.dropout(self.symbol_emb(relations)) # (batch, 200, embed_dim)
        multi_embeds = self.dropout(self.multi_emb(entities-self.num_old_rel))
        multi_embeds_map = self.multi_map(multi_embeds)
        multiself_embeds = self.dropout(self.multi_emb(entself-self.num_old_rel))
        multiself_embeds_map = self.multi_map(multiself_embeds)
        concat_embeds = torch.cat((rel_embeds, multi_embeds_map), dim=-1) # (batch, 200, 2*embed_dim)

        out = self.gcn_w(concat_embeds) + self.gcn_b
        out = F.leaky_relu(out)
        attn_out = self.attn_w(out)
        attn_weight = F.softmax(attn_out, dim=1)
        out_attn = torch.bmm(out.transpose(1,2), attn_weight)
        out_attn = out_attn.squeeze(2)
        out_neighbor = out_attn + multiself_embeds_map
        return out_neighbor

    def neighbor_encoder(self, connections, istest):
        relations = connections[:,:,1].squeeze(-1)
        entities = connections[:,:,2].squeeze(-1)
        entself = connections[:,0,0].squeeze(-1)
        rel_embeds = self.dropout(self.symbol_emb(relations)) # (batch, 200, embed_dim)
        ent_embeds = self.dropout(self.symbol_emb(entities)) # (batch, 200, embed_dim)
        entself_embeds = self.dropout(self.symbol_emb(entself))
        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1) # (batch, 200, 2*embed_dim)
        out = self.gcn_w(concat_embeds) + self.gcn_b
        out = F.leaky_relu(out)
        attn_out = self.attn_w(out)
        attn_weight = F.softmax(attn_out, dim=1)
        out_attn = torch.bmm(out.transpose(1,2), attn_weight)
        out_attn = out_attn.squeeze(2)
        out_neighbor = out_attn + entself_embeds

        return out_neighbor
    
    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def get_few_support_graph(self, support_meta, istest):

        support_left_connections, _, support_right_connections, _ = support_meta[0]

        support_left = self.neighbor_encoder(support_left_connections,istest)
        support_right = self.neighbor_encoder(support_right_connections, istest)
        support_few = torch.cat((support_left, support_right), dim=-1)
        support_few = support_few.view(support_few.shape[0], -1, 2 * self.embed_dim)

        for i in range(self.few-1):
            support_left_connections, _, support_right_connections, _ = support_meta[i+1]
            support_left = self.neighbor_encoder(support_left_connections, istest)
            support_right = self.neighbor_encoder(support_right_connections, istest)
            support_pair = torch.cat((support_left, support_right), dim=-1)  # tanh
            support_pair = support_pair.view(support_pair.shape[0], -1, 2 * self.embed_dim)
            support_few = torch.cat((support_few, support_pair), dim=1)
        return support_few

    def get_few_support_multi(self, support_meta, istest):

        support_left_connections, _, support_right_connections, _ = support_meta[0]

        support_left = self.neighbot_multi(support_left_connections,istest)
        support_right = self.neighbot_multi(support_right_connections, istest)
        support_few = torch.cat((support_left, support_right), dim=-1)
        support_few = support_few.view(support_few.shape[0], -1, 2 * self.embed_dim)

        for i in range(self.few-1):
            support_left_connections, _, support_right_connections, _ = support_meta[i+1]
            support_left = self.neighbot_multi(support_left_connections, istest)
            support_right = self.neighbot_multi(support_right_connections, istest)
            support_pair = torch.cat((support_left, support_right), dim=-1)  # tanh
            support_pair = support_pair.view(support_pair.shape[0], -1, 2 * self.embed_dim)
            support_few = torch.cat((support_few, support_pair), dim=1)
        return support_few

    def gating_network(self, struc_support_few, multi_support_few):
        multi_support_temp = self.fuse_w(multi_support_few) + self.fuse_b
        tmp_fuse_gate = self.fuse_attn_w(multi_support_temp)
        fuse_gate = torch.sigmoid(tmp_fuse_gate)
        support_few = torch.mul(multi_support_few, fuse_gate) + torch.mul(struc_support_few, 1-fuse_gate)

        return support_few

    def get_s_multi(self, struc, multi):

        multi_emb = self.multi_map(multi)
        s_multi = torch.cat((struc, multi_emb), dim=-1)
        out = self.s_w(s_multi) + self.s_b
        attn_out = self.s_attn_w(out)
        attn_weight = F.softmax(attn_out, dim=1)
        # attn_weight =torch.sigmoid(attn_out)
        out_attn = multi_emb * attn_weight
        
        return out_attn

    def get_multi(self, triples):
        
        idx = [[[self.ent2id[t[0]], self.ent2id[t[2]]] for t in batch] for batch in triples]
        idx = torch.LongTensor(idx).to(self.device)
        return self.multi_emb(idx)

    def forward(self, task, iseval=False, curr_rel='', support_meta=None, istest=False):
        # transfer task string into embedding
        support, support_negative, query, negative = [self.embedding(t) for t in task]
        # print("support",support.shape)
        support_multi, support_negative_multi, query_multi, negative_multi = [self.get_multi(t) for t in task]
        support_s_multi = self.get_s_multi(support, support_multi)
        support_negative_s_multi = self.get_s_multi(support_negative, support_negative_multi)
        query_s_multi = self.get_s_multi(query, query_multi)
        negative_s_multi = self.get_s_multi(negative, negative_multi)

        few = support.shape[1]              # num of few
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative

        support_few_graph = self.get_few_support_graph(support_meta, istest)
        support_few_multi = self.get_few_support_multi(support_meta, istest)
        support_few = self.gating_network(support_few_graph, support_few_multi)
        support_few = support_few.view(support_few.shape[0], self.few, 2, self.embed_dim)
        rel = self.relation_learner(support_few)
        rel.retain_grad()

        # relation for support
        rel_s = rel.expand(-1, few+num_sn, -1, -1)
        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]

        else:
            if not self.abla:
                # split on e1/e2 and concat on pos/neg
                sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)
                sup_neg_e1_multi, sup_neg_e2_multi = self.split_concat(support_s_multi, support_negative_s_multi)

                p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, sup_neg_e1_multi, sup_neg_e2_multi, rel_s, few)	# revise norm_vector
                
                y = torch.ones_like(p_score).to(self.device)
                self.zero_grad()
                loss = self.loss_func(p_score, n_score, y)
                loss.backward(retain_graph=True)
                grad_meta = rel.grad
                rel_q = rel - self.beta*grad_meta			# hyper-plane update
            else:
                rel_q = rel

            self.rel_q_sharing[curr_rel] = rel_q
        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)

        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        que_neg_e1_multi, que_neg_e2_multi = self.split_concat(query_s_multi, negative_s_multi)
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, que_neg_e1_multi, que_neg_e2_multi, rel_q, num_q)

        return p_score, n_score