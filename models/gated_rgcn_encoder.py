
import torch
import torch.nn as nn
# from onmt.const import relations_vocab
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import math
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


class RGCNConvWithGate(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConvWithGate, self).__init__(aggr='add', flow='target_to_source', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = Param(torch.Tensor(num_relations, num_bases))

        # Gate Mechanism
        self.gate_weight = Param(torch.Tensor(num_relations, out_channels, 1))

        if root_weight:
            self.root = Param(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)
        uniform(size, self.gate_weight)

    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """

        :param x: node feature
        :param edge_index: coo adj
        :param edge_type: type of edge
        :param edge_norm:
        :param size:
        :return:
        """
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)

    def message(self, x_j, edge_index_j, edge_type, edge_norm):

        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        # get from utterance encoder
        w = w.view(self.num_relations, self.in_channels, self.out_channels)
        # type specific matrix
        w = torch.index_select(w, 0, edge_type)
        """Gate Mechanism """
        gate_weight = torch.index_select(self.gate_weight, 0, edge_type)  # (20, 200, 1)
        gate = torch.sigmoid(torch.bmm(x_j.unsqueeze(1), gate_weight))
        x_j = torch.bmm(gate, x_j.unsqueeze(1))
        out = torch.bmm(x_j, w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)

class RGCNGateEncoder(torch.nn.Module):
    def __init__(self, hidden_size):
        super(RGCNGateEncoder, self).__init__()
        # self.relation_embeddings = nn.Embedding(len(relations_vocab), hidden_size)
        # self.relation_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.conv1 = RGCNConvWithGate(hidden_size, hidden_size, 2, num_bases=30)
        self.conv2 = RGCNConvWithGate(hidden_size, hidden_size, 2, num_bases=30)

    def forward(self, meeting_utterance_enc_hidden_states, adj_coos, edge_types): #rels
        # create graph batch
        list_geometric_data = []
        seg_len = []
        for meeting_utterance_enc_hidden_state, adj_coo, edge_type in zip(meeting_utterance_enc_hidden_states,
                                                                               adj_coos, edge_types):
            # rel = torch.LongTensor(rel).cuda()
            # rel_embed = self.relation_embeddings(rel)  # (relation num, hidden size)
            # emb = torch.cat((meeting_utterance_enc_hidden_state, rel_embed),
            #                 0)  # (utterance num + relation num, hidden_size)

            # assert emb.size(0) == max(adj_coo[0]) + 1, "make sure the emb == adj matrix"
            emb = meeting_utterance_enc_hidden_state
            seg_len.append(emb.size(0))

            edge_index = torch.tensor(adj_coo, dtype=torch.long).cuda()
            edge_type = torch.tensor(edge_type, dtype=torch.long).cuda()
            data = Data(x=emb, edge_index=edge_index)
            data.edge_type = edge_type  # edge_type
            list_geometric_data.append(data)

        batch_geometric = Batch.from_data_list(list_geometric_data).to('cuda')

        x = F.relu(self.conv1(batch_geometric.x, batch_geometric.edge_index, batch_geometric.edge_type))
        x = self.conv2(x, batch_geometric.edge_index, batch_geometric.edge_type)
        x = x.split(seg_len)

        return x
