import dgl

import torch as to
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
import dgl.function as df

from dgl.nn.pytorch import GraphConv

DIM_FEAT = 5


def build_graph():
    # 0->1, 0->2, 1-2
    src = [0, 0, 1]
    dst = [1, 2, 2]
    graph = dgl.DGLGraph((src, dst))
    n_nodes = graph.number_of_nodes()
    features = to.zeros((n_nodes, DIM_FEAT))
    for i in range(n_nodes):
        features[i,i] = 1
    return graph, features


def show_graph(graph):
    netx = graph.to_networkx()
    pos = nx.kamada_kawai_layout(netx)
    nx.draw(netx, pos, with_labels=True, font_size=9, node_color=[[.8, .8, .8]])


gcn_msg = df.copy_u('h', 'm')
gcn_reduce = df.sum('m', 'h')


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        #self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, graph, features):
        with graph.local_scope():
            graph.ndata['h'] = features
            graph.update_all(gcn_msg, gcn_reduce)
            h = graph.ndata['h']
            return h
            #return self.linear(h)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = GCNLayer(DIM_FEAT, 2)
        # self.layer2 = GCNLayer(5, 2)

    def forward(self, g, f):
        h = self.layer1(g, f)
        # h = to.relu(h)
        # h = self.layer2(g, h)
        return h


if __name__ == "__main__":
    net = Net()
    g, f = build_graph()
    print(f)
    print(net(g, f))
    show_graph(g)
    plt.show()
