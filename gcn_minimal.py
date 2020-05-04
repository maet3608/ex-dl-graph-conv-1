import dgl

import torch as to
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt


def build_graph():
    # 0->1, 0->2, 1-2
    src = [0, 0, 1]
    tar = [1, 2, 2]
    graph = dgl.DGLGraph((src, tar))
    n_nodes = graph.number_of_nodes()
    features = to.eye(n_nodes)  # row contains feature vector for node
    return graph, features


def show_graph(graph):
    netx = graph.to_networkx()
    pos = nx.kamada_kawai_layout(netx)
    nx.draw(netx, pos, with_labels=True, font_size=9, node_color=[[.8, .8, .8]])


def map_f(edges):  # = dgl.function.copy_u('h', 'm')
    return {'m': edges.src['h']}


def reduce_f(nodes):  # = dgl.function.sum('m', 'h')
    return {'h': to.sum(nodes.mailbox['m'], dim=1)}


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, graph, features):
        with graph.local_scope():
            graph.ndata['h'] = features
            graph.update_all(map_f, reduce_f)
            h = graph.ndata['h']
            return h


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = GCNLayer()

    def forward(self, g, f):
        h = self.layer1(g, f)
        # h = to.relu(h)
        return h


if __name__ == "__main__":
    net = Net()
    g, f = build_graph()
    print(f)
    print(net(g, f))
    show_graph(g)
    plt.show()
