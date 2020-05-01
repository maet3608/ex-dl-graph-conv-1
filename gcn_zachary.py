import dgl
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import matplotlib.animation as animation
from torch.optim import Adam
from dgl.nn.pytorch import GraphConv

D_INPUT = 5
D_HIDDEN = 5
N_HIDDEN = 3


def build_karate_club_graph():
    src = np.array(
        [1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
         10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
         25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
         32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
         33, 33, 33, 33, 33, 33, 33, 33, 33, 33])
    dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
                    5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2,
                    23,
                    24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20,
                    22, 23,
                    29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27,
                    28, 29, 30,
                    31, 32])
    # Edges are directional in DGL; Make them bi-directional.
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    G = dgl.DGLGraph((u, v))
    return G


def show_graph(G):
    nx_G = G.to_networkx().to_undirected()
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True, font_size=9, node_color=[[.8, .8, .8]])


class GCN(nn.Module):
    def __init__(self, n_in, n_hidden, num_classes):
        super(GCN, self).__init__()
        layers = torch.nn.ModuleList()
        layers.append(GraphConv(n_in, n_hidden, activation=torch.relu))
        for _ in range(N_HIDDEN):
            layers.append(GraphConv(n_hidden, n_hidden, activation=torch.relu))
        layers.append(GraphConv(n_hidden, num_classes))
        self.layers = layers


    def forward(self, g, inputs):
        h = self.layers[0](g, inputs)
        for layer in self.layers[1:]:
            h = layer(g, h)
        return h


def create_network():
    net = GCN(D_INPUT, D_HIDDEN, 2)
    print(net)
    print(list(net.parameters()))
    return net


def train(net, G, device):
    from itertools import chain

    net.to(device)
    embed = nn.Embedding(34, D_INPUT)  # 34 nodes with embedding dim equal to 5
    embed = embed.to(device)

    # only the instructor and the president nodes are labeled
    labeled_nodes = torch.tensor([0, 33], device=device)
    labels = torch.tensor([0, 1], device=device)  # their labels are different

    optimizer = Adam(chain(net.parameters(), embed.parameters()), lr=0.01)
    # optimizer = Adam(net.parameters(), lr=0.1)

    all_logits = []
    for epoch in range(100):
        logits = net(G, embed.weight)
        # we save the logits for visualization later
        all_logits.append(logits.detach().cpu())
        logp = F.log_softmax(logits, 1)
        # we only compute loss for labeled nodes
        loss = F.nll_loss(logp[labeled_nodes], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
    return all_logits


def draw(i, ax, G, all_logits):
    cls1color = 'red'
    cls2color = 'steelblue'
    pos = {}
    colors = []
    for v in range(34):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(cls1color if cls else cls2color)
    nx_G = G.to_networkx().to_undirected()
    ax.cla()
    ax.axis('off')
    ax.set_title('Epoch: %d' % i)
    # pos = nx.kamada_kawai_layout(nx_G)
    nx.draw_networkx(nx_G, pos, node_color=colors, font_size=9,
                     with_labels=True, node_size=300, ax=ax)


if __name__ == '__main__':
    G = build_karate_club_graph()
    # print('%d nodes' % G.number_of_nodes())
    # print('%d edges' % G.number_of_edges())
    #
    # show_graph(G)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'running on {device}')

    net = create_network()
    all_logits = train(net, G, device)

    fig = plt.figure(dpi=150)
    fig.clf()
    ax = fig.subplots()
    ani = animation.FuncAnimation(fig, draw,
                                  fargs=(ax, G, all_logits),
                                  frames=len(all_logits),
                                  interval=200)
    # draw(len(all_logits) - 1, G, all_logits, )

    plt.show()
