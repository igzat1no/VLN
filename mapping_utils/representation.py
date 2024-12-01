import numpy as np
from enum import Enum


class NodeState(Enum):
    UNEXPLORED = 0
    EXPLORED = 1


class our_Node:
    def __init__(self, rgb_img, depth_img, position, encoder):
        self.state = NodeState.UNEXPLORED
        self.rgb_imgs = [rgb_img]
        self.depth_imgs = [depth_img]
        self.position = position
        self.rgb_features = [encoder.encode_rgb(rgb_img)]
        self.depth_features = [encoder.encode_depth(depth_img)]

    def upgrade(self, rgb_imgs, depth_imgs, encoder):
        self.state = NodeState.EXPLORED
        self.rgb_imgs = rgb_imgs
        self.depth_imgs = depth_imgs
        self.rgb_features = []
        self.depth_features = []
        for rgb_img in rgb_imgs:
            self.rgb_features.append(encoder.encode_rgb(rgb_img))
        for depth_img in depth_imgs:
            self.depth_features.append(encoder.encode_depth(depth_img))

    def combine_unexplored(self, node):
        assert node.state == NodeState.UNEXPLORED
        num1 = len(self.rgb_imgs)
        num2 = len(node.rgb_imgs)
        self.position = (num1 * self.position + num2 * node.position) / (num1 + num2)
        self.rgb_imgs += node.rgb_imgs
        self.depth_imgs += node.depth_imgs
        self.rgb_features += node.rgb_features
        self.depth_features += node.depth_features


class our_Graph:
    def __init__(self):
        self.nodes = []
        self.node_cnt = 0
        self.neighbors = []

    def add_node(self, node, parent=None):
        self.nodes.append(node)
        self.node_cnt += 1
        self.neighbors.append([])
        if parent is not None:
            self.neighbors[parent].append(self.node_cnt - 1)
            self.neighbors[self.node_cnt - 1].append(parent)

        for i in range(self.node_cnt - 1):
            if np.linalg.norm(self.nodes[i].position - node.position) < 0.1:
                self.combine_nodes(i, self.node_cnt - 1)
                break

    def combine_nodes(self, ind1, ind2):
        node1 = self.nodes[ind1]
        node2 = self.nodes[ind2]
        if node1.state == NodeState.EXPLORED and node2.state == NodeState.EXPLORED:
            # TODO how to cope with this situation
            return
        elif node1.state == NodeState.EXPLORED or node2.state == NodeState.EXPLORED:
            if node2.state == NodeState.EXPLORED:
                node1, node2 = node2, node1
                ind1, ind2 = ind2, ind1
        else:
            node1.combine_unexplored(node2)

        self.neighbors[ind1] += self.neighbors[ind2]
        self.neighbors[ind1].remove(ind2)
        self.neighbors[ind1] = list(set(self.neighbors[ind1]))
        for v in self.neighbors[ind2]:
            self.neighbors[v].remove(ind2)
            if ind1 not in self.neighbors[v]:
                self.neighbors[v].append(ind1)