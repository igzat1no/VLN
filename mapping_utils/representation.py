import numpy as np
from enum import Enum
import heapq
from scipy.spatial import KDTree


class NodeState(Enum):
    UNEXPLORED = 0
    EXPLORED = 1


class our_Node:
    def __init__(self, rgb_img, depth_img, position, encoder, idx):
        self.state = 0
        # self.rgb_imgs = [rgb_img]
        # self.depth_imgs = [depth_img]
        self.position = position
        self.idx = idx
        self.objects = []
        # self.rgb_features = [encoder.encode_rgb(rgb_img)]
        # self.depth_features = [encoder.encode_depth(depth_img)]


    def update(self, rgb_img, depth_img, pointcloud, encoder=None):
        self.rgb_imgs = [rgb_img]
        self.depth_imgs = [depth_img]
        self.rgb_features = []
        self.depth_features = []
        if encoder is not None:
            self.rgb_features = [encoder.encode_rgb(rgb_img)]
            self.depth_features = [encoder.encode_depth(depth_img)]
        self.pointcloud = pointcloud

    def update_obj(self, obj_indices):
        self.objects += obj_indices
        self.objects = list(set(self.objects))

    def upgrade(self, rgb_imgs, depth_imgs, encoder):
        self.state = 1
        self.rgb_imgs = rgb_imgs
        self.depth_imgs = depth_imgs
        self.rgb_features = []
        self.depth_features = []
        for rgb_img in rgb_imgs:
            self.rgb_features.append(encoder.encode_rgb(rgb_img))
        for depth_img in depth_imgs:
            self.depth_features.append(encoder.encode_depth(depth_img))

    # def combine_unexplored(self, node):
    #     assert node.state == NodeState.UNEXPLORED
    #     num1 = len(self.rgb_imgs)
    #     num2 = len(node.rgb_imgs)
    #     self.position = (num1 * self.position + num2 * node.position) / (num1 + num2)
    #     self.rgb_imgs += node.rgb_imgs
    #     self.depth_imgs += node.depth_imgs
    #     self.rgb_features += node.rgb_features
    #     self.depth_features += node.depth_features


class our_Graph:
    def __init__(self):
        self.nodes = []
        self.nodes_pos_to_idx = {}
        self.node_cnt = 0
        self.neighbors = []

    def update_obj(self, position, obj_indices):
        node = self.find_closest_node(position)
        node.update_obj(obj_indices)

    def add_node(self, node_pos):
        node_pos = node_pos.copy()
        self.nodes_pos_to_idx[tuple(node_pos)] = self.node_cnt
        node = our_Node(None, None, node_pos, None, self.node_cnt)
        self.nodes.append(node)
        self.node_cnt += 1
        self.neighbors.append([])

        # for i in range(self.node_cnt - 1):
        #     if np.linalg.norm(self.nodes[i].position - node.position) < 0.1:
        #         self.combine_nodes(i, self.node_cnt - 1)
        #         self.nodes.pop()
        #         self.node_cnt -= 1
        #         return i

    # def combine_nodes(self, ind1, ind2):
    #     node1 = self.nodes[ind1]
    #     node2 = self.nodes[ind2]
    #     if node1.state == NodeState.EXPLORED and node2.state == NodeState.EXPLORED:
    #         # TODO how to cope with this situation
    #         return
    #     elif node1.state == NodeState.EXPLORED or node2.state == NodeState.EXPLORED:
    #         if node2.state == NodeState.EXPLORED:
    #             node1, node2 = node2, node1
    #             ind1, ind2 = ind2, ind1
    #     else:
    #         node1.combine_unexplored(node2)

    #     self.neighbors[ind1] += self.neighbors[ind2]
    #     self.neighbors[ind1].remove(ind2)
    #     self.neighbors[ind1] = list(set(self.neighbors[ind1]))
    #     for v in self.neighbors[ind2]:
    #         self.neighbors[v].remove(ind2)
    #         if ind1 not in self.neighbors[v]:
    #             self.neighbors[v].append(ind1)

    def add_edge(self, node1, node2):
        node1 = node1.copy()
        node2 = node2.copy()

        node1 = self.find_closest_node(node1).position
        node2 = self.find_closest_node(node2).position

        node1_idx = self.nodes_pos_to_idx[tuple(node1)]
        node2_idx = self.nodes_pos_to_idx[tuple(node2)]

        self.neighbors[node1_idx].append(node2_idx)
        self.neighbors[node2_idx].append(node1_idx)

    def visit_node(self, node_pos):
        for key, value in self.nodes_pos_to_idx.items():
            print(key, value)
        node_indx = self.nodes_pos_to_idx[tuple(node_pos)]
        self.nodes[node_indx].state = 1

    def update_edges(self, point_cloud_position, curren_pos):
        current_node = self.find_closest_node(curren_pos)

        average_z = np.mean(point_cloud_position[:, 2])
        point_cloud_position[:, 2] = np.ones_like(point_cloud_position[:, 2])*average_z
        tree = KDTree(point_cloud_position)

        for node in self.nodes:
            pos_new = np.array([node.position[0], node.position[1], average_z])
            indices = tree.query_ball_point(pos_new, 0.16)
            local_density = len(indices)

            if local_density > 15:
                node_idx = self.nodes_pos_to_idx[tuple(node.position)]
                if node_idx not in self.neighbors[self.nodes_pos_to_idx[tuple(current_node.position)]]:
                    self.add_edge(current_node.position, node.position)

    def find_closest_node(self, position):
        # find the closet node in the graph to any given position
        dist = []
        for node in self.nodes:
            dist.append(np.linalg.norm(node.position[:2] - position[:2]))
        dist = np.array(dist)

        return self.nodes[np.argmin(dist)]


    def find_closest_unexplored_node(self, node_pos):
        # find the closet node to any given node in the graph, distance is the sum of edge weights
        node = self.find_closest_node(node_pos)
        node_indx = self.nodes_pos_to_idx[tuple(node.position)]
        dist = np.full(self.node_cnt, np.inf)
        dist[node_indx] = 0

        pq = []
        heapq.heappush(pq, (0, node_indx))
        while pq:
            current_dist, u = heapq.heappop(pq)
            if self.nodes[u].state == 0:
                return self.nodes[u].position
            for v in self.neighbors[u]:
                alt = current_dist + np.linalg.norm(self.nodes[u].position[:2] - self.nodes[v].position[:2])
                if alt < dist[v]:
                    dist[v] = alt
                    heapq.heappush(pq, (alt, v))
        return None


    def find_the_closest_path(self, start, end):
        # Initialize distances and previous nodes
        start = self.find_closest_node(start)
        end = self.find_closest_node(end)

        start_idx = self.nodes_pos_to_idx[tuple(start.position)]
        end_idx = self.nodes_pos_to_idx[tuple(end.position)]

        dist = np.full(self.node_cnt, np.inf)
        prev = np.full(self.node_cnt, -1, dtype=int)
        dist[start_idx] = 0

        # Priority queue for Dijkstra's algorithm
        pq = []
        heapq.heappush(pq, (0, start_idx))  # (distance, node)

        while pq:
            current_dist, u_idx = heapq.heappop(pq)
            u = self.nodes[u_idx]
            # If we reach the end node, stop early
            if u == end:
                break

            # Skip if the distance is not optimal
            if current_dist > dist[u_idx]:
                continue

            # Iterate over neighbors
            for v_idx in self.neighbors[u_idx]:
                alt = dist[u_idx] + np.linalg.norm(self.nodes[u_idx].position[:2] - self.nodes[v_idx].position[:2])
                if alt < dist[v_idx]:
                    dist[v_idx] = alt
                    prev[v_idx] = u_idx
                    heapq.heappush(pq, (alt, v_idx))

        # Reconstruct path from end to start
        path = []
        u = end
        u_idx = self.nodes_pos_to_idx[tuple(u.position)]
        while u_idx != -1:
            path.append(self.nodes[u_idx].position)
            u_idx = prev[u_idx]
        path.reverse()

        # If the path doesn't reach the start node, return an empty path
        if tuple(path[0]) != tuple(start.position):
            return []
        return np.array(path)

    def get_nodes_positions(self):
        return np.array([node.position for node in self.nodes])

    def get_nodes_states(self):
        return np.array([node.state for node in self.nodes])

    def get_edges(self):
        edges = []
        for i in range(self.node_cnt):
            for j in self.neighbors[i]:
                if i < j:
                    edges.append((i, j))
        return edges