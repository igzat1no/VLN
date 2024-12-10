import numpy as np

from mapping_utils.geometry import *
from mapping_utils.preprocess import *
from mapping_utils.projection import *
from mapping_utils.transform import *
from mapping_utils.path_planning import *
from cv_utils.image_percevior import GLEE_Percevior
from matplotlib import colormaps
from habitat_sim.utils.common import d3_40_colors_rgb
from constants import *
import open3d as o3d
from scipy.spatial import KDTree
from lavis.models import load_model_and_preprocess
from PIL import Image
from matplotlib.path import Path
import matplotlib.pyplot as plt
import networkx as nx
import os
from mapping_utils.representation import *


class Instruct_Mapper:
    def __init__(self,
                 camera_intrinsic,
                 pcd_resolution=0.05,
                 grid_resolution=0.1,
                 grid_size=5,
                 floor_height=-0.8,
                 ceiling_height=0.8,
                 translation_func=habitat_translation,
                 rotation_func=habitat_rotation,
                 rotate_axis=[0, 1, 0],
                 device='cuda:0'):
        self.device = device
        self.camera_intrinsic = camera_intrinsic
        self.pcd_resolution = pcd_resolution
        self.grid_resolution = grid_resolution
        self.grid_size = grid_size
        self.floor_height = floor_height
        self.ceiling_height = ceiling_height
        self.translation_func = translation_func
        self.rotation_func = rotation_func
        self.rotate_axis = np.array(rotate_axis)
        self.object_percevior = GLEE_Percevior(device=device)
        self.pcd_device = o3d.core.Device(device.upper())

        self.representation = our_Graph()
        self.current_obj_indices = []

        # self.nodes = np.array([])
        # self.nodes_state = np.array([])

        self.process_obs_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.process_nav_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

    def reset(self, position, rotation):
        self.update_iterations = 0
        self.initial_position = self.translation_func(position)
        self.current_position = self.translation_func(position) - self.initial_position
        self.current_rotation = self.rotation_func(rotation)
        self.scene_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.object_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.object_entities = []
        self.trajectory_position = []

        self.representation = our_Graph()
        # self.nodes = np.array([])
        # self.nodes_state = np.array([])

        self.process_obs_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.process_nav_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

    def update(self, rgb, depth, position, rotation):
        self.current_position = self.translation_func(position) - self.initial_position
        self.current_rotation = self.rotation_func(rotation)
        self.current_depth = preprocess_depth(depth)
        self.current_rgb = preprocess_image(rgb)
        self.trajectory_position.append(self.current_position)
        # to avoid there is no valid depth value (especially in real-world)
        if np.sum(self.current_depth) > 0:
            camera_points, camera_colors = get_pointcloud_from_depth(self.current_rgb, self.current_depth,
                                                                     self.camera_intrinsic)
            world_points = translate_to_world(camera_points, self.current_position, self.current_rotation)
            self.current_pcd = gpu_pointcloud_from_array(world_points, camera_colors,
                                                         self.pcd_device).voxel_down_sample(self.pcd_resolution)
        else:
            return

        # semantic masking and project object mask to pointcloud
        classes,masks,confidences,visualization = self.object_percevior.perceive(self.current_rgb)
        self.segmentation = visualization[0]
        current_object_entities = self.get_object_entities(self.current_depth,classes,masks,confidences)
        print("+++++++++++++++++++++")
        for objs in current_object_entities:
            print(objs["class"], objs["confidence"])
        print()
        self.object_entities, obj_indices = self.associate_object_entities(self.object_entities,current_object_entities)
        self.current_obj_indices += obj_indices
        print("+++++++++++++++++++++")
        print()
        self.object_pcd = self.update_object_pcd()

        # pointcloud update
        self.scene_pcd = gpu_merge_pointcloud(self.current_pcd, self.scene_pcd).voxel_down_sample(self.pcd_resolution)
        self.scene_pcd = self.scene_pcd.select_by_index(
            (self.scene_pcd.point.positions[:, 2] > self.floor_height - 0.2).nonzero()[0])
        self.useful_pcd = self.scene_pcd.select_by_index(
            (self.scene_pcd.point.positions[:, 2] < self.ceiling_height).nonzero()[0])

        # # all the stairs will be regarded as navigable
        # for entity in current_object_entities:
        #     if entity['class'] == 'stairs':
        #         self.navigable_pcd = gpu_merge_pointcloud(self.navigable_pcd,entity['pcd'])

        # geometry
        current_navigable_point = self.current_pcd.select_by_index(
            (self.current_pcd.point.positions[:, 2] < self.floor_height).nonzero()[0])
        current_navigable_position = current_navigable_point.point.positions.cpu().numpy()
        standing_position = np.array(
            [self.current_position[0], self.current_position[1], current_navigable_position[:, 2].mean()])

        if current_navigable_position.shape[0] != 0:
            distance = np.linalg.norm(current_navigable_position - standing_position, axis=1)
            index = np.argmin(distance)
            closest_distance = distance[index]
            closest_points = current_navigable_position[distance < closest_distance + 0.8]
            interpolate_points = np.linspace(np.ones_like(closest_points) * standing_position, closest_points, 25).reshape(
                -1, 3)
            interpolate_points = interpolate_points[
                (interpolate_points[:, 2] > self.floor_height - 0.2) & (interpolate_points[:, 2] < self.floor_height + 0.2)]
            interpolate_points = np.concatenate((current_navigable_position, interpolate_points), axis=0)

            interpolate_points[:, 2] = np.ones_like(interpolate_points[:, 2]) * np.mean(interpolate_points[:, 2])

            interpolate_colors = np.ones_like(interpolate_points) * 100
            try:
                current_navigable_pcd = gpu_pointcloud_from_array(interpolate_points, interpolate_colors,
                                                                  self.pcd_device)
                self.navigable_pcd = gpu_merge_pointcloud(self.navigable_pcd, current_navigable_pcd).voxel_down_sample(
                    self.pcd_resolution)

                # update th process obs_pcd
                self.process_nav_pcd = gpu_merge_pointcloud(self.process_nav_pcd, current_navigable_pcd).voxel_down_sample(
                    self.pcd_resolution)
            except:
                self.navigable_pcd = self.useful_pcd.select_by_index(
                    (self.useful_pcd.point.positions[:, 2] < self.floor_height).nonzero()[0])

        # update th process obs_pcd
        self.process_obs_pcd = gpu_merge_pointcloud(self.process_obs_pcd, self.current_pcd).voxel_down_sample(
            self.pcd_resolution)
        self.process_obs_pcd = self.process_obs_pcd.select_by_index(
            (self.process_obs_pcd.point.positions[:, 2] > self.floor_height - 0.2).nonzero()[0])
        self.process_obs_pcd = self.process_obs_pcd.select_by_index(
            (self.process_obs_pcd.point.positions[:, 2] < self.ceiling_height).nonzero()[0])

            # save_pcd = o3d.geometry.PointCloud()
            # save_pcd.points = o3d.utility.Vector3dVector(self.navigable_pcd.point.positions.cpu().numpy())
            # save_pcd.colors = o3d.utility.Vector3dVector(self.navigable_pcd.point.colors.cpu().numpy())
            # os.makedirs(f'tmp_with_whole_obs/episode-0', exist_ok=True)
            # o3d.io.write_point_cloud(f'tmp_with_whole_obs/episode-0/navigable_{self.update_iterations}.ply', save_pcd)

        # interpolate_points = np.linspace(np.ones_like(current_navigable_position) * standing_position,
        #                                  current_navigable_position, 25).reshape(-1, 3)
        # interpolate_points = interpolate_points[
        #     (interpolate_points[:, 2] > self.floor_height - 0.2) & (interpolate_points[:, 2] < self.floor_height + 0.2)]
        # interpolate_colors = np.ones_like(interpolate_points) * 100
        #
        # try:
        #     current_navigable_pcd = gpu_pointcloud_from_array(interpolate_points, interpolate_colors,
        #                                                       self.pcd_device).voxel_down_sample(self.grid_resolution)
        #     self.navigable_pcd = gpu_merge_pointcloud(self.navigable_pcd, current_navigable_pcd).voxel_down_sample(
        #         self.pcd_resolution)
        # except:
        #     self.navigable_pcd = self.useful_pcd.select_by_index(
        #         (self.useful_pcd.point.positions[:, 2] < self.floor_height).nonzero()[0])

        # try:
        #     self.navigable_pcd = self.navigable_pcd.voxel_down_sample(self.pcd_resolution)
        # except:
        #     self.navigable_pcd = self.useful_pcd.select_by_index((self.useful_pcd.point.positions[:,2]<self.floor_height).nonzero()[0])
        # print("Warning: hello world")
        # self.navigable_pcd = self.useful_pcd.select_by_index((self.useful_pcd.point.positions[:,2]<self.floor_height).nonzero()[0])

        # filter the obstacle pointcloud
        self.obstacle_pcd = self.useful_pcd.select_by_index(
            (self.useful_pcd.point.positions[:, 2] > self.floor_height + 0.1).nonzero()[0])
        self.trajectory_pcd = gpu_pointcloud_from_array(np.array(self.trajectory_position),
                                                        np.zeros((len(self.trajectory_position), 3)), self.pcd_device)

        # self.frontier_pcd = project_frontier(self.obstacle_pcd,self.navigable_pcd,self.floor_height+0.2,self.grid_resolution)
        # self.frontier_pcd[:,2] = self.navigable_pcd.point.positions.cpu().numpy()[:,2].mean()
        # self.frontier_pcd = gpu_pointcloud_from_array(self.frontier_pcd,np.ones((self.frontier_pcd.shape[0],3))*np.array([[255,0,0]]),self.pcd_device)

        self.update_iterations += 1

    def update_object_pcd(self):
        object_pcd = o3d.geometry.PointCloud()
        for entity in self.object_entities:
            points = entity['pcd'].point.positions.cpu().numpy()
            colors = entity['pcd'].point.colors.cpu().numpy()
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(points)
            new_pcd.colors = o3d.utility.Vector3dVector(colors)
            object_pcd = object_pcd + new_pcd
        try:
            return gpu_pointcloud(object_pcd, self.pcd_device)
        except:
            return self.scene_pcd

    def get_view_pointcloud(self, rgb, depth, translation, rotation):
        current_position = self.translation_func(translation) - self.initial_position
        current_rotation = self.rotation_func(rotation)
        current_depth = preprocess_depth(depth)
        current_rgb = preprocess_image(rgb)
        camera_points, camera_colors = get_pointcloud_from_depth(current_rgb, current_depth, self.camera_intrinsic)
        world_points = translate_to_world(camera_points, current_position, current_rotation)
        current_pcd = gpu_pointcloud_from_array(world_points, camera_colors, self.pcd_device).voxel_down_sample(
            self.pcd_resolution)
        return current_pcd

    def get_object_entities(self, depth, classes, masks, confidences):
        entities = []
        exist_objects = np.unique([ent['class'] for ent in self.object_entities]).tolist()
        for cls, mask, score in zip(classes, masks, confidences):
            if depth[mask > 0].min() < 1.0 and score < 0.5:
                continue
            if cls not in exist_objects:
                exist_objects.append(cls)
            camera_points = get_pointcloud_from_depth_mask(depth, mask, self.camera_intrinsic)
            world_points = translate_to_world(camera_points, self.current_position, self.current_rotation)
            point_colors = np.array([d3_40_colors_rgb[exist_objects.index(cls) % 40]] * world_points.shape[0])
            if world_points.shape[0] < 10:
                continue
            object_pcd = gpu_pointcloud_from_array(world_points, point_colors, self.pcd_device).voxel_down_sample(
                self.pcd_resolution)
            object_pcd = gpu_cluster_filter(object_pcd)
            if object_pcd.point.positions.shape[0] < 10:
                continue
            entity = {'class': cls, 'pcd': object_pcd, 'confidence': score}
            entities.append(entity)
        return entities

    def associate_object_entities(self, ref_entities, eval_entities):
        entity_indices = []
        for entity in eval_entities:
            if len(ref_entities) == 0:
                ref_entities.append(entity)
                entity_indices.append(0)
                continue
            overlap_score = []
            eval_pcd = entity['pcd']
            for ref_entity in ref_entities:
                if eval_pcd.point.positions.shape[0] == 0:
                    break
                cdist = pointcloud_distance(eval_pcd,ref_entity['pcd'])
                overlap_condition = (cdist < 0.1)
                nonoverlap_condition = overlap_condition.logical_not()
                eval_pcd = eval_pcd.select_by_index(o3d.core.Tensor(nonoverlap_condition.cpu().numpy(),device=self.pcd_device).nonzero()[0])
                overlap_score.append((overlap_condition.sum()/(overlap_condition.shape[0]+1e-6)).cpu().numpy())
            max_overlap_score = np.max(overlap_score)
            arg_overlap_index = np.argmax(overlap_score)
            if max_overlap_score < 0.25:
                entity['pcd'] = eval_pcd
                ref_entities.append(entity)
                entity_indices.append(len(ref_entities)-1)
            else:
                argmax_entity = ref_entities[arg_overlap_index]
                argmax_entity['pcd'] = gpu_merge_pointcloud(argmax_entity['pcd'],eval_pcd)
                if argmax_entity['pcd'].point.positions.shape[0] < entity['pcd'].point.positions.shape[0] or entity['class'] in INTEREST_OBJECTS:
                    argmax_entity['class'] = entity['class']
                ref_entities[arg_overlap_index] = argmax_entity
                entity_indices.append(arg_overlap_index)
        return ref_entities, entity_indices

    def get_obstacle_affordance(self):
        try:
            distance = pointcloud_distance(self.navigable_pcd, self.obstacle_pcd)
            affordance = (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
            affordance[distance < 0.25] = 0
            return affordance.cpu().numpy()
        except:
            return np.zeros((self.navigable_pcd.point.positions.shape[0],), dtype=np.float32)

    def get_trajectory_affordance(self):
        try:
            distance = pointcloud_distance(self.navigable_pcd, self.trajectory_pcd)
            affordance = (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
            return affordance.cpu().numpy()
        except:
            return np.zeros((self.navigable_pcd.point.positions.shape[0],), dtype=np.float32)

    def get_semantic_affordance(self, target_class, threshold=0.1):
        semantic_pointcloud = o3d.t.geometry.PointCloud()
        for entity in self.object_entities:
            if entity['class'] in target_class:
                semantic_pointcloud = gpu_merge_pointcloud(semantic_pointcloud, entity['pcd'])
        try:
            distance = pointcloud_2d_distance(self.navigable_pcd, semantic_pointcloud)
            affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
            affordance[distance > threshold] = 0
            affordance = affordance.cpu().numpy()
            return affordance
        except:
            return np.zeros((self.navigable_pcd.point.positions.shape[0],), dtype=np.float32)

    def get_gpt4v_affordance(self, gpt4v_pcd):
        try:
            distance = pointcloud_distance(self.navigable_pcd, gpt4v_pcd)
            affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
            affordance[distance > 0.1] = 0
            return affordance.cpu().numpy()
        except:
            return np.zeros((self.navigable_pcd.point.positions.shape[0],), dtype=np.float32)

    def get_action_affordance(self, action):
        try:
            if action == 'Explore':
                distance = pointcloud_2d_distance(self.navigable_pcd, self.frontier_pcd)
                affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
                affordance[distance > 0.2] = 0
                return affordance.cpu().numpy()
            elif action == 'Move_Forward':
                pixel_x, pixel_z, depth_values = project_to_camera(self.navigable_pcd, self.camera_intrinsic,
                                                                   self.current_position, self.current_rotation)
                filter_condition = (pixel_x >= 0) & (pixel_x < self.camera_intrinsic[0][2] * 2) & (pixel_z >= 0) & (
                            pixel_z < self.camera_intrinsic[1][2] * 2) & (depth_values > 1.5) & (depth_values < 2.5)
                filter_pcd = self.navigable_pcd.select_by_index(
                    o3d.core.Tensor(np.where(filter_condition == 1)[0], device=self.navigable_pcd.device))
                distance = pointcloud_distance(self.navigable_pcd, filter_pcd)
                affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
                affordance[distance > 0.1] = 0
                return affordance.cpu().numpy()
            elif action == 'Turn_Around':
                R = np.array([np.pi, np.pi, np.pi]) * self.rotate_axis
                turn_extrinsic = np.matmul(self.current_rotation,
                                           quaternion.as_rotation_matrix(quaternion.from_euler_angles(R)))
                pixel_x, pixel_z, depth_values = project_to_camera(self.navigable_pcd, self.camera_intrinsic,
                                                                   self.current_position, turn_extrinsic)
                filter_condition = (pixel_x >= 0) & (pixel_x < self.camera_intrinsic[0][2] * 2) & (pixel_z >= 0) & (
                            pixel_z < self.camera_intrinsic[1][2] * 2) & (depth_values > 1.5) & (depth_values < 2.5)
                filter_pcd = self.navigable_pcd.select_by_index(
                    o3d.core.Tensor(np.where(filter_condition == 1)[0], device=self.navigable_pcd.device))
                distance = pointcloud_distance(self.navigable_pcd, filter_pcd)
                affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
                affordance[distance > 0.1] = 0
                return affordance.cpu().numpy()
            elif action == 'Turn_Left':
                R = np.array([np.pi / 2, np.pi / 2, np.pi / 2]) * self.rotate_axis
                turn_extrinsic = np.matmul(self.current_rotation,
                                           quaternion.as_rotation_matrix(quaternion.from_euler_angles(R)))
                pixel_x, pixel_z, depth_values = project_to_camera(self.navigable_pcd, self.camera_intrinsic,
                                                                   self.current_position, turn_extrinsic)
                filter_condition = (pixel_x >= 0) & (pixel_x < self.camera_intrinsic[0][2] * 2) & (pixel_z >= 0) & (
                            pixel_z < self.camera_intrinsic[1][2] * 2) & (depth_values > 1.5) & (depth_values < 2.5)
                filter_pcd = self.navigable_pcd.select_by_index(
                    o3d.core.Tensor(np.where(filter_condition == 1)[0], device=self.navigable_pcd.device))
                distance = pointcloud_distance(self.navigable_pcd, filter_pcd)
                affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
                affordance[distance > 0.1] = 0
                return affordance.cpu().numpy()
            elif action == 'Turn_Right':
                R = np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2]) * self.rotate_axis
                turn_extrinsic = np.matmul(self.current_rotation,
                                           quaternion.as_rotation_matrix(quaternion.from_euler_angles(R)))
                pixel_x, pixel_z, depth_values = project_to_camera(self.navigable_pcd, self.camera_intrinsic,
                                                                   self.current_position, turn_extrinsic)
                filter_condition = (pixel_x >= 0) & (pixel_x < self.camera_intrinsic[0][2] * 2) & (pixel_z >= 0) & (
                            pixel_z < self.camera_intrinsic[1][2] * 2) & (depth_values > 1.5) & (depth_values < 2.5)
                filter_pcd = self.navigable_pcd.select_by_index(
                    o3d.core.Tensor(np.where(filter_condition == 1)[0], device=self.navigable_pcd.device))
                distance = pointcloud_distance(self.navigable_pcd, filter_pcd)
                affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
                affordance[distance > 0.1] = 0
                return affordance.cpu().numpy()
            elif action == 'Enter':
                return self.get_semantic_affordance(['doorway', 'door', 'entrance', 'exit'])
            elif action == 'Exit':
                return self.get_semantic_affordance(['doorway', 'door', 'entrance', 'exit'])
            else:
                return np.zeros((self.navigable_pcd.point.positions.shape[0],), dtype=np.float32)
        except:
            return np.zeros((self.navigable_pcd.point.positions.shape[0],), dtype=np.float32)

    def get_objnav_affordance_map(self, action, target_class, gpt4v_pcd, complete_flag=False, failure_mode=False):
        if failure_mode:
            obstacle_affordance = self.get_obstacle_affordance()
            affordance = self.get_action_affordance('Explore')
            affordance = np.clip(affordance, 0.1, 1.0)
            affordance[obstacle_affordance == 0] = 0
            return affordance, self.visualize_affordance(affordance)
        elif complete_flag:
            affordance = self.get_semantic_affordance([target_class], threshold=0.1)
            return affordance, self.visualize_affordance(affordance)
        else:
            obstacle_affordance = self.get_obstacle_affordance()
            semantic_affordance = self.get_semantic_affordance([target_class], threshold=1.5)
            action_affordance = self.get_action_affordance(action)
            gpt4v_affordance = self.get_gpt4v_affordance(gpt4v_pcd)
            history_affordance = self.get_trajectory_affordance()
            affordance = 0.25 * semantic_affordance + 0.25 * action_affordance + 0.25 * gpt4v_affordance + 0.25 * history_affordance
            affordance = np.clip(affordance, 0.1, 1.0)
            affordance[obstacle_affordance == 0] = 0
            return affordance, self.visualize_affordance(affordance / (affordance.max() + 1e-6))

    def get_debug_affordance_map(self, action, target_class, gpt4v_pcd):
        obstacle_affordance = self.get_obstacle_affordance()
        semantic_affordance = self.get_semantic_affordance([target_class], threshold=1.5)
        action_affordance = self.get_action_affordance(action)
        gpt4v_affordance = self.get_gpt4v_affordance(gpt4v_pcd)
        history_affordance = self.get_trajectory_affordance()
        return self.visualize_affordance(semantic_affordance / (semantic_affordance.max() + 1e-6)), \
            self.visualize_affordance(history_affordance / (history_affordance.max() + 1e-6)), \
            self.visualize_affordance(action_affordance / (action_affordance.max() + 1e-6)), \
            self.visualize_affordance(gpt4v_affordance / (gpt4v_affordance.max() + 1e-6)), \
            self.visualize_affordance(obstacle_affordance / (obstacle_affordance.max() + 1e-6))

    def visualize_affordance(self, affordance):
        cmap = colormaps.get('jet')
        color_affordance = cmap(affordance)[:, 0:3]
        color_affordance = cpu_pointcloud_from_array(self.navigable_pcd.point.positions.cpu().numpy(), color_affordance)
        return color_affordance

    def get_appeared_objects(self):
        return [entity['class'] for entity in self.object_entities]

    def save_pointcloud_debug(self, path="./"):
        save_pcd = o3d.geometry.PointCloud()
        try:
            assert self.useful_pcd.point.positions.shape[0] > 0
            save_pcd.points = o3d.utility.Vector3dVector(self.useful_pcd.point.positions.cpu().numpy())
            save_pcd.colors = o3d.utility.Vector3dVector(self.useful_pcd.point.colors.cpu().numpy())
            o3d.io.write_point_cloud(path + "scene.ply", save_pcd)
        except:
            pass
        try:
            assert self.navigable_pcd.point.positions.shape[0] > 0
            save_pcd.points = o3d.utility.Vector3dVector(self.navigable_pcd.point.positions.cpu().numpy())
            save_pcd.colors = o3d.utility.Vector3dVector(self.navigable_pcd.point.colors.cpu().numpy())
            o3d.io.write_point_cloud(path + "navigable.ply", save_pcd)
        except:
            pass
        try:
            assert self.obstacle_pcd.point.positions.shape[0] > 0
            save_pcd.points = o3d.utility.Vector3dVector(self.obstacle_pcd.point.positions.cpu().numpy())
            save_pcd.colors = o3d.utility.Vector3dVector(self.obstacle_pcd.point.colors.cpu().numpy())
            o3d.io.write_point_cloud(path + "obstacle.ply",save_pcd)
        except:
            pass

        object_pcd = o3d.geometry.PointCloud()
        for entity in self.object_entities:
            points = entity['pcd'].point.positions.cpu().numpy()
            colors = entity['pcd'].point.colors.cpu().numpy()
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(points)
            new_pcd.colors = o3d.utility.Vector3dVector(colors)
            object_pcd = object_pcd + new_pcd
        if len(object_pcd.points) > 0:
            o3d.io.write_point_cloud(path + "object.ply",object_pcd)

        # save the nodes
        center_spheres = []
        centers = self.representation.get_nodes_positions()
        for center in centers:
            center[2] = self.floor_height
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.paint_uniform_color([1, 0, 0])
            sphere.translate(center)
            center_spheres.append(sphere)
        # save the center_spheres
        combined_sphere = o3d.geometry.TriangleMesh()
        for sphere in center_spheres:
            combined_sphere += sphere
        o3d.io.write_triangle_mesh(path + "center_spheres.ply", combined_sphere)

        # save nodes with object
        nwpcd = object_pcd
        combined_sphere_pcd = o3d.geometry.PointCloud()
        combined_sphere_pcd.points = combined_sphere.vertices
        combined_sphere_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(combined_sphere.vertices) * np.array([1, 0, 0]))

        nwpcd = nwpcd + combined_sphere_pcd
        o3d.io.write_point_cloud(path + "combined_scene.ply", nwpcd)

        obj_centers = []
        for i in range(len(self.object_entities)):
            entity = self.object_entities[i]
            points = entity['pcd'].point.positions.cpu().numpy()
            pos = np.mean(points, axis=0)
            obj_centers.append(pos)
        num_obj = len(obj_centers)

        for i in range(self.representation.node_cnt):
            center = centers[i]
            center[2] = self.floor_height
            obj_centers.append(center)

        obj_centers = np.array(obj_centers)

        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(obj_centers)

        edges = []

        for i in range(self.representation.node_cnt):
            node = self.representation.nodes[i]
            center = node.position
            print("Node: ", i, " Center: ", center)
            for ind in node.objects:
                obj = self.object_entities[ind]
                print("Object: ", obj['class'])
                edges.append([ind, i + num_obj])

        lineset.lines = o3d.utility.Vector2iVector(edges)
        lineset.colors = o3d.utility.Vector3dVector(np.ones((len(edges), 3)) * np.array([1, 0, 1]))

        # visualize the edges with nwpcd
        o3d.io.write_line_set(path + "edges.ply", lineset)


        # save the self.nodes in the txt file
        np.savetxt(path + "nodes.txt", np.array(centers), fmt='%f')

        edges = self.representation.get_edges()

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(centers)
        line_set.lines = o3d.utility.Vector2iVector(edges)
        line_set.paint_uniform_color([1, 0, 0])

        o3d.io.write_line_set(path + "graph.ply", line_set)



    def is_out_of_boundary(self, point_cloud, point, radius=0.1, density_threshold=1):
        tree = KDTree(point_cloud)

        indices = tree.query_ball_point(point, radius)

        local_density = len(indices) - 1  # 减去自身

        if local_density < density_threshold:
            return True
        else:
            return False

    def is_out_of_boundary_nav(self, nav_points, obstacle_points, radius=0.06, density_threshold=1):
        radius = radius
        density_threshold = density_threshold
        # 创建KD树以加速邻域搜索
        tree = KDTree(obstacle_points)

        points = []
        for point in nav_points:
            # 查询半径内的点
            indices = tree.query_ball_point(point, radius)
            # 统计邻域内点的数量
            local_density = len(indices) - 1
            if local_density < density_threshold:
                points.append(point)

        return np.asarray(points)

    def get_nodes(self, current_pcd, idx, step):
        def calculate_intersections(point_cloud, current_position, num_rays=72, max_distance=2.5):
            intersections = []
            angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

            for angle in angles:
                direction = np.array([np.cos(angle), np.sin(angle)])
                delta = 200
                for idx, d in enumerate(np.linspace(0, max_distance, delta)):  # 逐渐增大距离
                    point = current_position + d * direction

                    if self.is_out_of_boundary(point_cloud, point):
                        # if self.is_out_of_boundary(point_cloud, point) or idx == delta - 1:
                        intersections.append(point)
                        break

            return intersections

        floor_height = self.floor_height

        obstcale_pcd_points = self.obstacle_pcd.point.positions.cpu().numpy()

        # only for interpolation
        current_navigable_pcd = current_pcd.select_by_index(
            (current_pcd.point.positions[:, 2] < self.floor_height).nonzero()[0])
        current_position = self.current_position

        current_navigable_position = current_navigable_pcd.point.positions.cpu().numpy()

        # check the visibility of the nodes in the graph
        if step!=12:
            self.representation.update_edges(current_navigable_position, self.current_position)

        # change the z value of the points to the mean of the floor points
        obstcale_pcd_points[:, 2] = np.ones_like(obstcale_pcd_points[:, 2]) * np.mean(current_navigable_position[:, 2])

        standing_position = np.array(
            [current_position[0], current_position[1], current_navigable_position[:, 2].mean()])
        # get the closet point and distance to the current position
        distance = np.linalg.norm(current_navigable_position - standing_position, axis=1)
        index = np.argmin(distance)
        closest_distance = distance[index]
        index_farthest = np.argmax(distance)
        farthest_distance = distance[index_farthest]
        print(f"Closest point index: {index}, distance: {distance[index]}")
        print(f"Farthest point index: {index_farthest}, distance: {distance[index_farthest]}")

        closest_points = current_navigable_position[distance < closest_distance + 0.8]

        interpolate_points = np.linspace(np.ones_like(closest_points) * standing_position, closest_points, 25).reshape(
            -1, 3)
        interpolate_points = interpolate_points[
            (interpolate_points[:, 2] > floor_height - 0.2) & (interpolate_points[:, 2] < floor_height + 0.2)]

        # merge current_navigable_position and interpolate points

        current_navigable_pcd = self.process_obs_pcd.select_by_index(
                                            (self.process_obs_pcd.point.positions[:, 2] < self.floor_height).nonzero()[0])
        current_navigable_position = current_navigable_pcd.point.positions.cpu().numpy()
        current_navigable_position = np.concatenate((current_navigable_position, interpolate_points), axis=0)
        current_navigable_position[:, 2] = np.ones_like(current_navigable_position[:, 2]) * np.mean(current_navigable_position[:, 2])

        current_navigable_colors = np.ones_like(current_navigable_position) * 100
        current_navigable_pcd = gpu_pointcloud_from_array(current_navigable_position, current_navigable_colors,
                                                          self.pcd_device).voxel_down_sample(self.pcd_resolution)

        current_navigable_position = current_navigable_pcd.point.positions.cpu().numpy()

        # get the frontier_clusters

        # self-implemented function
        # frontier_clusters, frontier_centers = self.get_frontiers(current_navigable_pcd, current_pcd.point.positions.cpu().numpy())

        # offered function
        if step == 12:
            frontier_clusters, frontier_centers = self.get_frontiers_offerd(self.obstacle_pcd, self.navigable_pcd,
                                                                self.floor_height + 0.2, 0.1, closest_distance)
        else:
            frontier_clusters_whole, frontier_centers_whole = self.get_frontiers_offerd(self.obstacle_pcd, self.navigable_pcd,
                                                                            self.floor_height + 0.2, 0.1,0.1)

            # extract frontiers in current observation
            frontier_clusters = []
            frontier_centers = []
            for frontier_index, frontier_cluster in enumerate(frontier_clusters_whole):
                distances = np.linalg.norm(frontier_cluster[:, np.newaxis, :2] - current_navigable_position[np.newaxis, :, :2],
                                           axis=2)
                if np.min(distances) < 0.2:
                    frontier_cluster_new = frontier_cluster[np.min(distances, axis=1) < 0.2]
                    frontier_center_new = np.mean(frontier_cluster_new, axis=0)
                    frontier_clusters.append(frontier_cluster_new)
                    frontier_centers.append(frontier_center_new)

        if len(frontier_clusters) != 0:
            # print('Frontier clusters', (frontier_clusters))
            frontiers_to_save = np.concatenate(frontier_clusters, axis=0)
            save_pcd = o3d.geometry.PointCloud()
            save_pcd.points = o3d.utility.Vector3dVector(frontiers_to_save)
            save_pcd.paint_uniform_color([0, 1, 0])
            import os
            os.makedirs(f'tmp_with_whole_obs/episode-{idx}', exist_ok=True)
            o3d.io.write_point_cloud(f'tmp_with_whole_obs/episode-{idx}/frontier_{step}.ply', save_pcd)

        print("Current position:", standing_position)
        # print(len(current_navigable_position))

        # decide the max distance
        if len(frontier_clusters) != 0:
            frontiers_all = np.concatenate(frontier_clusters, axis=0)
            distance_to_frontiers = np.linalg.norm(frontiers_all - standing_position, axis=1)
            min_distance_to_frontiers = np.min(distance_to_frontiers)
            print(f"Min distance to frontiers: {min_distance_to_frontiers}")
            max_distance = min(2.5, min_distance_to_frontiers * 1.1)
        else:
            max_distance = 2.5

        print(f"Max distance Threshold: {max_distance}")
        intersections = calculate_intersections(current_navigable_position[:, :2], standing_position[:2],
                                                max_distance=max_distance)

        distance_inter = np.linalg.norm(intersections - standing_position[:2], axis=1)
        print(f'Closest intersection distance: {np.min(distance_inter)}')
        # print("Intersections:", intersections)

        intersections = np.array(intersections)
        # extand the dimention of the intersections to 3
        intersections = np.concatenate(
            (intersections, np.ones((intersections.shape[0], 1)) * np.mean(current_navigable_position[:, 2])),
            axis=1)
        # keep the intersections that are away from the current state above a threshold
        distance_inter = np.linalg.norm(intersections - standing_position, axis=1)
        intersections = intersections[distance_inter > 0.2]

        polygon_path = Path(intersections[:, :2])
        # Remove points inside the polygon
        points = current_navigable_position[:, :2]
        mask = 1 - polygon_path.contains_points(points)

        # Ensure the mask is a tensor on the correct device
        mask_tensor = o3d.core.Tensor(np.where(mask)[0], o3d.core.Dtype.Int64, device=current_navigable_pcd.device)

        # Select points by index
        pcd_removed = current_navigable_pcd.select_by_index(mask_tensor)

        mask_tensor_in_circle = o3d.core.Tensor(np.where(1 - mask)[0], o3d.core.Dtype.Int64,
                                                device=current_navigable_pcd.device)
        pcd_in_circle = current_navigable_pcd.select_by_index(mask_tensor_in_circle)
        colors = np.random.rand(3)
        pcd_in_circle.paint_uniform_color(colors)

        # cluster the remaing points
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd_removed.cluster_dbscan(eps=0.1, min_points=4, print_progress=False))

        # print(labels)
        if step != 12 and len(labels) == 0:
            return
        max_label = labels.max().cpu().numpy()
        # print(f"point cloud has {max_label + 1} clusters")

        # extract each cluster and rule out the small ones caluclate the center of each cluster
        clusters = []
        centers_from_frontiers = []
        centers_from_clusters = []
        for i in range(max_label + 1):
            mask_idx_tensor = o3d.core.Tensor((labels == i).nonzero()[0], o3d.core.Dtype.Int64,
                                              device=current_navigable_pcd.device)
            cluster = pcd_removed.select_by_index(mask_idx_tensor)
            cluster_point = cluster.point.positions.cpu().numpy()
            # generate random colors, shape is (n, 3)
            colors = np.random.rand(3)
            cluster.paint_uniform_color(colors)

            # extract frontiers in this cluster
            current_frontier_clusters = []
            current_frontier_centers = []
            for frontier_index, frontier_cluster in enumerate(frontier_clusters):
                distances = np.linalg.norm(frontier_cluster[:, np.newaxis, :2] - cluster_point[np.newaxis, :, :2],
                                           axis=2)
                if np.min(distances) < 0.2:
                    frontier_cluster_new = frontier_cluster[np.min(distances, axis=1) < 0.2]
                    frontier_center_new = np.mean(frontier_cluster_new, axis=0)
                    current_frontier_clusters.append(frontier_cluster_new)
                    current_frontier_centers.append(frontier_center_new)

            if len(current_frontier_clusters) != 0 and len(cluster_point) > 20:
                clusters.append(cluster)
                # merge the frontier clusters
                merged_frontier_clusters, merged_frontier_centers, line_set = self.merge_frontier_with_visibility_1(cluster_point,
                                                                                                          obstcale_pcd_points,
                                                                                                          current_frontier_clusters,
                                                                                                          current_frontier_centers)

                o3d.io.write_line_set(f'tmp_with_whole_obs/episode-{idx}/line_set_{step}_cluster_{i}.ply', line_set)

                center_idx_to_remove = []
                for index1 in range(len(merged_frontier_centers)):
                    # check every 2 centers in the center, if their angle is less than 5 degree, then only keep the closest one
                    for index2 in range(index1 + 1, len(merged_frontier_centers)):
                        center1 = merged_frontier_centers[index1]
                        center2 = merged_frontier_centers[index2]
                        angle1 = np.arctan2(center1[1] - standing_position[1], center1[0] - standing_position[0])
                        angle2 = np.arctan2(center2[1] - standing_position[1], center2[0] - standing_position[0])
                        angle_diff = np.abs(angle1 - angle2)
                        if angle_diff < 1.5 * np.pi / 36:
                            distance1 = np.linalg.norm(center1[:2] - standing_position[:2])
                            distance2 = np.linalg.norm(center2[:2] - standing_position[:2])
                            if distance1 < distance2:
                                center_idx_to_remove.append(index2)
                            else:
                                center_idx_to_remove.append(index1)
                center_idx_to_remove = list(set(center_idx_to_remove))
                centers_new = [center for idx, center in enumerate(merged_frontier_centers) if
                               idx not in center_idx_to_remove]
                for center in centers_new:
                    centers_from_frontiers.append(center)
            else:
                if len(cluster.point.positions.cpu().numpy()) > 100:
                    clusters.append(cluster)
                    center = np.mean(cluster.point.positions.cpu().numpy(), axis=0)
                    centers_from_clusters.append(center)

        current_pcd_to_save = o3d.t.geometry.PointCloud(self.pcd_device)
        current_pcd_to_save = gpu_merge_pointcloud(current_pcd_to_save, pcd_in_circle)
        for cluster in clusters:
            current_pcd_to_save = gpu_merge_pointcloud(current_pcd_to_save, cluster)
        save_pcd = o3d.geometry.PointCloud()
        save_pcd.points = o3d.utility.Vector3dVector(current_pcd_to_save.point.positions.cpu().numpy())
        save_pcd.colors = o3d.utility.Vector3dVector(current_pcd_to_save.point.colors.cpu().numpy())
        import os
        os.makedirs(f'tmp_with_whole_obs/episode-{idx}', exist_ok=True)
        o3d.io.write_point_cloud(f'tmp_with_whole_obs/episode-{idx}/nav_{step}.ply', save_pcd)

        # for cluster in clusters:
        #     print(len(cluster.point.positions.cpu().numpy()))

        # visualize the clusters
        # o3d.visualization.draw_geometries(clusters)

        # compute the distance between the current centers and self.nodes,
        # if the distance is less than 0.5, then remove the center
        # and add the new center to self.nodes
        centers_from_frontiers = np.array(centers_from_frontiers)
        centers_from_clusters = np.array(centers_from_clusters)
        valid_centers = []
        if len(self.representation.nodes) == 0:
            # in case the centers is empty
            print(len(centers_from_frontiers)+len(centers_from_clusters))
            if len(centers_from_frontiers)+len(centers_from_clusters) == 0:
                # offered function
                frontier_clusters, frontier_centers = (
                    self.get_frontiers_offerd(obstcale_pcd, current_navigable_pcd, self.floor_height + 0.2, 0.1,
                                              closest_distance * 0.7))

                if len(frontier_clusters) != 0:
                    # print('Frontier clusters', (frontier_clusters))
                    frontiers_to_save = np.concatenate(frontier_clusters, axis=0)
                    save_pcd = o3d.geometry.PointCloud()
                    save_pcd.points = o3d.utility.Vector3dVector(frontiers_to_save)
                    save_pcd.paint_uniform_color([0, 1, 0])
                    import os
                    os.makedirs(f'tmp_with_whole_obs/episode-{idx}', exist_ok=True)
                    o3d.io.write_point_cloud(f'tmp_with_whole_obs/episode-{idx}/frontier_{step}.ply', save_pcd)

                # decide the max distance
                if len(frontier_clusters) != 0:
                    frontiers_all = np.concatenate(frontier_clusters, axis=0)
                    distance_to_frontiers = np.linalg.norm(frontiers_all - standing_position, axis=1)
                    min_distance_to_frontiers = np.min(distance_to_frontiers)
                    print(f"Min distance to frontiers: {min_distance_to_frontiers}")
                    max_distance = min(2.5, min_distance_to_frontiers * 1.1)
                else:
                    max_distance = 2.5

                print(f"Max distance Threshold: {max_distance}")
                intersections = calculate_intersections(current_navigable_position[:, :2], standing_position[:2],
                                                        num_rays=180, max_distance=max_distance)

                intersections = np.array(intersections)
                # extand the dimention of the intersections to 3
                intersections = np.concatenate(
                    (intersections, np.ones((intersections.shape[0], 1)) * np.mean(current_navigable_position[:, 2])),
                    axis=1)
                # keep the intersections that are away from the current state above a threshold
                distance_inter = np.linalg.norm(intersections - standing_position, axis=1)
                intersections = intersections[distance_inter > 0.2]

                polygon_path = Path(intersections[:, :2])
                # Remove points inside the polygon
                points = current_navigable_position[:, :2]
                mask = 1 - polygon_path.contains_points(points)

                # Ensure the mask is a tensor on the correct device
                mask_tensor = o3d.core.Tensor(np.where(mask)[0], o3d.core.Dtype.Int64,
                                              device=current_navigable_pcd.device)

                # Select points by index
                pcd_removed = current_navigable_pcd.select_by_index(mask_tensor)

                mask_tensor_in_circle = o3d.core.Tensor(np.where(1 - mask)[0], o3d.core.Dtype.Int64,
                                                        device=current_navigable_pcd.device)
                pcd_in_circle = current_navigable_pcd.select_by_index(mask_tensor_in_circle)
                colors = np.random.rand(3)
                pcd_in_circle.paint_uniform_color(colors)

                # cluster the remaing points
                with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                    labels = np.array(pcd_removed.cluster_dbscan(eps=0.1, min_points=4, print_progress=False))

                max_label = labels.max().cpu().numpy()
                # print(f"point cloud has {max_label + 1} clusters")

                # extract each cluster and rule out the small ones caluclate the center of each cluster
                clusters = []
                centers_from_frontiers = []
                centers_from_clusters = []
                for i in range(max_label + 1):
                    mask_idx_tensor = o3d.core.Tensor((labels == i).nonzero()[0], o3d.core.Dtype.Int64,
                                                      device=current_navigable_pcd.device)
                    cluster = pcd_removed.select_by_index(mask_idx_tensor)
                    cluster_point = cluster.point.positions.cpu().numpy()
                    # generate random colors, shape is (n, 3)
                    colors = np.random.rand(3)
                    cluster.paint_uniform_color(colors)

                    # extract frontiers in this cluster
                    current_frontier_clusters = []
                    current_frontier_centers = []
                    for frontier_index, frontier_cluster in enumerate(frontier_clusters):
                        distances = np.linalg.norm(
                            frontier_cluster[:, np.newaxis, :2] - cluster_point[np.newaxis, :, :2],
                            axis=2)
                        if np.min(distances) < 0.2:
                            frontier_cluster_new = frontier_cluster[np.min(distances, axis=1) < 0.2]
                            frontier_center_new = np.mean(frontier_cluster_new, axis=0)
                            current_frontier_clusters.append(frontier_cluster_new)
                            current_frontier_centers.append(frontier_center_new)

                    if len(current_frontier_clusters) != 0 and len(cluster_point) > 20:
                        clusters.append(cluster)
                        # merge the frontier clusters
                        merged_frontier_clusters, merged_frontier_centers, line_set = self.merge_frontier_with_visibility_1(
                            cluster_point,
                            obstcale_pcd_points,
                            current_frontier_clusters,
                            current_frontier_centers)

                        center_idx_to_remove = []
                        for index1 in range(len(merged_frontier_centers)):
                            # check every 2 centers in the center, if their angle is less than 5 degree, then only keep the closest one
                            for index2 in range(index1 + 1, len(merged_frontier_centers)):
                                center1 = merged_frontier_centers[index1]
                                center2 = merged_frontier_centers[index2]
                                angle1 = np.arctan2(center1[1] - standing_position[1],
                                                    center1[0] - standing_position[0])
                                angle2 = np.arctan2(center2[1] - standing_position[1],
                                                    center2[0] - standing_position[0])
                                angle_diff = np.abs(angle1 - angle2)
                                if angle_diff < 1.5 * np.pi / 36:
                                    distance1 = np.linalg.norm(center1 - standing_position)
                                    distance2 = np.linalg.norm(center2 - standing_position)
                                    if distance1 < distance2:
                                        center_idx_to_remove.append(index2)
                                    else:
                                        center_idx_to_remove.append(index1)
                        center_idx_to_remove = list(set(center_idx_to_remove))
                        centers_new = [center for idx, center in enumerate(merged_frontier_centers) if
                                       idx not in center_idx_to_remove]
                        for center in centers_new:
                            centers_from_frontiers.append(center)
                    else:
                        if len(cluster.point.positions.cpu().numpy()) > 10:
                            clusters.append(cluster)
                            center = np.mean(cluster.point.positions.cpu().numpy(), axis=0)
                            centers_from_frontiers.append(center)

                    current_pcd_to_save = o3d.t.geometry.PointCloud(self.pcd_device)
                    current_pcd_to_save = gpu_merge_pointcloud(current_pcd_to_save, pcd_in_circle)
                    for cluster in clusters:
                        current_pcd_to_save = gpu_merge_pointcloud(current_pcd_to_save, cluster)
                    save_pcd = o3d.geometry.PointCloud()
                    save_pcd.points = o3d.utility.Vector3dVector(current_pcd_to_save.point.positions.cpu().numpy())
                    save_pcd.colors = o3d.utility.Vector3dVector(current_pcd_to_save.point.colors.cpu().numpy())
                    import os
                    os.makedirs(f'tmp_with_whole_obs/episode-{idx}', exist_ok=True)
                    o3d.io.write_point_cloud(f'tmp_with_whole_obs/episode-{idx}/nav_{step}.ply', save_pcd)

            # # calculate the distance between the current position and the centers
            # for center in centers:
            #     distance = np.linalg.norm(self.current_position - center)
            #     if distance > max_distance:
            #         # interploate the points between the current position and the center
            #         interpolate_points = np.linspace(self.current_position, center, 1).reshape(-1, 3)
            #         centers = np.concatenate((centers, interpolate_points), axis=0)

            self.representation.add_node(self.current_position)
            self.representation.visit_node(self.current_position)
            # valid_centers.append(self.current_position)

            for center in centers_from_frontiers:
                center[2] = self.current_position[2]
                valid_centers.append(center)

            for center in centers_from_clusters:
                center[2] = self.current_position[2]
                valid_centers.append(center)

            valid_centers = np.array(valid_centers)

            for center in valid_centers:
                self.representation.add_node(center)
                self.representation.add_edge(self.current_position, center)

            # self.nodes_state = np.concatenate((np.array([1]), np.zeros(len(valid_centers)-1)), axis=0)

        else:
            nodes_positions = self.representation.get_nodes_positions()
            # print(f'Nodes positions: {nodes_positions}')
            for center in centers_from_frontiers:
                distance = np.linalg.norm(nodes_positions[:, :2] - center[:2], axis=1)
                # print(f'Center: {center}, min distance: {np.min(distance)}, threshold: {closest_distance*0.7}')
                if np.min(distance) > closest_distance * 0.7:
                    center[2] = self.current_position[2]
                    valid_centers.append(center)

            for center in centers_from_clusters:
                distance = np.linalg.norm(nodes_positions[:, :2] - center[:2], axis=1)
                # print(f'Center: {center}, min distance: {np.min(distance)}, threshold: {closest_distance*0.7}')
                if np.min(distance) > closest_distance * 0.7:
                    center[2] = self.current_position[2]
                    valid_centers.append(center)
            valid_centers = np.array(valid_centers)

            # for center in valid_centers:
            #     distance = np.linalg.norm(self.current_position - center)
            #     if distance > max_distance:
            #         # interploate the points between the current position and the center
            #         interpolate_points = np.linspace(self.current_position, center, 1).reshape(-1, 3)
            #         valid_centers = np.concatenate((valid_centers, interpolate_points), axis=0)

            # print(self.nodes)
            # print(valid_centers)
            if len(valid_centers) > 0:
                for center in valid_centers:
                    self.representation.add_node(center)
                    self.representation.add_edge(self.current_position, center)

                # self.nodes = np.concatenate((self.nodes, np.asarray(valid_centers)), axis=0)
                # # update the nodes_state
                # self.nodes_state = np.concatenate((self.nodes_state, np.zeros(len(valid_centers))), axis=0)

            # nodes_positions = self.representation.get_nodes_positions()
            # print(nodes_positions)

        # save the nodes
        if len(valid_centers) > 0:
            center_spheres = []
            for center in valid_centers:
                center[2] = floor_height
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                sphere.paint_uniform_color([1, 0, 0])
                sphere.translate(center)
                center_spheres.append(sphere)
            # save the center_spheres
            combined_sphere = o3d.geometry.TriangleMesh()
            for sphere in center_spheres:
                combined_sphere += sphere
            o3d.io.write_triangle_mesh(f'tmp_with_whole_obs/episode-{idx}/centers_{step}.ply', combined_sphere)


    def get_frontiers(self, current_navigable_pcd, real_boundary):
        def is_out_of_boundary_frontier(point_cloud, radius=0.1, density_threshold=1):
            # 这里假设有一个函数，检查点是否在点云的边界外
            # 例如可以使用点云的凸包或距离来判断
            radius = radius
            density_threshold = density_threshold
            # 创建KD树以加速邻域搜索
            tree = KDTree(point_cloud)

            boundary_points = []
            for point in point_cloud:
                # 查询半径内的点
                indices = tree.query_ball_point(point, radius)
                # 统计邻域内点的数量
                local_density = len(indices) - 1
                if local_density < density_threshold:
                    boundary_points.append(point)

            return boundary_points

        current_position = self.current_position[:2]
        current_navigable_position = current_navigable_pcd.point.positions.cpu().numpy()

        boundary_points = is_out_of_boundary_frontier(current_navigable_position[:, :2], radius=0.16,
                                                      density_threshold=23)
        boundary_points = np.array(boundary_points)

        boundary_points = np.concatenate(
            (boundary_points, np.ones((boundary_points.shape[0], 1)) * np.mean(current_navigable_position[:, 2])),
            axis=1)

        real_boundary = real_boundary[real_boundary[:, 2] > self.floor_height]
        real_boundary = real_boundary[real_boundary[:, 2] < self.ceiling_height]

        real_boundary[:, 2] = np.ones_like(real_boundary[:, 2]) * np.mean(current_navigable_position[:, 2])

        frontiers = []
        for point in boundary_points:
            if self.is_out_of_boundary(real_boundary[:, :2], point[:2], radius=0.1, density_threshold=1):
                frontiers.append(point)
        frontiers = np.array(frontiers)

        # calculate the distance between frontiers and current position
        distance_frontiers = np.linalg.norm(frontiers[:, :2] - np.array(current_position), axis=1)
        frontiers = frontiers[distance_frontiers > 1.7]

        frontiers_pcd = o3d.t.geometry.PointCloud()
        frontiers_pcd.point.positions = o3d.core.Tensor(frontiers, device=frontiers_pcd.device)

        # cluster the frontiers
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(frontiers_pcd.cluster_dbscan(eps=0.2, min_points=5, print_progress=False))

        if len(labels) == 0:
            return [], []
        max_label = labels.max().cpu().numpy()
        # extract each cluster and rule out the small ones caluclate the center of each cluster
        frontier_clusters = []
        frontier_centers = []
        for i in range(max_label + 1):
            mask_idx_tensor = o3d.core.Tensor((labels == i).nonzero()[0], o3d.core.Dtype.Int64,
                                              device=frontiers_pcd.device)
            frontier_cluster = frontiers_pcd.select_by_index(mask_idx_tensor)
            frontier_clusters.append(frontier_cluster.point.positions.cpu().numpy())
            frontier_center = np.mean(frontier_cluster.point.positions.cpu().numpy(), axis=0)
            frontier_centers.append(frontier_center)

        return frontier_clusters, frontier_centers

    def get_frontiers_offerd(self, obstacle_pcd, navigable_pcd, obstacle_height=-0.7, grid_resolution=0.1,
                             closest_distance=1.6):
        frontiers = project_frontier(obstacle_pcd, navigable_pcd, obstacle_height, grid_resolution)

        frontiers[:, 2] = np.ones_like(frontiers[:, 2]) * np.mean(
            np.array(navigable_pcd.point.positions.cpu().numpy())[:, 2])

        current_position = self.current_position[:2]
        distance_frontiers = np.linalg.norm(frontiers[:, :2] - np.array(current_position), axis=1)
        frontiers = frontiers[distance_frontiers > closest_distance]

        frontiers_pcd = o3d.t.geometry.PointCloud()
        frontiers_pcd.point.positions = o3d.core.Tensor(frontiers, device=frontiers_pcd.device)

        # cluster the frontiers
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(frontiers_pcd.cluster_dbscan(eps=0.3, min_points=3, print_progress=False))

        if len(labels) == 0:
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                labels = np.array(frontiers_pcd.cluster_dbscan(eps=0.3, min_points=1, print_progress=False))
            if len(labels) == 0:
                return [], []
        max_label = labels.max().cpu().numpy()
        # extract each cluster and rule out the small ones caluclate the center of each cluster
        frontier_clusters = []
        frontier_centers = []
        for i in range(max_label + 1):
            mask_idx_tensor = o3d.core.Tensor((labels == i).nonzero()[0], o3d.core.Dtype.Int64,
                                              device=frontiers_pcd.device)
            frontier_cluster = frontiers_pcd.select_by_index(mask_idx_tensor)
            frontier_clusters.append(frontier_cluster.point.positions.cpu().numpy())
            frontier_center = np.mean(frontier_cluster.point.positions.cpu().numpy(), axis=0)
            frontier_centers.append(frontier_center)

        return frontier_clusters, frontier_centers

    # def merge_frontier_with_visibility(self, cluster_points, frontier_clusters, frontier_centers):
    #     def is_visible(point1, point2):
    #         interpolate_points = np.linspace(point1, point2, 100)
    #         for point in interpolate_points:
    #             if self.is_out_of_boundary(cluster_points, point, 0.3):
    #                 return False
    #
    #         return True
    #
    #     # 检查frontier center两两之间的可见性
    #     # 如果可见，则把对应的index记录到一个列表中
    #     visible_frontier_index = []
    #     for i in range(len(frontier_centers)):
    #         for j in range(i + 1, len(frontier_centers)):
    #             # check visibility
    #             if is_visible(frontier_centers[i], frontier_centers[j]):
    #                 visible_frontier_index.append([i, j])
    #
    #     # 根据是否可见，将frontier center分为不同的cluster
    #     # 一个cluster中的frontier center两两直接可见
    #     from collections import defaultdict
    #     graph = defaultdict(list)
    #     for u, v in visible_frontier_index:
    #         graph[u].append(v)
    #         graph[v].append(u)
    #
    #     visited = set()
    #     clusters = []
    #
    #     def dfs(node, cluster):
    #         visited.add(node)
    #         cluster.append(node)
    #         for neighbor in graph[node]:
    #             if neighbor not in visited:
    #                 dfs(neighbor, cluster)
    #
    #     # 遍历所有点
    #     for node in range(len(frontier_centers)):
    #         if node not in visited:
    #             cluster = []
    #             dfs(node, cluster)
    #             clusters.append(cluster)
    #
    #     # 合并一个cluster中的frontier_cluster
    #     merged_frontier_clusters = []
    #     merged_frontier_centers = []
    #     for cluster in clusters:
    #         merged_cluster = np.empty((0, 3))
    #         for idx in cluster:
    #             # print(frontier_clusters[idx])
    #             merged_cluster = np.concatenate((merged_cluster, frontier_clusters[idx]), axis=0)
    #         merged_frontier_clusters.append(merged_cluster)
    #         merged_frontier_centers.append(np.mean(merged_cluster, axis=0))
    #
    #     return merged_frontier_clusters, merged_frontier_centers

    def merge_frontier_with_visibility_1(self, cluster_points, obstacle_points, frontier_clusters, frontier_centers):
        def is_out_of_boundary_frontier_cluster(point_cloud, obstacle_points, points, radius=0.1, density_threshold=1):
            # 这里假设有一个函数，检查点是否在点云的边界外
            # 例如可以使用点云的凸包或距离来判断
            radius = radius
            density_threshold = density_threshold
            # 创建KD树以加速邻域搜索
            tree = KDTree(point_cloud)
            obstcale_tree = KDTree(obstacle_points)

            for point in points:
                # 查询半径内的点
                indices = tree.query_ball_point(point, radius)
                obstacle_indices = obstcale_tree.query_ball_point(point, radius/2)
                # 统计邻域内点的数量
                local_density = len(indices) - 1
                obstacle_density = len(obstacle_indices) - 1
                if local_density < density_threshold or obstacle_density > 0:
                    return True

            return False

        def is_visible(point1, point2):
            distance = np.linalg.norm(point1 - point2)
            interpolate_points = np.linspace(point1, point2, int(distance / 0.05))[2:-2]
            print(f'Number of interpolate points: {len(interpolate_points)}')
            if is_out_of_boundary_frontier_cluster(cluster_points, obstacle_points, interpolate_points, radius=0.21, density_threshold=1):
                return False

            return True

        obstacle_points[:, 2] = np.ones_like(obstacle_points[:, 2]) * np.mean(cluster_points[:, 2])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(frontier_centers)
        lines = []

        G = nx.Graph()
        for i in range(len(frontier_centers)):
            G.add_node(i)
            for j in range(i + 1, len(frontier_centers)):
                # check visibility
                if is_visible(frontier_centers[i], frontier_centers[j]):
                    G.add_edge(i, j)
                    lines.append([i, j])

        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([1, 0, 0])

        cliques = list(nx.find_cliques(G))

        # print("All cliques in the graph:", cliques)

        # 合并一个cluster中的frontier_cluster
        merged_frontier_clusters = []
        merged_frontier_centers = []
        for cluster in cliques:
            merged_cluster = np.empty((0, 3))
            merged_cluster_center = np.empty((0, 3))
            for idx in cluster:
                # print(frontier_clusters[idx])
                merged_cluster = np.concatenate((merged_cluster, frontier_clusters[idx]), axis=0)
                merged_cluster_center = np.concatenate((merged_cluster_center, np.array([frontier_centers[idx]])), axis=0)

            merged_frontier_clusters.append(merged_cluster)
            merged_frontier_centers.append(np.mean(merged_cluster_center, axis=0))

        return merged_frontier_clusters, merged_frontier_centers, line_set

    def change_state(self, node):
        nodes_states = self.representation.get_nodes_states()
        print(f'Node state before: {nodes_states}')

        self.representation.visit_node(node)

        nodes_states = self.representation.get_nodes_states()
        print(f'Node state after: {nodes_states}')
        # node_idx = np.where((self.nodes == node).all(axis=1))[0]
        # print(f'Node index: {node_idx}')
        # print(f'Node state before: {self.nodes_state}')
        # print(f'Selected nodes {self.nodes[node_idx]}')
        # self.nodes_state[node_idx] = 1
        # print(f'Node state after: {self.nodes_state}')

    def get_candidate_node(self):
        self.process_obs_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.process_nav_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        return self.representation.find_closest_unexplored_node(self.current_position)

    def get_path(self, end):
        return self.representation.find_the_closest_path(self.current_position, end)

    # def choose_waypoint(self, candidate_nodes):
    #     node_idx = np.argmin(
    #         [np.linalg.norm(np.array([node[0], node[1]]) - self.current_position[:2]) for node in
    #          candidate_nodes])
    #     node = candidate_nodes[node_idx]
    #
    #     self.process_obs_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
    #
    #     return node

