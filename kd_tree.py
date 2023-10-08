from cmath import isinf
import math
import numpy as np
import pickle as pckl
import common
import plots
import struct
from enum import Enum, unique
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

@unique
class error_code(Enum):
    ERROR_NONE = 0
    ERROR_NOT_BALANCED_TREE = 1
    ERROR_UNSUPPORTED_OP = 2
    ERROR_ZERO_VOLUME = 3
    ERROR_ZERO_AREA = 4
    ERROR_EMPTY_NODE = 5

@unique
class strategy(Enum):
    VOLUME_HEURISTIC_GREEDY = 2
    VOLUME_HEURISTIC_RECURSIVE = 3
    SURFACE_HEURISTIC_GREEDY = 4
    SURFACE_HEURISTIC_RECURSIVE = 5
    DENSITY_HEURISTIC_GREEDY = 6
    DENSITY_HEURISTIC_RECURSIVE = 7

@unique
class prim_tp(Enum):
    POINT = 0
    TRIANGLE = 1

class treeNode() :
    def __init__(self, name, points, parentSplitDim, level, primitive_tp, aabb = None, parent_node=None, indices=None) :
        self.name = name
        self.aabb = common.getAABBox(points) if aabb is None else aabb
        self.splitDim = np.argmax(self.aabb[1] - self.aabb[0])
        self.parentSplitDim = self.splitDim if parentSplitDim is None else parentSplitDim
        self.indices = indices
        self.points = points
        self.primitive_tp = primitive_tp
        self.isLeaf = False
        self.level = level
        self.leftChild = None
        self.rightChild = None
        self.splitPoint = None
        self.parent_node = parent_node
        self.plane = np.zeros((4,))
        self.plane[self.splitDim] = 1.0
        self.cost = 0.0
        self.domain_costs = {0 : [], 1 : [], 2 : []}

        if primitive_tp == prim_tp.POINT :
            self.N = points.shape[0]
        else :
            if points.shape[0] % 2 == 0 :
                self.N = points.shape[0] / 2
            else :
                self.N = points.shape[0] / 2 + 1

class split_node() :
    def __init__(self, cost) :
        self.cost = cost
        self.splitDim = 0
        self.plane = None
        self.left_aabb = None
        self.right_aabb = None
        self.left_points = None
        self.right_points = None
        self.left_indices = None
        self.right_indices = None
        self.num_left = 0
        self.num_right = 0

class dummy_node() :
    def __init__(self, aabb, index=None, points=None, name='', parent_node=None) :
        self.aabb = aabb
        self.index = index
        self.points = points
        self.lvl = 0
        self.name = name
        self.parent_node = parent_node
        self.N = points.shape[0] if points is not None else None

class kd_tree() :
    def __init__(self,
        pMaxLevels = 1,
        pName = '',
        pNumBins = 10,
        pStrategy=strategy.SURFACE_HEURISTIC_RECURSIVE,
        pMaxLeafCapacity = 0,
        pPrimitiveType = prim_tp.POINT) :

        self.root = None
        self.points = None
        self.zpoints = None
        self.dim = 3
        self.strategy = pStrategy
        self.maxLeafCapacity = pMaxLeafCapacity
        self.maxlevels = pMaxLevels
        self.max_splits, self.max_nodes = common.num_splits_nodes(self.maxlevels - 1)
        self.nodes_map = dict()
        self.nodes_arr = list()
        self.leaves_arr = list()
        self.levels = 1
        self.name = pName
        self.density_voxels = None
        self.binary_voxels = None
        self.prob_voxels = None
        self.traversal_cost = 12.0
        self.intersection_cost = 10.0
        self.pc_translation = 1.0
        self.pc_scaling = 1.0
        self.num_bins = pNumBins
        self.global_N = 0.0
        self.primitive_tp = pPrimitiveType
        self.record_costs = False
        self.tight_fit = False
        self.__get_build_funcs(self.strategy)

    def build(self, primitives, pForceBalancedTree=True, pAABB=None, pKeepPointInNodes=False, indices=None, normalizeInput=True) :
        #self.__removeDuplicate()

        points = primitives
        if self.primitive_tp == prim_tp.TRIANGLE :
            points = self.__convert_triangles_to_points(primitives)

        self.global_N = points.shape[0]

        if pAABB is None :
            if normalizeInput :
                self.points = common.applyNormalization(points, common.getAABBox(points), self.pc_translation, self.pc_scaling)
            else :
                self.points = points
            self.root = treeNode('root.', self.points, None, 0, self.primitive_tp)
        else :
            self.points = points
            self.root = treeNode('root.', self.points, None, 0, self.primitive_tp, pAABB)

        if self.__vol(self.root.aabb) < 1.e-4 :
            return error_code.ERROR_ZERO_VOLUME

        self.root.indices = np.array([
            np.argsort(self.points[:, 0]),
            np.argsort(self.points[:, 1]),
            np.argsort(self.points[:, 2])])

        if self.strategy == strategy.VOLUME_HEURISTIC_GREEDY or \
            self.strategy == strategy.SURFACE_HEURISTIC_GREEDY or \
            self.strategy == strategy.DENSITY_HEURISTIC_GREEDY :
            self.root.indices = indices
            err = self.__build_greedy_tree(self.root)

        if self.strategy == strategy.VOLUME_HEURISTIC_RECURSIVE or \
            self.strategy == strategy.SURFACE_HEURISTIC_RECURSIVE or \
            self.strategy == strategy.DENSITY_HEURISTIC_RECURSIVE :
            err, cost, steps = self.__build_recursive_tree(self.root, 0)
            print('No. recursive steps {0} - cost {1:.2f}'.format(steps, cost))

        if err == error_code.ERROR_NONE :
            num_nodes = self.__flatten(pKeepPointInNodes)

            if num_nodes != self.max_nodes and pForceBalancedTree :
                err = error_code.ERROR_NOT_BALANCED_TREE

            if err == error_code.ERROR_NONE :
                self.__update_tree_cost()

        if err != error_code.ERROR_NONE :
            self.root.cost = self.__getTreeMaxCost()

        return err

    def farthest_point_samples(self, sample_size) :
        return common.farthest_point_samples(self.points, sample_size)

    def uniform_point_samples(self, sample_size) :
        return common.uniform_point_samples(self.points, sample_size)

    def getPC(self) :
        return self.points

    def normFactor(self) :
        return 1.0 / self.__getTreeMaxCost()

    def getTreeCost(self, pNormalized = True) :
        if pNormalized == True :
            return self.__normalize_cost(self.root.cost)
        else :
            return self.root.cost

    def normalizeCost(self, cost) :
        return self.__normalize_cost(cost)

    def getDepth(self) :
        return self.levels

    def __getArray(self, pMaxLevel) :
        node_values = []

        for lvl in range(self.levels - 1) :
            if lvl < pMaxLevel :
                for node in self.nodes_map[lvl] :
                    plane = np.zeros(shape=(5,))
                    plane[:3] = node.plane[:3]
                    plane[4] = node.plane[3] - 1.0
                    node_values += [plane, ]

        return np.array(node_values).astype(np.float32)

    def getNormalArray(self, maxLevel) :
        return self.__getArray(maxLevel, lambda n : n.plane[:3])

    def getOffsetArray(self, maxLevel) :
        return self.__getArray(maxLevel, lambda n : n.plane[3])

    def getCostArray(self, maxLevel) :
        return self.__getArray(maxLevel, lambda n : self.__normalize_cost(n.cost))

    def getPlaneArray(self, maxLevel) :
        return self.__getArray(maxLevel)

    def getAABBsArray(self, maxLevel) :
        return self.__getArray(maxLevel, lambda n : n.aabb.flatten())

    def getNodeArray(self, maxLevel) :
        return self.__getArray(maxLevel, lambda n : np.concatenate([n.plane, np.expand_dims(np.array(self.__normalize_cost(n.cost)), axis=0)]))

    def getNodeNameArray(self, maxLevel) :
        return self.__getArray(maxLevel, lambda n : n.name)

    def exportTree(self, file) :
        with open(file + '.tr', 'wb') as outF:
            pckl.dump([self.nodes_arr, self.nodes_map, self.intersection_cost, self.traversal_cost, self.strategy, self.name], outF)

    def exportTree_structure(self, file) :
        np.savez_compressed(file, a=self.points, b=self.getPlaneArray(self.levels))

    def importTree(self, file, point_cloud_src_file=None) :
        with open(file, 'rb') as inF :
            structure = pckl.load(inF)
            self.points = None
            self.nodes_arr = structure[0]
            self.nodes_map = structure[1]
            self.intersection_cost = structure[2]
            self.traversal_cost = structure[3]
            self.strategy = structure[4]
            self.name = structure[5]
            self.root = self.nodes_map[0][0]
            self.levels = len(self.nodes_map)
            self.maxlevels = self.levels

            self.max_splits, self.max_nodes = common.num_splits_nodes(self.levels - 1)
            self.__get_build_funcs(self.strategy)

            if point_cloud_src_file is not None :
                with np.load(point_cloud_src_file, allow_pickle=True) as f :
                    pc = f['a']
                    self.points = common.applyNormalization(pc, common.getAABBox(pc), self.pc_translation)

    def __convert_triangles_to_points(self, triangles) :
        points = np.empty(shape=(2 * triangles.shape[0], 3))

        for tr_i, tr in enumerate(triangles) :
            aabb = common.getAABBox(tr)
            points[2 * tr_i, :] = aabb[0, :]
            points[2 * tr_i + 1, :] = aabb[1, :]

        return points

    def __get_binned_offsets(self, num_bins, offsets, useLinSpace=False, bbox=None) :
        if (offsets.size == 0 or offsets is None) and useLinSpace==False :
            return np.linspace(0.0 + self.pc_translation, 1.0 + self.pc_translation, num_bins)
        elif num_bins == 0 :
            return offsets

        if useLinSpace :
            min_p = np.min(offsets)
            max_p = np.max(offsets)

            if max_p - min_p < 0.000001 :
                return np.array([min_p])
            else :
                size = (1 / num_bins) * (max_p - min_p)
                return np.array([min_p + i * size for i in range(1, num_bins)])
        elif bbox is not None :
            min_p = bbox[0]
            max_p = bbox[1]

            if max_p - min_p < 0.000001 :
                return np.array([min_p])
            else :
                size = (1 / num_bins) * (max_p - min_p)
                return np.array([min_p + i * size for i in range(1, num_bins)])
        else :
            return common.farthest_point_samples(offsets, num_bins)

    def getIndicesFromLevel(self, lvl) :
        return [node.indices for node in self.nodes_map[lvl]]
    
    def __getTreeMaxCost(self) :
        return self.leaf_cost_fn(self.root)

    def __normalize_cost(self, cost) :
        return cost / self.__getTreeMaxCost()

    def __build_recursive_tree(self, node, steps) :
        nodePoints = node.points
        cost_opt = float('inf')

        if (self.strategy == strategy.VOLUME_HEURISTIC_RECURSIVE or \
           self.strategy == strategy.DENSITY_HEURISTIC_RECURSIVE) and self.__vol(node.aabb) < 1.e-6 :
            return error_code.ERROR_ZERO_VOLUME, float('inf'), steps

        if self.strategy == strategy.SURFACE_HEURISTIC_RECURSIVE and self.__area(node.aabb) < 1.e-6 :
            return error_code.ERROR_ZERO_AREA, float('inf'), steps

        if node.N > self.maxLeafCapacity and (node.level + 1 < self.maxlevels or self.maxlevels == 0) :
            for splitDim in range(3) :
                offsets = self.__get_binned_offsets(self.num_bins, nodePoints[:, splitDim], True)

                for offset in offsets :
                    mask_left = nodePoints[:, splitDim] < offset
                    mask_right = nodePoints[:, splitDim] >= offset

                    left_points = nodePoints[mask_left, :]
                    right_points = nodePoints[mask_right, :]

                    num_left = np.sum(mask_left)
                    num_right = np.sum(mask_right)

                    if self.primitive_tp == prim_tp.TRIANGLE :
                        num_left = num_left / 2 if num_left % 2 == 0 else num_left / 2 + 1
                        num_right = num_right / 2 if num_right % 2 == 0 else num_right / 2 + 1

                    if num_left >= self.maxLeafCapacity and num_right >= self.maxLeafCapacity :
                        left_aabb = np.copy(node.aabb)
                        right_aabb = np.copy(node.aabb)

                        if self.tight_fit :
                            left_aabb[0, :] = np.min(left_points, axis=0)
                            left_aabb[1, :] = np.max(left_points, axis=0)

                            right_aabb[0, :] = np.min(right_points, axis=0)
                            right_aabb[1, :] = np.max(right_points, axis=0)

                        left_aabb[1, splitDim] = offset
                        right_aabb[0, splitDim] = offset

                        left_node = treeNode(node.name + 'L.', left_points, splitDim, node.level + 1, self.primitive_tp, left_aabb)
                        right_node = treeNode(node.name + 'R.', right_points, splitDim, node.level + 1, self.primitive_tp, right_aabb)

                        local_left_steps = 0
                        local_right_steps = 0

                        left_err, left_cost, left_steps = self.__build_recursive_tree(left_node, local_left_steps)
                        right_err, right_cost, right_steps = self.__build_recursive_tree(right_node, local_right_steps)

                        if np.isinf(left_cost) or np.isinf(right_cost) :
                            continue

                        node_cost = self.inter_cost_fn(node) + \
                            self.inter_prob_fn(left_node, node) * left_cost + \
                            self.inter_prob_fn(right_node, node) * right_cost

                        steps += left_steps + right_steps

                        if self.record_costs :
                            node.domain_costs[splitDim] += [[offset, node_cost],]

                        if node_cost < cost_opt and left_err == error_code.ERROR_NONE and right_err == error_code.ERROR_NONE:
                            node.rightChild = right_node
                            node.leftChild = left_node
                            node.plane[3] = offset
                            node.plane[:3] = np.zeros((3,))
                            node.plane[splitDim] = 1.0
                            node.cost = node_cost
                            cost_opt = node_cost

        if node.level + 1 == self.maxlevels or (node.leftChild is None and node.rightChild is None) :
            node.isLeaf = True

        node_cost = self.leaf_cost_fn(node)
        cost_opt = cost_opt if cost_opt < node_cost else node_cost
        
        return error_code.ERROR_NONE, cost_opt, steps + 1

    def __build_greedy_tree(self, root_node) :
        treeStack = deque()
        treeStack.append(root_node)
        #print(root_node.aabb)

        while len(treeStack) != 0 :
            node = treeStack.pop()
            
            if node.N > self.maxLeafCapacity and (node.level + 1 < self.maxlevels or self.maxlevels == 0) :
                #print(node.aabb)

                if self.strategy == strategy.VOLUME_HEURISTIC_GREEDY and self.__vol(node.aabb) < 1.e-6 :
                    if self.maxlevels == 0 :
                        node.isLeaf = True
                        continue
                    else :
                        return error_code.ERROR_ZERO_VOLUME

                if self.strategy == strategy.SURFACE_HEURISTIC_GREEDY and self.__area(node.aabb) < 1.e-6 :
                    if self.maxlevels == 0 :
                        node.isLeaf = True
                        continue
                    else :
                        return error_code.ERROR_ZERO_AREA

                leaf_cost = self.leaf_cost_fn(node)
                node_cost = self.inter_cost_fn(node)

                candidate_split = self.__eval_splits(node, node.points, node_cost,
                    lambda aabbA, aabbB : self.leaf_prob_fn(dummy_node(aabbA), dummy_node(aabbB)),
                    lambda aabb, points : self.leaf_cost_fn(dummy_node(aabb=aabb, points=points)),
                    node.indices)

                if leaf_cost < candidate_split.cost :
                    node.isLeaf = True
                    node.cost = leaf_cost
                else :
                    node.leftChild = treeNode(node.name + 'L.',
                        candidate_split.left_points,
                        candidate_split.splitDim,
                        node.level + 1,
                        self.primitive_tp,
                        candidate_split.left_aabb,
                        indices=candidate_split.left_indices)
                    
                    node.rightChild = treeNode(node.name + 'R.',
                        candidate_split.right_points,
                        candidate_split.splitDim,
                        node.level + 1,
                        self.primitive_tp,
                        candidate_split.right_aabb,
                        indices=candidate_split.right_indices)

                    node.plane = candidate_split.plane

                    treeStack.append(node.rightChild)
                    treeStack.append(node.leftChild)
            else :
                node.isLeaf = True

        return error_code.ERROR_NONE

    def __update_tree_cost(self, nodes=None) :
        if nodes is None :
            nodes = self.nodes_map

        total_cost = 0.0

        for lvl in range(self.levels - 1, -1, -1) :
            for node in nodes[lvl] :
                if node.isLeaf == True :
                    node.cost = self.leaf_prob_fn(node, self.root) * self.leaf_cost_fn(node)
                else :
                    node.cost = self.inter_prob_fn(node, self.root) * self.inter_cost_fn(node)

                total_cost += node.cost

        self.root.cost = total_cost

    def __eval_recursive_splits(self, node, splitDim, pOffsets) :
        nodePoints = node.points
        cost_opt = float('inf')

        if node.level == 0 :
            cost_records = np.ones((pOffsets.shape[0],))

        if node.N > self.maxLeafCapacity and (node.level + 1 < self.maxlevels or self.maxlevels == 0) :

            if self.strategy == strategy.VOLUME_HEURISTIC_RECURSIVE and self.__vol(node.aabb) < 1.e-6 :
                return error_code.ERROR_ZERO_VOLUME, float('inf'), []

            if self.strategy == strategy.SURFACE_HEURISTIC_RECURSIVE and self.__area(node.aabb) < 1.e-6 :
                return error_code.ERROR_ZERO_AREA, float('inf'), []

            if pOffsets is None :
                for splitDim in range(3) :
                    offsets = self.__get_binned_offsets(self.num_bins, nodePoints[:, splitDim])

                    for offset in offsets :
                        mask_left = nodePoints[:, splitDim] < offset
                        mask_right = nodePoints[:, splitDim] >= offset

                        left_points = nodePoints[mask_left, :]
                        right_points = nodePoints[mask_right, :]

                        if left_points.shape[0] >= self.maxLeafCapacity and right_points.shape[0] >= self.maxLeafCapacity :
                            left_aabb = np.copy(node.aabb)
                            left_aabb[1, splitDim] = offset

                            right_aabb = np.copy(node.aabb)
                            right_aabb[0, splitDim] = offset

                            left_node = treeNode(node.name + 'L.', left_points, splitDim, node.level + 1, self.primitive_tp, left_aabb)
                            right_node = treeNode(node.name + 'R.', right_points, splitDim, node.level + 1, self.primitive_tp, right_aabb)

                            left_err, left_cost, _ = self.__eval_recursive_splits(left_node, splitDim, None)
                            right_err, right_cost, _ = self.__eval_recursive_splits(right_node, splitDim, None)

                            node_cost = self.inter_cost_fn(node) + \
                                self.inter_prob_fn(left_node, node) * left_cost + \
                                self.inter_prob_fn(right_node, node) * right_cost

                            if node_cost < cost_opt and left_err == error_code.ERROR_NONE and right_err == error_code.ERROR_NONE :
                                node.rightChild = right_node
                                node.leftChild = left_node
                                node.plane[3] = offset
                                node.plane[:3] = np.zeros((3,))
                                node.plane[splitDim] = 1.0
                                cost_opt = node_cost

                            if node.level == 0 :
                                cost_records[i] = self.__normalize_cost(node_cost)
            else :
                for i, offset in enumerate(pOffsets) :
                    mask_left = nodePoints[:, splitDim] < offset
                    mask_right = nodePoints[:, splitDim] >= offset

                    left_points = nodePoints[mask_left, :]
                    right_points = nodePoints[mask_right, :]

                    if left_points.shape[0] >= self.maxLeafCapacity and right_points.shape[0] >= self.maxLeafCapacity :
                        left_aabb = np.copy(node.aabb)
                        left_aabb[1, splitDim] = offset

                        right_aabb = np.copy(node.aabb)
                        right_aabb[0, splitDim] = offset

                        left_node = treeNode(node.name + 'L.', left_points, splitDim, node.level + 1, self.primitive_tp, left_aabb)
                        right_node = treeNode(node.name + 'R.', right_points, splitDim, node.level + 1, self.primitive_tp, right_aabb)

                        left_err, left_cost, _ = self.__eval_recursive_splits(left_node, splitDim, None)
                        right_err, right_cost, _ = self.__eval_recursive_splits(right_node, splitDim, None)

                        node_cost = self.inter_cost_fn(node) + \
                            self.inter_prob_fn(left_node, node) * left_cost + \
                            self.inter_prob_fn(right_node, node) * right_cost

                        if node_cost < cost_opt and left_err == error_code.ERROR_NONE and right_err == error_code.ERROR_NONE :
                            node.rightChild = right_node
                            node.leftChild = left_node
                            node.plane[3] = offset
                            node.plane[:3] = np.zeros((3,))
                            node.plane[splitDim] = 1.0
                            cost_opt = node_cost

                        if node.level == 0 :
                            cost_records[i] = self.__normalize_cost(node_cost)

        if node.level + 1 == self.maxlevels or (node.leftChild is None and node.rightChild is None) :
            node.isLeaf = True

        node_cost = self.leaf_cost_fn(node)
        cost_opt = cost_opt if cost_opt < node_cost else node_cost

        if node.level == 0 :
            return error_code.ERROR_NONE, cost_opt, cost_records
        else :
            return error_code.ERROR_NONE, cost_opt, []

    def __eval_splits(self, node, points, node_cost, leaf_prob_fn, leaf_cost_fn, indices=None) :
        offsets_x = self.__get_binned_offsets(self.num_bins, points[:, 0], useLinSpace=False, bbox=node.aabb[:, 0])
        offsets_y = self.__get_binned_offsets(self.num_bins, points[:, 1], useLinSpace=False, bbox=node.aabb[:, 1])
        offsets_z = self.__get_binned_offsets(self.num_bins, points[:, 2], useLinSpace=False, bbox=node.aabb[:, 2])

        xyz_splits = [
            self.__eval_split(node, node.aabb, 0, points, offsets_x, node_cost, leaf_prob_fn, leaf_cost_fn, indices),
            self.__eval_split(node, node.aabb, 1, points, offsets_y, node_cost, leaf_prob_fn, leaf_cost_fn, indices),
            self.__eval_split(node, node.aabb, 2, points, offsets_z, node_cost, leaf_prob_fn, leaf_cost_fn, indices),]

        #print('X:{0} - Y:{1} - Z:{2}'.format(xyz_splits[0].cost, xyz_splits[1].cost, xyz_splits[2].cost))
        
        candidate_split = xyz_splits[0]

        if xyz_splits[1].cost < candidate_split.cost :
            candidate_split = xyz_splits[1]

        if xyz_splits[2].cost < candidate_split.cost :
            candidate_split = xyz_splits[2]

        return candidate_split

    def __eval_split(self, node, node_aabb, splitDim, points, offsets, node_cost, leaf_prob_fn, leaf_cost_fn, indices=None) :
        plane = np.zeros(shape=(4,))
        plane[splitDim] = 1.0

        candidate_split = split_node(float('inf'))

        for i, offset in enumerate(offsets) :
            plane[3] = offset

            mask_left = points[:, splitDim] <= plane[3]
            mask_right = points[:, splitDim] > plane[3]

            num_left = np.sum(mask_left)
            num_right = np.sum(mask_right)

            if self.primitive_tp == prim_tp.TRIANGLE :
                num_left = num_left / 2 if num_left % 2 == 0 else num_left / 2 + 1
                num_right = num_right / 2 if num_right % 2 == 0 else num_right / 2 + 1

            if num_left < self.maxLeafCapacity or num_right < self.maxLeafCapacity :
                continue

            left_points = points[mask_left, :]
            right_points = points[mask_right, :]
            
            left_aabb = np.copy(node_aabb)
            right_aabb = np.copy(node_aabb)

            if self.tight_fit :
                left_aabb[0, :] = np.min(left_points, axis=0)
                left_aabb[1, :] = np.max(left_points, axis=0)

                right_aabb[0, :] = np.min(right_points, axis=0)
                right_aabb[1, :] = np.max(right_points, axis=0)

            left_aabb[1, splitDim] = plane[3]
            right_aabb[0, splitDim] = plane[3]

            if (self.strategy == strategy.VOLUME_HEURISTIC_GREEDY or \
               self.strategy == strategy.DENSITY_HEURISTIC_GREEDY) and (
                self.__vol(right_aabb) < 1.e-6 or self.__vol(left_aabb) < 1.e-6):
                    continue

            if self.strategy == strategy.SURFACE_HEURISTIC_GREEDY and (
                self.__area(right_aabb) < 1.e-6 or self.__area(left_aabb) < 1.e-6):
                continue

            left_cost = leaf_prob_fn(left_aabb, node_aabb) * leaf_cost_fn(left_aabb, left_points)#num_left * self.intersection_cost
            right_cost = leaf_prob_fn(right_aabb, node_aabb) * leaf_cost_fn(right_aabb, right_points)#num_right * self.intersection_cost
            split_cost = node_cost + left_cost + right_cost

            if self.record_costs :
                node.domain_costs[splitDim] += [[offset, split_cost],]

            if split_cost < candidate_split.cost :
                candidate_split.cost = split_cost
                candidate_split.splitDim = splitDim
                candidate_split.plane = np.copy(plane)
                candidate_split.left_aabb = left_aabb
                candidate_split.right_aabb = right_aabb
                candidate_split.left_points = left_points
                candidate_split.right_points = right_points
                candidate_split.left_indices = indices[mask_left, :] if indices is not None else None
                candidate_split.right_indices = indices[mask_right, :] if indices is not None else None
                candidate_split.num_left = num_left
                candidate_split.num_right = num_right

        return candidate_split

    def __vol(self, aabb, r=1.e-4) :
        bmin = aabb[0] - r
        bmax = aabb[1] + r
        diag = bmax - bmin
        return (diag[0] * diag[1] * diag[2])

    def __area(self, aabb) :
        diag = aabb[1] - aabb[0]
        return 2.0 * (diag[0] * diag[1] + diag[1] * diag[2] + diag[0] * diag[2])

    def __flatten(self, pKeepPointInNodes=False) :
        self.nodes_map.clear()
        stack = deque()
        stack.append((self.root, 0))
        num_nodes = 0
        self.nodes_arr = [None,]
        #print('\n')
        while len(stack) != 0 :
            node, index = stack.pop()
            num_nodes += 1

            #print('{0} - {1}/{2}'.format(index, node.aabb[0, :], node.aabb[1, :]))

            if node.isLeaf :
                self.leaves_arr += [node,]

            if not node.isLeaf and not pKeepPointInNodes :
                node.points = None
                node.indices = None

            if node.level in self.nodes_map :
                self.nodes_map[node.level] += [node, ]
            else :
                self.nodes_map[node.level] = [node, ]

            if index > len(self.nodes_arr) - 1 :
                self.nodes_arr += [None] * 2**node.level

            self.nodes_arr[index] = node

            if node.rightChild != None :
                node.rightChild.parent_node = node
                stack.append((node.rightChild, 2 * index + 2))

            if node.leftChild != None :
                node.leftChild.parent_node = node
                stack.append((node.leftChild, 2 * index + 1))

        self.levels = len(self.nodes_map)
        return num_nodes

    def __removeDuplicate(self) :
        pointMap = dict()

        for point in self.points :
            pointMap[(point[0], point[1], point[2])] = point

        self.points = np.array([key for key in pointMap])

    def __get_build_funcs(self, strategy) :
        if strategy == strategy.VOLUME_HEURISTIC_GREEDY or strategy == strategy.VOLUME_HEURISTIC_RECURSIVE:
            prob = lambda nodeA, nodeB : self.__vol(nodeA.aabb) / self.__vol(nodeB.aabb)
        elif strategy == strategy.SURFACE_HEURISTIC_GREEDY or strategy == strategy.SURFACE_HEURISTIC_RECURSIVE:
            prob = lambda nodeA, nodeB : self.__area(nodeA.aabb) / self.__area(nodeB.aabb)
        elif strategy == strategy.DENSITY_HEURISTIC_GREEDY or strategy == strategy.DENSITY_HEURISTIC_RECURSIVE:
            prob = lambda nodeA, nodeB : 1.0
        else :
            prob = None

        self.inter_prob_fn = prob
        self.leaf_prob_fn = prob

        if strategy == strategy.DENSITY_HEURISTIC_GREEDY or strategy == strategy.DENSITY_HEURISTIC_RECURSIVE:
            self.inter_cost_fn = lambda node : self.traversal_cost * node.N
            self.leaf_cost_fn = lambda node : -(node.N / self.global_N)**2 * (1.0 / self.__vol(node.aabb))
        else :
            self.inter_cost_fn = lambda node : self.traversal_cost
            self.leaf_cost_fn = lambda node : self.intersection_cost * node.N

    def density(self, samples) :
        ret = np.zeros(shape=(samples.shape[0]))
        for i, x in enumerate(samples) :
            for leaf in self.leaves_arr :
                if common.isect_point_AABB(x, leaf.aabb) :
                    ret[i] += (leaf.N / self.global_N) * (1.0 / self.__vol(leaf.aabb))

        return ret

    def abs_diff_pre_order(self, point_cloud, pred_planes, pred_cost, true_cost,
        normalize_cost=False, allow_empty_nodes=False, allow_out_of_bounds_nodes=False,
        train_unbalanced=False) :
        if len(pred_planes.shape) == 1 :
            pred_planes = pred_planes[np.newaxis, :]

        if pred_planes.shape[0] != self.max_splits :
            raise ValueError('Incompatible tree structures')

        root = dummy_node(common.getAABBox(point_cloud), 0, point_cloud)
        root.lvl = self.maxlevels - 1

        treeStuck = deque()
        treeStuck.append(root)

        eval_cost = 0.0
        tree_err = 0
        isUnbalanced = False

        while len(treeStuck) != 0 :
            node = treeStuck.pop()

            if node.points.shape[0] == 0 :
                if not allow_empty_nodes:
                    eval_cost = self.intersection_cost * point_cloud.shape[0]
                    if node.lvl == 0 :
                        tree_err = 2
                    else :
                        tree_err = 3
                    break
                else :
                    if train_unbalanced and not node.lvl == 0 :
                        p = pred_planes[node.index][:3]
                        o = pred_planes[node.index][4:5]
                        plane = np.concatenate([p, o], axis=-1)
                        splitDim = np.argmax(pred_planes[node.index][:4])
                        if splitDim == 3 :
                            eval_cost += self.leaf_prob_fn(node, root) * self.leaf_cost_fn(node) # 0 anyway
                            isUnbalanced = True
                            continue
                    else :
                        eval_cost += self.leaf_prob_fn(node, root) * self.leaf_cost_fn(node) # 0 anyway
                        continue

            if node.lvl == 0:
                eval_cost += self.leaf_prob_fn(node, root) * self.leaf_cost_fn(node)
            else:
                #print('Processing node : {0}'.format(node.index))

                if train_unbalanced :
                    p = pred_planes[node.index][:3]
                    o = pred_planes[node.index][4:5]
                    plane = np.concatenate([p, o], axis=-1)
                    splitDim = np.argmax(pred_planes[node.index][:4])
                    if splitDim == 3 :
                        eval_cost += self.leaf_prob_fn(node, root) * self.leaf_cost_fn(node)
                        isUnbalanced = True
                        continue
                else :
                    plane = pred_planes[node.index]
                    splitDim = np.argmax(plane[:3])

                left_aabb = np.copy(node.aabb)
                right_aabb = np.copy(node.aabb)

                nodePoints = node.points
                left_mask = nodePoints[:, splitDim] <= plane[3]
                right_mask = nodePoints[:, splitDim] > plane[3]

                left_points = nodePoints[left_mask, :]
                right_points = nodePoints[right_mask, :]
                
                if plane[3] < (0.0 + self.pc_translation) or plane[3] > (1.0 * self.pc_scaling + self.pc_translation) :
                    if not allow_out_of_bounds_nodes :
                        eval_cost = self.intersection_cost * point_cloud.shape[0]
                        tree_err = 1
                        break
                    else :
                        eval_cost += self.leaf_prob_fn(node, root) * self.leaf_cost_fn(node)
                        continue
                
                if self.tight_fit :
                    if left_points.shape[0] > 0 :
                        left_aabb[0, :] = np.min(left_points, axis=0)
                        left_aabb[1, :] = np.max(left_points, axis=0)

                    if right_points.shape[0] > 0 :
                        right_aabb[0, :] = np.min(right_points, axis=0)
                        right_aabb[1, :] = np.max(right_points, axis=0)
                
                left_aabb[1, splitDim] = plane[3]
                right_aabb[0, splitDim] = plane[3]

                left_aabb = common.refit_aabb(left_aabb)
                right_aabb = common.refit_aabb(right_aabb)

                eval_cost += self.inter_prob_fn(node, root) * self.inter_cost_fn(node)

                idxR = node.index + 1 + common.sumPowerSeries(2, node.lvl - 2)
                idxL = node.index + 1
                right_node = dummy_node(right_aabb, idxR, right_points)
                left_node = dummy_node(left_aabb, idxL, left_points)

                right_node.lvl = node.lvl - 1
                left_node.lvl = node.lvl - 1

                treeStuck.append(right_node)
                treeStuck.append(left_node)

        if normalize_cost :
            eval_cost /= (point_cloud.shape[0] * self.intersection_cost)

        percentage_err = np.abs(true_cost - eval_cost) * 100.0

        if not true_cost == 0.0 :
            percentage_err = np.abs((true_cost - eval_cost) / true_cost) * 100.0

        return tree_err, isUnbalanced, \
            np.abs(true_cost - eval_cost), \
            percentage_err, \
            eval_cost

    @staticmethod
    def preOrder_to_lvlOrder(maxlevels, pred_planes) :
        root = dummy_node(np.ones((2, 3), dtype=np.float32), 0, None, 'root.')
        root.lvl = maxlevels - 1

        treeStuck = deque()
        treeStuck.append(root)

        planes_map = { lvl_i : [] for lvl_i in range(maxlevels) }

        while len(treeStuck) != 0 :
            node = treeStuck.pop()

            if node.lvl == 0 :
                continue

            plane = pred_planes[node.index]
            planes_map[maxlevels - node.lvl - 1] += [plane,]

            left_aabb = np.copy(node.aabb)
            right_aabb = np.copy(node.aabb)

            idxR = node.index + 1 + common.sumPowerSeries(2, node.lvl - 2)
            idxL = node.index + 1
            right_node = dummy_node(right_aabb, idxR, None, node.name + 'R.')
            left_node = dummy_node(left_aabb, idxL, None, node.name + 'L.')

            right_node.lvl = node.lvl - 1
            left_node.lvl = node.lvl - 1

            treeStuck.append(right_node)
            treeStuck.append(left_node)

        lvlorder_planes = []
        for key in planes_map.keys() :
            for plane in planes_map[key] :
                lvlorder_planes += [plane,]

        return np.array(lvlorder_planes)