import tensorflow as tf
from tensorflow import keras
import numpy as np
import tree_modules
import tree_common
import custom_layers

class spatialSplit_Model(keras.Model) :
    def __init__(self, pConfig) :
        super(spatialSplit_Model, self).__init__()
        self.config = pConfig
        self.treeLevels = self.config['tree_levels']

        self.batch_size = pConfig['batch_size']
        self.splitter = tree_modules.neuralNode_splitter()

        self.w_eval = self.config['weight_fn']
        self.p_eval = self.config['p_fn']
        self.q_eval = self.config['q_fn']
        self.gr_q_eval = self.config['greedy_q_fn']
        self.pooling_treelet = self.config['pooling_fn']
        self.train_unbalanced = self.config['train_unbalanced']
        self.max_inter_lvl = self.treeLevels - 1
        self.pooling_lvl = self.max_inter_lvl - 1
        self.branch_size = 6
        self.offset_encoders = []
        self.level_modules = []
        self.nX = tf.constant([1.0, 0.0, 0.0], dtype=tf.float32)
        self.nY = tf.constant([0.0, 1.0, 0.0], dtype=tf.float32)
        self.nZ = tf.constant([0.0, 0.0, 1.0], dtype=tf.float32)

        self.loss_fn = self.config['loss_fn']
        self.penalty_loss_fn = self.config['penalty_fn']

        for lvl_i in range(0, self.treeLevels - 1) :
            encoder = custom_layers.recursive_tree_level_encoder(lvl_i, **self.config)
            self.offset_encoders += [encoder,]

    def call(self, inputs) :
        assert False

    def compile(self) :
        super(spatialSplit_Model, self).compile()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        self.__make_empty_tree_6wide()
        self.__make_empty_tree_2wide()

    def lvl(self, level) :
        return str(level)

    def __make_empty_tree_2wide(self) :
        self.greedy_nodes = {}
        self.name_map = {0 : 'L.', 1 : 'R.', }
        root_node = tree_common.neural_spatial_node('root.', 0)
        self.greedy_nodes[self.lvl(0)] = [root_node,]

        for lvl_i in range(1, self.treeLevels) :
            self.greedy_nodes[self.lvl(lvl_i)] = []

            for node_i in range(2 * len(self.greedy_nodes[self.lvl(lvl_i - 1)])) :
                parent_node = self.greedy_nodes[self.lvl(lvl_i - 1)][node_i // 2]

                node = tree_common.neural_spatial_node(
                    parent_node.name + self.name_map[node_i % 2], lvl_i)

                if node_i % 2 == 0 :
                    parent_node.L = node
                elif node_i % 2 == 1 :
                    parent_node.R = node

                self.greedy_nodes[self.lvl(lvl_i)] += [node,]

    def __make_empty_tree_6wide(self) :
        self.neural_nodes = {}

        self.name_map = {
            0 : 'x.L.',
            1 : 'x.R.',
            2 : 'y.L.',
            3 : 'y.R.',
            4 : 'z.L.',
            5 : 'z.R.', }

        root_node = tree_common.neural_spatial_node('root.', 0)
        self.neural_nodes[self.lvl(0)] = [root_node,]
        root_node.theta = tf.Variable(tf.ones(shape=(self.config['batch_size'], 3)) * 0.5)

        for lvl_i in range(1, self.treeLevels) :
            self.neural_nodes[self.lvl(lvl_i)] = []

            for node_i in range(6 * len(self.neural_nodes[self.lvl(lvl_i - 1)])) :
                parent_node = self.neural_nodes[self.lvl(lvl_i - 1)][node_i // 6]

                node = tree_common.neural_spatial_node(
                    parent_node.name + self.name_map[node_i % 6], lvl_i)

                node.theta = tf.Variable(tf.ones(shape=(self.config['batch_size'], 3)) * np.random.rand())

                self.neural_nodes[self.lvl(lvl_i)] += [node,]

    def __build_flag(self, lvl_i, node_i) :
        if self.train_unbalanced :
            if lvl_i == 0 :
                flag = [
                    tf.constant([1], dtype=tf.float32),
                    tf.constant([0], dtype=tf.float32),
                    tf.constant([0], dtype=tf.float32)]
            elif node_i % 2 == 0 :
                flag = [
                    tf.constant([0], dtype=tf.float32),
                    tf.constant([1], dtype=tf.float32),
                    tf.constant([0], dtype=tf.float32)]
            else :
                flag = [
                    tf.constant([0], dtype=tf.float32),
                    tf.constant([0], dtype=tf.float32),
                    tf.constant([1], dtype=tf.float32)]
        else :
            flag = [
                tf.constant([0], dtype=tf.float32),
                tf.constant([0], dtype=tf.float32),
                tf.constant([0], dtype=tf.float32)]

        return flag

    def set_initial_input(self, point_cloud, root_node) :
        bmin = tf.reduce_min(point_cloud, axis=1)
        bmax = tf.reduce_max(point_cloud, axis=1)
        root_bounds = tf.concat([bmin, bmax], axis=-1)
        root_node.bounds = root_bounds
        root_node.parent_bounds = root_bounds
        root_node.parent_normal = tf.constant([1.0, 1.0, 1.0], tf.float32)
        root_node.parent_offset = tf.ones(shape=(tf.shape(point_cloud)[0], 1))

    def get_trainable_weights(self) :
        trainable_weights = []

        for offset_encoder in self.offset_encoders :
            trainable_weights += offset_encoder.trainable_weights

        return trainable_weights

    def deferred_train_step(self, point_clouds) :
        root_node = self.neural_nodes[self.lvl(0)][0]
        self.set_initial_input(point_clouds, root_node)
        per_lvl_lthetas = [[None] * len(self.neural_nodes[self.lvl(lvl_i)]) for lvl_i in range(self.max_inter_lvl) ]
        num_out_of_bounds_thetas = tf.zeros(shape=(), dtype=tf.int32)

        with tf.GradientTape(persistent=False) as tape :
            pen_loss = tf.zeros(shape=())
            for lvl_i in range(self.max_inter_lvl) :
                nodes = self.neural_nodes[self.lvl(lvl_i)]
                child_nodes = self.neural_nodes[self.lvl(lvl_i + 1)]
                encoder = self.offset_encoders[lvl_i]

                for node_i, node in enumerate(nodes) :
                    with tape.stop_recording() :
                        pred_lthetas, scale, translate, = encoder([point_clouds, node.bounds])

                    tape.watch(pred_lthetas)

                    pred_thetas = pred_lthetas * scale + translate
                    per_lvl_lthetas[lvl_i][node_i] = pred_lthetas

                    num_out_of_bounds_thetas += tf.reduce_sum(
                            tf.cast(pred_lthetas > 1, dtype=tf.int32) +
                            tf.cast(pred_lthetas < 0, dtype=tf.int32))

                    pen_loss += self.penalty_loss_fn(pred_lthetas,
                        tf.constant([self.max_inter_lvl], dtype=tf.int32),
                        tf.constant([lvl_i], dtype=tf.int32))

                    offsets, bboxes = self.splitter(node.bounds, pred_thetas, point_clouds)

                    for child_i in range(self.branch_size) :
                        child_node = child_nodes[6 * node_i + child_i]
                        child_node.bounds = bboxes[child_i]
                        child_node.parent_bounds = node.bounds

                        if child_i % 6 == 0 or child_i % 6 == 1 :
                            child_node.parent_normal = self.nX
                            child_node.parent_offset = offsets[0]
                        elif child_i % 6 == 2 or child_i % 6 == 3:
                            child_node.parent_normal = self.nY
                            child_node.parent_offset = offsets[1]
                        elif child_i % 6 == 4 or child_i % 6 == 5 :
                            child_node.parent_normal = self.nZ
                            child_node.parent_offset = offsets[2]

            agglomerative_pooling = {lvl_i : [] for lvl_i in range(len(self.neural_nodes) - 1)}
            leaves = self.neural_nodes[self.lvl(self.pooling_lvl + 1)]

            for node_i, node in enumerate(self.neural_nodes[self.lvl(self.pooling_lvl)]) :
                treelet_leaves = leaves[node_i * self.branch_size : node_i * self.branch_size + self.branch_size]
                flag = self.__build_flag(self.max_inter_lvl, node_i)

                agglomerative_pooling[self.pooling_lvl] += [
                    self.pooling_treelet.pool_leaves_soft(flag, point_clouds,
                        root_node.bounds, node.parent_bounds, node.parent_normal, node.parent_offset, node.bounds,
                        treelet_leaves[0].parent_offset, treelet_leaves[2].parent_offset, treelet_leaves[4].parent_offset,
                        treelet_leaves[0].bounds, treelet_leaves[1].bounds,
                        treelet_leaves[2].bounds, treelet_leaves[3].bounds,
                        treelet_leaves[4].bounds, treelet_leaves[5].bounds),]

            for lvl_i in range(self.pooling_lvl - 1, -1, -1) :
                branches = agglomerative_pooling[lvl_i + 1]
                leaves = self.neural_nodes[self.lvl(lvl_i + 1)]

                for branch_i in range(0, len(branches), self.branch_size) :
                    subtree_branches = branches[branch_i:branch_i + self.branch_size]
                    node_i = branch_i // 6
                    node = self.neural_nodes[self.lvl(lvl_i)][node_i]
                    flag = self.__build_flag(lvl_i, node_i)

                    C_recur = self.pooling_treelet.pool_interior_soft(flag, point_clouds,
                        root_node.bounds, node.parent_bounds, node.parent_normal, node.parent_offset, node.bounds,
                        subtree_branches[0][0], subtree_branches[1][0],
                        subtree_branches[2][0], subtree_branches[3][0],
                        subtree_branches[4][0], subtree_branches[5][0],
                        subtree_branches[0][1], subtree_branches[1][1],
                        subtree_branches[2][1], subtree_branches[3][1],
                        subtree_branches[4][1], subtree_branches[5][1])

                    agglomerative_pooling[lvl_i] += [C_recur,]

                agglomerative_pooling[lvl_i + 1] = []

            pred_costs = agglomerative_pooling[0][0] * self.config['norm_factor']
            tree_loss = self.loss_fn(tf.zeros_like(pred_costs), pred_costs)
            loss = tree_loss + pen_loss
            mae = tf.reduce_mean(pred_costs)

        batch_log = {}
        loss_over_offset = tape.gradient(loss, per_lvl_lthetas)
        grad = []
       
        for lvl_i in range(self.max_inter_lvl) :
            nodes = self.neural_nodes[self.lvl(lvl_i)]
            offset_encoder = self.offset_encoders[lvl_i]
            encoder_grads = [tf.zeros_like(w) for w in offset_encoder.trainable_weights]

            for node_i, node in enumerate(nodes) :
                upstream_grad_node_i = loss_over_offset[lvl_i][node_i]

                with tf.GradientTape() as tape :
                    pred_lthetas, _, _ = offset_encoder([point_clouds, node.bounds])

                loss_over_encoder = tape.gradient(pred_lthetas,
                    offset_encoder.trainable_weights,
                    output_gradients=upstream_grad_node_i)

                for grad_i, grads in enumerate(loss_over_encoder) :
                    encoder_grads[grad_i] += grads

            grad += [encoder_grads,]
     
        return {
            'loss' : loss,
            'tree_loss' : tree_loss,
            'pen_loss' : pen_loss,
            'mae' : mae,
            'out_of_bounds_splits'  : num_out_of_bounds_thetas }, \
            grad, batch_log

    def train_step(self, epoch, step, input) :
        dict_losses, grads, batch_log = self.deferred_train_step(input)
        trainable_weights = self.get_trainable_weights()

        flat_grads = [grad for lvl_grads in grads for grad in lvl_grads]
        self.optimizer.apply_gradients(zip(flat_grads, trainable_weights))

        return { key : value.numpy() for key, value in dict_losses.items() }, batch_log

    def predict_step_with_grads(self, input) :
        dict_losses, batch_log = self.deferred_train_step(input)
        return { key : value.numpy() for key, value in dict_losses.items() }, batch_log

    def test_step(self, input) :
        return {}

    #@tf.function
    def predict_step(self, point_clouds) :
        root_node = self.neural_nodes[self.lvl(0)][0]
        self.set_initial_input(point_clouds, root_node)

        for lvl_i in range(self.max_inter_lvl) :
            nodes = self.neural_nodes[self.lvl(lvl_i)]
            child_nodes = self.neural_nodes[self.lvl(lvl_i + 1)]
            encoder = self.offset_encoders[lvl_i]

            for node_i, node in enumerate(nodes) :
                pred_lthetas, scale, translate = encoder([point_clouds, node.bounds])
                pred_thetas = tf.clip_by_value(pred_lthetas * scale + translate, 0.0, 1.0)
                offsets, bboxes = self.splitter(node.bounds, pred_thetas, point_clouds)

                for child_i in range(self.branch_size) :
                    child_node = child_nodes[6 * node_i + child_i]
                    child_node.bounds = bboxes[child_i]
                    child_node.parent_bounds = node.bounds

                    if child_i % 6 == 0 or child_i % 6 == 1 :
                        child_node.parent_normal = self.nX
                        child_node.parent_offset = offsets[0]
                    elif child_i % 6 == 2 or child_i % 6 == 3:
                        child_node.parent_normal = self.nY
                        child_node.parent_offset = offsets[1]
                    elif child_i % 6 == 4 or child_i % 6 == 5 :
                        child_node.parent_normal = self.nZ
                        child_node.parent_offset = offsets[2]

        agglomerative_pooling = {lvl_i : [] for lvl_i in range(len(self.neural_nodes) - 1)}
        leaves = self.neural_nodes[self.lvl(self.pooling_lvl + 1)]

        for node_i, node in enumerate(self.neural_nodes[self.lvl(self.pooling_lvl)]) :
            treelet_leaves = leaves[node_i * self.branch_size : node_i * self.branch_size + self.branch_size]
            flag = self.__build_flag(self.max_inter_lvl, node_i)

            agglomerative_pooling[self.pooling_lvl] += [
                self.pooling_treelet.pool_leaves_hard(flag, point_clouds,
                    root_node.bounds, node.parent_bounds, node.parent_normal, node.parent_offset, node.bounds,
                    treelet_leaves[0].parent_offset, treelet_leaves[2].parent_offset, treelet_leaves[4].parent_offset,
                    treelet_leaves[0].bounds, treelet_leaves[1].bounds,
                    treelet_leaves[2].bounds, treelet_leaves[3].bounds,
                    treelet_leaves[4].bounds, treelet_leaves[5].bounds),]

        for lvl_i in range(self.pooling_lvl - 1, -1, -1) :
            branches = agglomerative_pooling[lvl_i + 1]
            child_nodes = self.neural_nodes[self.lvl(lvl_i + 1)]

            for branch_i in range(0, len(branches), self.branch_size) :
                subtree_branches = branches[branch_i:branch_i + self.branch_size]
                subtree_children = child_nodes[branch_i:branch_i + self.branch_size]
                node_i = branch_i // 6
                node = self.neural_nodes[self.lvl(lvl_i)][node_i]
                flag = self.__build_flag(lvl_i, node_i)

                agglomerative_pooling[lvl_i] += [
                    self.pooling_treelet.pool_interior_hard(flag, point_clouds,
                        root_node.bounds, node.parent_bounds, node.parent_normal, node.parent_offset, node.bounds,
                        subtree_branches[0], subtree_branches[1],
                        subtree_branches[2], subtree_branches[3],
                        subtree_branches[4], subtree_branches[5],
                        subtree_children[0].parent_offset, subtree_children[2].parent_offset, subtree_children[4].parent_offset),]

            agglomerative_pooling[lvl_i + 1] = []

        tree_cost = agglomerative_pooling[0][0][0]
        tree_structure = agglomerative_pooling[0][0][1]
        return (tree_cost, tree_structure)

    #@tf.function
    def greedy_predict_step(self, point_clouds) :
        root_node = self.greedy_nodes[self.lvl(0)][0]
        bmin = tf.reduce_min(point_clouds, axis=1)
        bmax = tf.reduce_max(point_clouds, axis=1)
        root_bounds = tf.concat([bmin, bmax], axis=-1)
        batch_size = tf.shape(point_clouds)[0]
        root_node.bounds = root_bounds
        root_node.parent_bounds = root_bounds
        root_node.parent_offset = tf.ones(shape=(1, 1))

        pred_costs = tf.zeros(shape=(batch_size, 1))
        pred_trees = None

        diag_eye = tf.eye(num_rows=4, batch_shape=[batch_size])

        for lvl_i in range(self.treeLevels) :
            nodes = self.greedy_nodes[self.lvl(lvl_i)]
            child_nodes = None
            encoder = None

            if lvl_i < self.treeLevels - 1 :
                child_nodes = self.greedy_nodes[self.lvl(lvl_i + 1)]
                encoder = self.offset_encoders[lvl_i]

            for node_i, node in enumerate(nodes) :
                if lvl_i == self.treeLevels - 1 :
                    q_cost = self.gr_q_eval(point_clouds,
                        node.parent_normal, node.parent_offset,
                        node.parent_bounds, node.bounds)
                    
                    pred_costs += q_cost * self.w_eval(root_node.bounds, node.bounds, point_clouds)
                    continue

                pred_lthetas, s, t = encoder([point_clouds, node.bounds])

                pred_thetas = tf.clip_by_value(pred_lthetas * s + t, 0.0, 1.0)
                offsets, bboxes = self.splitter(node.bounds, pred_thetas, point_clouds)

                left_boundsX, right_boundsX = bboxes[0], bboxes[1]
                left_boundsY, right_boundsY = bboxes[2], bboxes[3]
                left_boundsZ, right_boundsZ = bboxes[4], bboxes[5]

                left_wX = self.w_eval(node.bounds, left_boundsX, point_clouds)
                right_wX = self.w_eval(node.bounds, right_boundsX, point_clouds)
                left_wY = self.w_eval(node.bounds, left_boundsY, point_clouds)
                right_wY = self.w_eval(node.bounds, right_boundsY, point_clouds)
                left_wZ = self.w_eval(node.bounds, left_boundsZ, point_clouds)
                right_wZ = self.w_eval(node.bounds, right_boundsZ, point_clouds)

                p_costX = self.p_eval(point_clouds, self.nX, offsets[0], node.parent_bounds, node.bounds)
                p_costY = self.p_eval(point_clouds, self.nY, offsets[1], node.parent_bounds, node.bounds)
                p_costZ = self.p_eval(point_clouds, self.nZ, offsets[2], node.parent_bounds, node.bounds)

                qL_costX = self.gr_q_eval(point_clouds, self.nX, offsets[0], node.bounds, left_boundsX)
                qR_costX = self.gr_q_eval(point_clouds, self.nX, offsets[0], node.bounds, right_boundsX)
                qL_costY = self.gr_q_eval(point_clouds, self.nY, offsets[1], node.bounds, left_boundsY)
                qR_costY = self.gr_q_eval(point_clouds, self.nY, offsets[1], node.bounds, right_boundsY)
                qL_costZ = self.gr_q_eval(point_clouds, self.nZ, offsets[2], node.bounds, left_boundsZ)
                qR_costZ = self.gr_q_eval(point_clouds, self.nZ, offsets[2], node.bounds, right_boundsZ)

                cost_x = p_costX + left_wX * qL_costX + right_wX * qR_costX
                cost_y = p_costY + left_wY * qL_costY + right_wY * qR_costY
                cost_z = p_costZ + left_wZ * qL_costZ + right_wZ * qR_costZ

                axis = tf.argmin(tf.concat([cost_x, cost_y, cost_z], axis=-1), axis=-1)
                
                boundsX = tf.concat([left_boundsX, right_boundsX], axis=-1)[:, tf.newaxis, :]
                boundsY = tf.concat([left_boundsY, right_boundsY], axis=-1)[:, tf.newaxis, :]
                boundsZ = tf.concat([left_boundsZ, right_boundsZ], axis=-1)[:, tf.newaxis, :]
                bounds = tf.concat([boundsX, boundsY, boundsZ], axis=1)
                bounds = tf.gather(bounds, indices=axis, axis=1, batch_dims=1)

                offsets = tf.concat([offsets[0], offsets[1], offsets[2]], axis=-1)
                offset = tf.gather(offsets, indices=axis, axis=1, batch_dims=1)[:, tf.newaxis]
                normal = tf.gather(diag_eye, indices=axis, axis=1, batch_dims=1)
                plane = tf.concat([normal, offset], axis=-1)[:, tf.newaxis, :]

                pred_costs += self.p_eval(point_clouds,
                    node.parent_normal, node.parent_offset,
                    node.parent_bounds, node.bounds) * \
                    self.w_eval(root_node.bounds, node.bounds, point_clouds)

                for child_i in range(2) :
                    if child_i == 0 :
                        child_nodes[2 * node_i + child_i].bounds = bounds[:, 0:6]
                    else :
                        child_nodes[2 * node_i + child_i].bounds = bounds[:, 6:]

                    child_nodes[2 * node_i + child_i].parent_bounds = node.bounds
                    child_nodes[2 * node_i + child_i].parent_normal = normal
                    child_nodes[2 * node_i + child_i].parent_offset = offset
                
                if lvl_i == 0 :
                    pred_trees = plane
                else :
                    pred_trees = tf.concat([pred_trees, plane], axis=1)

        return pred_costs, pred_trees
    
    #@tf.function
    def greedy_predict_tree(self, point_clouds) :
        root_node = self.greedy_nodes[self.lvl(0)][0]
        bmin = tf.reduce_min(point_clouds, axis=1)
        bmax = tf.reduce_max(point_clouds, axis=1)
        root_bounds = tf.concat([bmin, bmax], axis=-1)
        batch_size = tf.shape(point_clouds)[0]
        root_node.bounds = root_bounds
        root_node.parent_bounds = root_bounds
        root_node.parent_offset = tf.ones(shape=(1, 1))
        pred_trees = None
        diag_eye = tf.eye(num_rows=4, batch_shape=[batch_size])

        for lvl_i in range(self.max_inter_lvl) :
            nodes = self.greedy_nodes[self.lvl(lvl_i)]
            child_nodes = self.greedy_nodes[self.lvl(lvl_i + 1)]
            encoder = self.offset_encoders[lvl_i]

            for node_i, node in enumerate(nodes) :
                pred_lthetas, s, t = encoder([point_clouds, node.bounds])
                pred_thetas = tf.clip_by_value(pred_lthetas * s + t, 0.0, 1.0)
                offsets, bboxes = self.splitter(node.bounds, pred_thetas, point_clouds)

                left_boundsX, right_boundsX = bboxes[0], bboxes[1]
                left_boundsY, right_boundsY = bboxes[2], bboxes[3]
                left_boundsZ, right_boundsZ = bboxes[4], bboxes[5]

                left_wX = self.w_eval(node.bounds, left_boundsX, point_clouds)
                right_wX = self.w_eval(node.bounds, right_boundsX, point_clouds)
                left_wY = self.w_eval(node.bounds, left_boundsY, point_clouds)
                right_wY = self.w_eval(node.bounds, right_boundsY, point_clouds)
                left_wZ = self.w_eval(node.bounds, left_boundsZ, point_clouds)
                right_wZ = self.w_eval(node.bounds, right_boundsZ, point_clouds)

                node_mask = tree_common.build_mask(point_clouds, node.bounds)
                qL_costX, qR_costX = self.q_eval(point_clouds, self.nX, offsets[0], node.bounds, node_mask)
                qL_costY, qR_costY = self.q_eval(point_clouds, self.nY, offsets[1], node.bounds, node_mask)
                qL_costZ, qR_costZ = self.q_eval(point_clouds, self.nZ, offsets[2], node.bounds, node_mask)

                cost_x = left_wX * qL_costX + right_wX * qR_costX
                cost_y = left_wY * qL_costY + right_wY * qR_costY
                cost_z = left_wZ * qL_costZ + right_wZ * qR_costZ
                
                axis = tf.argmin(tf.concat([cost_x, cost_y, cost_z], axis=-1), axis=-1)
                
                boundsX = tf.concat([left_boundsX, right_boundsX], axis=-1)[:, tf.newaxis, :]
                boundsY = tf.concat([left_boundsY, right_boundsY], axis=-1)[:, tf.newaxis, :]
                boundsZ = tf.concat([left_boundsZ, right_boundsZ], axis=-1)[:, tf.newaxis, :]
                bounds = tf.concat([boundsX, boundsY, boundsZ], axis=1)
                bounds = tf.gather(bounds, indices=axis, axis=1, batch_dims=1)

                offsets = tf.concat([offsets[0], offsets[1], offsets[2]], axis=-1)
                offset = tf.gather(offsets, indices=axis, axis=1, batch_dims=1)[:, tf.newaxis]
                normal = tf.gather(diag_eye, indices=axis, axis=1, batch_dims=1)
                plane = tf.concat([normal, offset], axis=-1)[:, tf.newaxis, :]

                for child_i in range(2) :
                    if child_i == 0 :
                        child_nodes[2 * node_i + child_i].bounds = bounds[:, 0:6]
                    else :
                        child_nodes[2 * node_i + child_i].bounds = bounds[:, 6:]

                    child_nodes[2 * node_i + child_i].parent_bounds = node.bounds
                    child_nodes[2 * node_i + child_i].parent_normal = normal
                    child_nodes[2 * node_i + child_i].parent_offset = offset
                
                if lvl_i == 0 :
                    pred_trees = plane
                else :
                    pred_trees = tf.concat([pred_trees, plane], axis=1)

        return pred_trees