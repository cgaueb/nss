import tensorflow as tf
import tree_common

class neuralNode_splitter(tf.Module) :
    def __init__(self) :
        super(neuralNode_splitter, self).__init__()
        self.nX = tf.constant([1.0, 0.0, 0.0], dtype=tf.float32)
        self.nY = tf.constant([0.0, 1.0, 0.0], dtype=tf.float32)
        self.nZ = tf.constant([0.0, 0.0, 1.0], dtype=tf.float32)
        self.gen_fn = self.gen_nodes

    @tf.function
    def tight_bounds(self, point_clouds, node_bounds) :
        mask = tf.stop_gradient(tree_common.build_mask(point_clouds, node_bounds))
        masked_pc = tf.stop_gradient(tf.einsum('bij, bik -> bij', point_clouds, mask))
        N = tf.reduce_sum(mask, axis=1)
        xyz_axis = tf.expand_dims(masked_pc, axis=-1)
        xyz_min = tf.reduce_min(tf.abs(xyz_axis - 1.0), axis=1, keepdims=True) + 1.0
        xyz_max = tf.reduce_max(xyz_axis, axis=1, keepdims=True)
        xyz_min = tf.math.minimum(xyz_min, xyz_max)
        tight_node_bmin = tf.squeeze(xyz_min)
        tight_node_bmax = tf.squeeze(xyz_max)
        node_bmin = node_bounds[:, 0:3]
        node_bmax = node_bounds[:, 3:6]
        diag = node_bmax - node_bmin
        theta_bmin = tf.stop_gradient(tf.math.divide_no_nan(tight_node_bmin - node_bmin, diag))
        theta_bmax = tf.stop_gradient(tf.math.divide_no_nan(tight_node_bmax - node_bmin, diag))

        return \
            tf.where(tf.greater(N, 0), node_bmin + theta_bmin * diag, node_bmin), \
            tf.where(tf.greater(N, 0), node_bmin + theta_bmax * diag, node_bmax)
        
    @tf.function
    def gen_nodes(self, normal, theta, node_bounds, point_clouds) :
        node_bmin = node_bounds[:, 0:3]
        node_bmax = node_bounds[:, 3:6]

        b0 = tf.einsum('bk, k -> b', node_bmin, normal)[..., tf.newaxis]
        b1 = tf.einsum('bk, k -> b', node_bmax, normal)[..., tf.newaxis]
        theta_offset = b0 + theta * (b1 - b0)

        right_bmin_temp = tf.einsum('bk, k -> bk', node_bmin, 1.0 - normal) + tf.einsum('bi, k -> bk', theta_offset, normal)
        right_bmax_temp = node_bmax

        right_bmin = tf.minimum(right_bmin_temp, right_bmax_temp)
        right_bmax = tf.maximum(right_bmin_temp, right_bmax_temp)

        left_bmin_temp = node_bmin
        left_bmax_temp = tf.einsum('bk, k -> bk', node_bmax, 1.0 - normal) + tf.einsum('bi, k -> bk', theta_offset, normal)

        left_bmin = tf.minimum(left_bmin_temp, left_bmax_temp)
        left_bmax = tf.maximum(left_bmin_temp, left_bmax_temp)

        left_bbox = tf.concat([left_bmin, left_bmax], axis=-1)
        right_bbox = tf.concat([right_bmin, right_bmax], axis=-1)
        
        return theta_offset, left_bbox, right_bbox

    @tf.function
    def __call__(self, node_bounds, thetas, point_clouds=None) :
        thetas_X = thetas[:, 0:1]
        thetas_Y = thetas[:, 1:2]
        thetas_Z = thetas[:, 2:3]

        offset_X, left_bbox_X, right_bbox_X = self.gen_fn(self.nX, thetas_X, node_bounds, point_clouds)
        offset_Y, left_bbox_Y, right_bbox_Y = self.gen_fn(self.nY, thetas_Y, node_bounds, point_clouds)
        offset_Z, left_bbox_Z, right_bbox_Z = self.gen_fn(self.nZ, thetas_Z, node_bounds, point_clouds)

        return (offset_X, offset_Y, offset_Z), \
            (left_bbox_X, right_bbox_X, left_bbox_Y, right_bbox_Y, left_bbox_Z, right_bbox_Z)

class sah_eval(tf.Module) :
    def __init__(self) :
        super(sah_eval, self).__init__()

    @tf.function
    def area(self, bounds) :
        bmin = bounds[:, 0:3]
        bmax = bounds[:, 3:6]
        diag = bmax - bmin
        x = diag[:, 0:1]
        y = diag[:, 1:2]
        z = diag[:, 2:3]
        return 2.0 * (x * y + x * z + y * z)

    @tf.function
    def __call__(self, parent_bounds, bounds, point_clouds) :
        parent_area = self.area(parent_bounds)
        self_area = self.area(bounds)
        return self_area / parent_area

class vh_eval(tf.Module) :
    def __init__(self, r_eps=1.e-4) :
        super(vh_eval, self).__init__()
        self.r = r_eps

    @tf.function
    def volume(self, bounds) :
        bmin = bounds[:, 0:3] - self.r
        bmax = bounds[:, 3:6] + self.r
        diag = bmax - bmin
        x = diag[:, 0:1]
        y = diag[:, 1:2]
        z = diag[:, 2:3]
        return (x * y * z)

    @tf.function
    def __call__(self, parent_bounds, bounds, point_clouds) :
        parent_vol = self.volume(parent_bounds)
        self_vol = self.volume(bounds)
        return self_vol / parent_vol

class gr_q_eval(tf.Module) :
    def __init__(self, t) :
        super(gr_q_eval, self).__init__()
        self.t_cost = t

    @tf.function
    def __call__(self, point_clouds, parent_normal, parent_offset, parent_bounds, bounds) :
        parent_mask = tf.stop_gradient(tree_common.build_mask(point_clouds, bounds))
        N = tf.reduce_sum(parent_mask, axis=1)
        return N * self.t_cost

class q_eval(tf.Module) :
    def __init__(self, t, beta) :
        super(q_eval, self).__init__()
        self.t_cost = t
        self.count_fn = qL_fn
        self.beta = beta

    @tf.function
    def __call__(self, point_clouds, parent_normal, parent_offset, parent_bounds, parent_mask) :
        axis_points = tf.einsum('bijk, j -> bik', point_clouds[..., tf.newaxis], parent_normal)
        parent_minmax = tf.einsum('bij, j -> bi', tf.reshape(parent_bounds, [-1, 2, 3]), parent_normal)
        N = tf.stop_gradient(tf.reduce_sum(parent_mask, axis=1))
        nL = self.count_fn(self.beta, axis_points, parent_mask, parent_minmax[..., 0:1], parent_minmax[..., 1:], parent_offset)
        return nL * self.t_cost, (N - nL) * self.t_cost

class p_eval(tf.Module) :
    def __init__(self, t) :
        super(p_eval, self).__init__()
        self.t_cost = t

    @tf.function
    def __call__(self, point_clouds, parent_normal, parent_offset, parent_bounds, bounds) :
        return self.t_cost

@tf.function
@tf.custom_gradient
def qL_fn(beta, axis_points, parent_mask, parent_min, parent_max, offset) :
    offset = offset[..., tf.newaxis]
    node_mask = tf.einsum('bij, bij -> bij', parent_mask, tf.cast(axis_points <= offset, tf.float32))
    N = tf.reduce_sum(node_mask, axis=1)

    @tf.function
    def next_step(mask, b) :
        mask_after_offset = tf.einsum('bij, bij -> bij', mask, tf.cast(axis_points > b, tf.float32))
        axisR = tf.einsum('bij, bij -> bij', axis_points, mask_after_offset)
        offset_above = tf.reduce_min(tf.abs(axisR - beta), axis=1, keepdims=True) + beta
        axis_max = tf.reduce_max(axisR, axis=1, keepdims=True)
        offset_above = tf.math.minimum(offset_above, axis_max)
        N1 = tf.reduce_sum(tf.einsum('bij, bij -> bij', mask, tf.cast(axis_points <= offset_above, tf.float32)), axis=1)
        return offset_above, N1

    @tf.function
    def grad(upstream) :
        offset_above, N1 = next_step(parent_mask, offset)
        
        slope = tf.math.divide_no_nan(N1 - N, offset_above[..., 0] - offset[..., 0])
        stepGrad = tf.clip_by_value(slope, 0.0, 1.0 / 0.0001)

        upstream_grad = stepGrad
        upstream_grad = tf.einsum('bi, bi -> bi', upstream, upstream_grad)
        upstream_grad = tf.einsum('bi, bi -> bi', upstream_grad, tf.cast(offset[..., 0] >= parent_min, tf.float32))
        upstream_grad = tf.einsum('bi, bi -> bi', upstream_grad, tf.cast(offset[..., 0] <= parent_max, tf.float32))
        return None, None, None, None, None, upstream_grad

    return N, grad

@tf.function
@tf.custom_gradient
def soft_min4(v0, v1, v2, v3, t) :
    vals = tf.concat([v0, v1, v2, v3], axis=-1)
    ret = tf.reduce_min(vals, axis=-1, keepdims=True)
    
    @tf.function
    def grad(upstream) :
        x = -t * vals
        x -= tf.reduce_max(x, axis=-1, keepdims=True)
        x = tf.nn.softmax(x, axis=-1)
        upstream_grad = upstream * x
        return upstream_grad[:, 0:1], upstream_grad[:, 1:2], upstream_grad[:, 2:3], upstream_grad[:, 3:4], None

    return ret, grad

@tf.function
@tf.custom_gradient
def soft_min3(v0, v1, v2, v3, t) :
    vals = tf.concat([v0, v1, v2], axis=-1)
    ret = tf.reduce_min(vals, axis=-1, keepdims=True)

    @tf.function
    def grad(upstream) :
        x = -t * vals
        x -= tf.reduce_max(x, axis=-1, keepdims=True)
        x = tf.nn.softmax(x, axis=-1)

        upstream_grad = upstream * x
        return upstream_grad[:, 0:1], upstream_grad[:, 1:2], upstream_grad[:, 2:3], None, None
    
    return ret, grad

@tf.function
def hard_min4(v0, v1, v2, v3) :
    vals = tf.concat([v0, v1, v2, v3], axis=-1)
    return vals, tf.reduce_min(vals, axis=-1, keepdims=True)

@tf.function
def hard_min3(v0, v1, v2, v3) :
    vals = tf.concat([v0, v1, v2], axis=-1)
    return vals, tf.reduce_min(vals, axis=-1, keepdims=True)

class pool_treelet(tf.Module) :
    def __init__(self, t, num_splits, p_eval, q_eval, gr_q_eval, w_eval, normFactor) :
        super(pool_treelet, self).__init__()
        self.t = t
        self.num_splits = num_splits
        self.soft_min = soft_min3 if num_splits == 3 else soft_min4
        self.hard_min = hard_min3 if num_splits == 3 else hard_min4
        self.p_eval = p_eval
        self.q_eval = q_eval
        self.w_eval = w_eval
        self.gr_q_eval = gr_q_eval
        self.normFactor = 1.0 / normFactor
        self.nX = tf.constant([1.0, 0.0, 0.0], dtype=tf.float32)
        self.nY = tf.constant([0.0, 1.0, 0.0], dtype=tf.float32)
        self.nZ = tf.constant([0.0, 0.0, 1.0], dtype=tf.float32)
        self.isect_cost = self.q_eval.t_cost
        self.beta = self.q_eval.beta

    @tf.function
    def get_pred_branch_from_leaves(self, cost_xyz, offsets_xyz) :
        batch_size = tf.shape(cost_xyz)[0]
        diag_eye = tf.eye(num_rows=self.num_splits, batch_shape=[batch_size])
        pred_axis = tf.argmin(cost_xyz, axis=1, output_type=tf.int32)
        pred_normal = tf.gather(diag_eye, pred_axis, axis=1, batch_dims=1)
        pred_offset = tf.gather(offsets_xyz, pred_axis, axis=1, batch_dims=1)[..., tf.newaxis]
        pred_planes = tf.concat([pred_normal, pred_offset], axis=-1)
        return pred_planes[:, tf.newaxis, :]

    @tf.function
    def get_pred_branch_from_interior(self, cost_xyz, offsets_xyz, subtree_splits) :
        batch_size = tf.shape(cost_xyz)[0]
        diag_eye = tf.eye(num_rows=self.num_splits, batch_shape=[batch_size])
        pred_axis = tf.argmin(cost_xyz, axis=1, output_type=tf.int32)
        pred_normal = tf.gather(diag_eye, pred_axis, axis=1, batch_dims=1)
        pred_offset = tf.gather(offsets_xyz, pred_axis, axis=1, batch_dims=1)[..., tf.newaxis]
        pred_planes = tf.concat([pred_normal, pred_offset], axis=-1)[:, tf.newaxis, :]
        pred_splits = tf.gather(subtree_splits, pred_axis, axis=1, batch_dims=1)
        pred_planes = tf.concat([pred_planes, pred_splits], axis=1)
        return pred_planes

    @tf.function
    def eval_leaves(self, point_clouds,
        root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
        offsetX, offsetY, offsetZ,
        xL_bounds, xR_bounds, yL_bounds, yR_bounds, zL_bounds, zR_bounds,
        flag) :

        Cnode = \
            self.p_eval(point_clouds, parent_normal, parent_offset, parent_bounds, node_bounds) * \
            self.w_eval(root_bounds, node_bounds, point_clouds)

        node_mask = tf.stop_gradient(tree_common.build_mask(point_clouds, node_bounds))
        
        qL_X, qR_X = self.q_eval(point_clouds, self.nX, offsetX, node_bounds, node_mask)
        qL_Y, qR_Y = self.q_eval(point_clouds, self.nY, offsetY, node_bounds, node_mask)
        qL_Z, qR_Z = self.q_eval(point_clouds, self.nZ, offsetZ, node_bounds, node_mask)
        
        CxL = qL_X * self.w_eval(root_bounds, xL_bounds, point_clouds)
        CxR = qR_X * self.w_eval(root_bounds, xR_bounds, point_clouds)
        CyL = qL_Y * self.w_eval(root_bounds, yL_bounds, point_clouds)
        CyR = qR_Y * self.w_eval(root_bounds, yR_bounds, point_clouds)
        CzL = qL_Z * self.w_eval(root_bounds, zL_bounds, point_clouds)
        CzR = qR_Z * self.w_eval(root_bounds, zR_bounds, point_clouds)

        parent_mask = tf.stop_gradient(tree_common.build_mask(point_clouds, parent_bounds))
        QleafL, QleafR = self.q_eval(point_clouds, parent_normal, parent_offset, parent_bounds, parent_mask)
        QleafRoot = tf.zeros_like(qL_X) * self.normFactor
        Qleaf = QleafRoot * flag[0] + QleafL * flag[1] + QleafR * flag[2]
        Cleaf = Qleaf * self.w_eval(root_bounds, node_bounds, point_clouds)
        
        return Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf

    @tf.function
    def eval_interior(self, point_clouds,
        root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds) :
        
        Cnode = \
            self.p_eval(point_clouds, parent_normal, parent_offset, parent_bounds, node_bounds) * \
            self.w_eval(root_bounds, node_bounds, point_clouds)

        return Cnode

    @tf.function
    def pool_leaves_soft(self, flag, point_clouds,
        root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
        offsetX, offsetY, offsetZ,
        xL_bounds, xR_bounds, yL_bounds, yR_bounds, zL_bounds, zR_bounds) :
        
        Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf = self.eval_leaves(point_clouds, \
            root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
            offsetX, offsetY, offsetZ,
            xL_bounds, xR_bounds, yL_bounds, yR_bounds, zL_bounds, zR_bounds,
            flag)
        
        return (self.pool_soft(Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf), Cleaf)
        
    
    @tf.function
    def pool_leaves(self, flag, point_clouds,
        root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
        offsetX, offsetY, offsetZ,
        xL_bounds, xR_bounds, yL_bounds, yR_bounds, zL_bounds, zR_bounds) :
        
        Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf = self.eval_leaves(point_clouds, \
            root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
            offsetX, offsetY, offsetZ,
            xL_bounds, xR_bounds, yL_bounds, yR_bounds, zL_bounds, zR_bounds,
            flag)
    
        return (Cnode + CxL + CxR, Cnode + CyL + CyR, Cnode + CzL + CzR)

    @tf.function
    def pool_interior_soft(self, flag, point_clouds,
        root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
        CxL, CxR, CyL, CyR, CzL, CzR,
        CxL_leaf, CxR_leaf,
        CyL_leaf, CyR_leaf,
        CzL_leaf, CzR_leaf) :
        
        Cnode = self.eval_interior(point_clouds,
            root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds)

        parent_mask = tf.stop_gradient(tree_common.build_mask(point_clouds, parent_bounds))
        QleafL, QleafR = self.q_eval(point_clouds, parent_normal, parent_offset, parent_bounds, parent_mask)
        QleafRoot = tf.ones_like(QleafL) * self.normFactor
        Qleaf = QleafRoot * flag[0] + QleafL * flag[1] + QleafR * flag[2]
        Cleaf = Qleaf * self.w_eval(root_bounds, node_bounds, point_clouds)
        
        return self.pool_soft(Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf)

    @tf.function
    def pool_soft(self, Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf) :
        cost_x = Cnode + CxL + CxR
        cost_y = Cnode + CyL + CyR
        cost_z = Cnode + CzL + CzR
        treelet_cost = self.soft_min(cost_x, cost_y, cost_z, Cleaf, self.t)
        return treelet_cost
    
    @tf.function
    def pool_leaves_hard(self, flag, point_clouds,
        root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
        offsetX, offsetY, offsetZ,
        xL_bounds, xR_bounds, yL_bounds, yR_bounds, zL_bounds, zR_bounds) :
        
        Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf = self.eval_leaves(point_clouds, \
            root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
            offsetX, offsetY, offsetZ,
            xL_bounds, xR_bounds, yL_bounds, yR_bounds, zL_bounds, zR_bounds,
            flag)

        return self.pool_structure_leaves(Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf, offsetX, offsetY, offsetZ)

    @tf.function
    def pool_interior_hard(self, flag, point_clouds,
        root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds,
        branch_xL, branch_xR, branch_yL, branch_yR, branch_zL, branch_zR,
        offsetX, offsetY, offsetZ) :
        
        Cnode = self.eval_interior(point_clouds,
            root_bounds, parent_bounds, parent_normal, parent_offset, node_bounds)

        parent_mask = tf.stop_gradient(tree_common.build_mask(point_clouds, parent_bounds))
        QleafL, QleafR = self.q_eval(point_clouds, parent_normal, parent_offset, parent_bounds, parent_mask)
        QleafRoot = tf.ones_like(QleafL) * self.normFactor
        Qleaf = QleafRoot * flag[0] + QleafL * flag[1] + QleafR * flag[2]
        Cleaf = Qleaf * self.w_eval(root_bounds, node_bounds, point_clouds)

        return self.pool_structure_interior(Cnode,
            branch_xL, branch_xR, branch_yL, branch_yR, branch_zL, branch_zR,
            Cleaf, offsetX, offsetY, offsetZ)

    @tf.function
    def pool_structure_leaves(self, Cnode, CxL, CxR, CyL, CyR, CzL, CzR, Cleaf, offsetX, offsetY, offsetZ) :
        cost_x = Cnode + CxL + CxR
        cost_y = Cnode + CyL + CyR
        cost_z = Cnode + CzL + CzR
        cost_xyz, min_cost = self.hard_min(cost_x, cost_y, cost_z, Cleaf)
        return min_cost, \
            self.get_pred_branch_from_leaves(cost_xyz,
                tf.concat([offsetX, offsetY, offsetZ, tf.ones_like(offsetZ)], axis=-1))

    @tf.function
    def pool_structure_interior(self, Cnode,
        branch_xL, branch_xR, branch_yL, branch_yR, branch_zL, branch_zR,
        Cleaf, offsetX, offsetY, offsetZ) :

        CxL, planes_xL = branch_xL
        CxR, planes_xR = branch_xR
        CyL, planes_yL = branch_yL
        CyR, planes_yR = branch_yR
        CzL, planes_zL = branch_zL
        CzR, planes_zR = branch_zR

        plane_x = tf.concat([planes_xL, planes_xR], axis=1)
        plane_y = tf.concat([planes_yL, planes_yR], axis=1)
        plane_z = tf.concat([planes_zL, planes_zR], axis=1)

        split_planes = tf.concat([
            plane_x[:, tf.newaxis, ...],
            plane_y[:, tf.newaxis, ...],
            plane_z[:, tf.newaxis, ...],
            ], axis=1)

        cost_x = Cnode + CxL + CxR
        cost_y = Cnode + CyL + CyR
        cost_z = Cnode + CzL + CzR

        cost_xyz, min_cost = self.hard_min(cost_x, cost_y, cost_z, Cleaf)

        return min_cost, \
            self.get_pred_branch_from_interior(cost_xyz,
                tf.concat([offsetX, offsetY, offsetZ, tf.ones_like(offsetZ)], axis=-1),
                split_planes)