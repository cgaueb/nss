
import tensorflow as tf

class neural_spatial_node() :
    def __init__(self, name, lvl) :
        self.name = name
        self.lvl = lvl

        self.bounds = None

        self.theta = None
        self.parent_bounds = None
        self.parent_offset = None
        self.parent_normal = None

@tf.function
def build_mask_(point_cloud, bmin, bmax, lower_bound = 0.0, upper_bound = 1.0) :
    x = point_cloud[..., 0:1]
    y = point_cloud[..., 1:2]
    z = point_cloud[..., 2:3]

    x_bmin = bmin[..., 0:1]
    y_bmin = bmin[..., 1:2]
    z_bmin = bmin[..., 2:3]

    x_bmax = bmax[..., 0:1]
    y_bmax = bmax[..., 1:2]
    z_bmax = bmax[..., 2:3]

    mask_x = tf.where(tf.logical_and(
        x >= x_bmin[..., tf.newaxis],
        x <= x_bmax[..., tf.newaxis]), upper_bound, lower_bound)

    mask_y = tf.where(tf.logical_and(
        y >= y_bmin[..., tf.newaxis],
        y <= y_bmax[..., tf.newaxis]), upper_bound, lower_bound)

    mask_z = tf.where(tf.logical_and(
        z >= z_bmin[..., tf.newaxis],
        z <= z_bmax[..., tf.newaxis]), upper_bound, lower_bound)

    return tf.math.multiply(tf.math.multiply(mask_x, mask_y), mask_z)

@tf.function
def build_mask(point_cloud, bounds, lower_bound = 0.0, upper_bound = 1.0) :
    return build_mask_(point_cloud, bounds[:, 0:3], bounds[:, 3:], lower_bound, upper_bound)

@tf.function
def build_mask_1D(point_cloud, bmin, bmax) :
    return tf.where(tf.logical_and(
        point_cloud >= bmin[..., tf.newaxis],
        point_cloud <= bmax[..., tf.newaxis]), 1.0, 0.0)

@tf.function
def build_mask1D(point_cloud, bounds) :
    return build_mask_1D(point_cloud, bounds[:, 0:], bounds[:, 1:])

@tf.function
def N_fn(point_clouds, bounds) :
    parent_mask = build_mask(point_clouds, bounds)
    N = tf.reduce_sum(parent_mask, axis=1)
    return N