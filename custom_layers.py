import tensorflow as tf
import tree_common
from tensorflow import keras
from tensorflow.keras.layers  import Conv2D, Flatten

class recursive_tree_level_encoder(tf.keras.layers.Layer) :
    def __init__(self, lvl, **pConfig) :
        super(recursive_tree_level_encoder, self).__init__(name='tree_level_encoder_{0}'.format(lvl))

        self.projection_layer = self._get_linear2D(1, 'proj_layer_' + str(lvl), activ='linear', kernel_init='glorot_uniform')
        self.layer1 = self._get_linear2D(pConfig['dense_units_point_enc'], 'Conv_layer_1_' + str(lvl), activ='relu', kernel_init='he_uniform')
        self.layer2 = self._get_linear2D(pConfig['dense_units_point_enc'], 'Conv_layer_2_' + str(lvl), activ='relu', kernel_init='he_uniform')
        self.layer3 = self._get_linear2D(pConfig['dense_units_point_enc'], 'Conv_layer_3_' + str(lvl), activ='relu', kernel_init='he_uniform')
        self.layer4 = self._get_linear2D(pConfig['dense_units_point_enc'], 'Conv_layer_4_' + str(lvl), activ='relu', kernel_init='he_uniform')
        self.layer5 = self._get_linear2D(pConfig['dense_units_point_enc'], 'Conv_layer_5_' + str(lvl), activ='relu', kernel_init='he_uniform')

        self.offset_layer1 = self._get_linear2D(pConfig['dense_units_point_enc'], 'regr_layer_1_' + str(lvl), activ='relu', kernel_init='he_uniform')
        self.offset_layer2 = self._get_linear2D(pConfig['dense_units_point_enc'] // 2, 'regr_layer_2_' + str(lvl), activ='relu', kernel_init='he_uniform')
        self.offset_layer3 = self._get_linear2D(1, 'regr_layer_3_' + str(lvl), activ='linear', kernel_init='glorot_uniform')

        self.lvl = lvl
        self.gamma = tf.constant([pConfig['layer_gamma']], dtype=tf.float32)
        self.beta = tf.constant([pConfig['beta']], dtype=tf.float32)
       
        self.flatten = Flatten()

    def _get_linear2D(self, filter_size, layer_name, activ, kernel_init, use_bias=False) :
        return Conv2D(filters=filter_size,
            kernel_size=(1, 1), strides=(1, 1), padding='valid',
            kernel_initializer=kernel_init,
            activation=activ,
            use_bias=use_bias,
            name=layer_name)
    
    @tf.function
    def map_from_to_opt(self, points, old_min, old_max, new_min, new_max) :
        do = old_max - old_min
        dn = new_max - new_min
        a = tf.math.divide_no_nan(dn, do)
        return tf.einsum('bpjk, bijk -> bpjk', (points - old_min), a) + new_min

    @tf.function
    def object_normalize(self, pc, mask) :
        xyz_axis = tf.expand_dims(pc, axis=-1)
        xyz_min = tf.reduce_min(tf.abs(xyz_axis - self.beta), axis=1, keepdims=True) + self.beta
        xyz_max = tf.reduce_max(xyz_axis, axis=1, keepdims=True)
        xyz_min = tf.math.minimum(xyz_min, xyz_max)

        features = self.map_from_to_opt(xyz_axis,
            xyz_min, xyz_max,
            tf.zeros_like(xyz_min), tf.ones_like(xyz_min)) * self.gamma + 1

        features = tf.einsum('bijf, bik -> bijf', features, mask)
        return (features, tf.squeeze(xyz_min), tf.squeeze(xyz_max))

    @tf.function
    def call(self, input) :
        point_cloud, node_bounds = input

        node_bmin = node_bounds[:, 0:3]
        node_bmax = node_bounds[:, 3:6]

        node_mask = tf.stop_gradient(tree_common.build_mask(point_cloud, node_bounds))
        point_cloud = tf.stop_gradient(tf.einsum('bij, bik -> bij', point_cloud, node_mask))
        features, min_values, max_values = self.object_normalize(point_cloud, node_mask)

        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        
        sum_per_dim = tf.reduce_sum(features, axis=1)
        denom = tf.math.divide_no_nan(1.0, tf.reduce_sum(node_mask, axis=1, keepdims=True))
        global_descr = denom * sum_per_dim
        global_descr = global_descr[:, tf.newaxis, :, :]

        local_thetas = self.offset_layer1(global_descr)
        local_thetas = self.offset_layer2(local_thetas)
        local_thetas = self.offset_layer3(local_thetas)
        local_thetas = self.flatten(local_thetas)

        s = tf.stop_gradient(tf.math.divide_no_nan(max_values - min_values, node_bmax - node_bmin))
        t = tf.stop_gradient(tf.math.divide_no_nan(min_values - node_bmin, node_bmax - node_bmin))

        return local_thetas, s, t