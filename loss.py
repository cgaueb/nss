
import tensorflow as tf

class unsupervised_tree_loss() :
    def __init__(self) :
        pass

    def __call__(self, true_cost, pred_cost) :
        loss1 = tf.math.squared_difference(tf.zeros_like(true_cost), pred_cost)
        return tf.reduce_mean(loss1)

class penalty_tree_loss() :
    def __init__(self, slope, **kwargs) :
        self.pen_slope = tf.ones(shape=(), dtype=tf.float32) * slope

    def eval(self, local_theta) :
        penalty_left = tf.einsum('bi, bi -> bi',
            tf.cast(local_theta < 0.0, tf.float32),
            tf.keras.losses.huber(tf.zeros_like(local_theta), local_theta, delta=0.1)[:, tf.newaxis])

        penalty_right = tf.einsum('bi, bi -> bi',
            tf.cast(local_theta > 1.0, tf.float32),
            tf.keras.losses.huber(tf.zeros_like(local_theta), local_theta - 1.0, delta=0.1)[:, tf.newaxis])

        return (penalty_left + penalty_right)

    @tf.function
    def sumPowerSeries(self, alpha, n) :
        return tf.cast(
            (tf.math.pow(alpha, n+1) - 1) / (alpha - 1),
            dtype=tf.float32)

    @tf.function
    def __call__(self, pred_thetas, max_inter_lvl, lvl_i) :
        slope = self.pen_slope * self.sumPowerSeries(2, max_inter_lvl - lvl_i)
        x_val = self.eval(pred_thetas[:, 0:1])
        y_val = self.eval(pred_thetas[:, 1:2])
        z_val = self.eval(pred_thetas[:, 2:3])
        return tf.squeeze(slope * tf.reduce_mean(x_val + y_val + z_val))