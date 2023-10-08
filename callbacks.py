import os

import kd_tree
import common

import tensorflow.keras as keras
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

class recur_trainLog(keras.callbacks.Callback) :
    def __init__(self, config,
        pTrainGen, pTestGen, pValidGen,
        pInferenceModel, pName='train tree log', pIsCheckpoint=False) :

        self.pConfig = config
        self.treeLevels = config['tree_levels']
        self.numPlanes, self.numNodes = common.num_splits_nodes(self.treeLevels)
        self.dir = os.path.join(os.getcwd(), 'plots', pName)
        self.checkpoint_dir = os.path.join(self.dir, pName + '_c')
        self.name = pName
        self.inference_model = pInferenceModel
        self.fig = plt.figure()
        self.isCheckpoint = pIsCheckpoint
        self.train_stream = pTrainGen
        self.test_stream = pTestGen
        self.valid_stream = pValidGen
        self.best_rec_cost = 100.0
        self.tree_strat = config['tree_strat']
        self.train_unbalanced = config['train_unbalanced']

        if not pIsCheckpoint :
            common.clear_make_dir(self.dir)
            common.clear_make_dir(self.checkpoint_dir)

    def on_train_begin(self, logs=None) :
        self.epoch_offset = 0

        if self.isCheckpoint :
            self.df_train = pd.read_csv(os.path.join(self.dir, 'train_records.csv'))
            self.epoch_offset = self.df_train['epoch'].iloc[-1]
            self.best_rec_cost = np.min(self.df_train['pred_cost_MACost_norm'])
        else :
            self.df_train = pd.DataFrame()
            self.df_trees = {}

    def on_train_end(self, logs=None) :
        plt.close('all')

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        scalar_log = dict()

        for key, value in logs.items() :
            scalar_log[key] = value

        df_scalar_data = { key : [value,] for key, value in scalar_log.items()}

        self._monitor_testSet(df_scalar_data)

        if df_scalar_data['pred_cost_MACost_norm'] < self.best_rec_cost :
            self.best_rec_cost = df_scalar_data['pred_cost_MACost_norm']
            print('Setting new checkpoint {0:.2f}'.format(self.best_rec_cost))
            self.__create_checkpoint()

        df_scalar_data.update({'epoch' : [self.epoch_offset + epoch,]})

        self.df_train = pd.concat([self.df_train, pd.DataFrame(df_scalar_data)], ignore_index=True)
        self.df_train.to_csv(os.path.join(self.dir, 'train_records.csv'), index=False)

        self.__export_plots()

    def __create_checkpoint(self) :
        self.inference_model.save_weights(os.path.join(self.checkpoint_dir, 'model_weights'))
        opt_weights = np.asanyarray(self.inference_model.optimizer.get_weights(), dtype='object')
        np.save(os.path.join(self.checkpoint_dir, 'opt_state'), opt_weights)

    def _monitor_testSet(self, pLogs) :
        sum_out_of_bounds = 0
        sum_empty_leaves = 0
        sum_empty_int = 0
        sum_per_pred_cost = 0.0
        sum_per_pred_cost2 = 0.0
        sum_unbalanced = 0
        sum_soft_costs = 0.0

        sah_tree = kd_tree.kd_tree(pMaxLevels=self.treeLevels,
            pMaxLeafCapacity=1, pStrategy=self.tree_strat)

        sah_tree.intersection_cost = self.pConfig['intersection_cost']
        sah_tree.traversal_cost = self.pConfig['traversal_cost']
        sah_tree.pc_translation = self.pConfig['beta']
        sah_tree.pc_scaling = self.pConfig['gamma']
        total_test_samples = 0

        for step, (names, point_clouds) in enumerate(self.test_stream.dataset) :
            pred_costs, pred_trees = self.inference_model.predict_step(point_clouds)
            total_test_samples += point_clouds.shape[0]
            point_clouds = point_clouds.numpy()
            result = list(zip(point_clouds, pred_costs, tf.zeros_like(pred_costs), pred_trees))

            for point_cloud, pred_cost, soft_pred_cost, pred_tree in result :
                t_cost = np.zeros_like(pred_cost)
                
                sah_tree.tight_fit = False
                err, isUnbalanced, _, _, _ = sah_tree.abs_diff_pre_order(
                    point_cloud, pred_tree, pred_cost, t_cost,
                    normalize_cost=False,
                    allow_empty_nodes=False,
                    allow_out_of_bounds_nodes=False,
                    train_unbalanced=self.train_unbalanced)

                if err == 1 :
                    sum_out_of_bounds += 1
                elif err == 2 :
                    sum_empty_leaves += 1
                elif err == 3 :
                    sum_empty_int += 1

                sum_unbalanced += 1 if isUnbalanced else 0
                norm_factor = 1.0 / (sah_tree.intersection_cost * point_cloud.shape[0])
                sum_soft_costs += np.abs(soft_pred_cost - pred_cost)

                sah_tree.tight_fit = False
                _, _, _, _, eval_cost = sah_tree.abs_diff_pre_order(
                    point_cloud, pred_tree, pred_cost, t_cost,
                    normalize_cost=False,
                    allow_empty_nodes=True,
                    allow_out_of_bounds_nodes=True,
                    train_unbalanced=self.train_unbalanced)
                sum_per_pred_cost2 += eval_cost

        pLogs['pred_cost_MACost_tf'] = sum_per_pred_cost / total_test_samples
        pLogs['pred_cost_MACost_notf'] = sum_per_pred_cost2 / total_test_samples
        pLogs['pred_cost_MACost_norm'] = (sum_per_pred_cost2 * norm_factor) / total_test_samples
        pLogs['num_empty_nodes_int'] = sum_empty_int
        pLogs['num_empty_nodes_leaf'] = sum_empty_leaves
        pLogs['num_unbalanced_trees'] = sum_unbalanced

    def __export_plots(self) :
        df_train = pd.read_csv(os.path.join(self.dir, 'train_records.csv'))

        for column in df_train.columns :
            if column == 'epoch' :
                continue

            if column.find('val_') == -1 :
                if self.treeLevels > 0 :
                    token_i = column.find('_')
                    isNode = column.find('.')

                    if isNode != -1 and token_i != -1 :
                        node_name = column[:token_i] + '\\'
                        column_name = column[token_i:]
                        if not os.path.exists(os.path.join(self.dir, node_name)) :
                            os.mkdir(os.path.join(self.dir, node_name))
                    else :
                        node_name = column
                        column_name = ''

                else :
                    node_name = column
                    column_name = ''

                self.fig.clf()
                ax = plt.axes()
                ax.grid(True)
                ax = sns.lineplot(ax=ax, data=df_train, x='epoch', y=column)
                ax.set_title(column)
                ax.legend(labels=[column,])
                self.fig.savefig(os.path.join(self.dir, node_name + column_name + '.png'), bbox_inches='tight')

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass