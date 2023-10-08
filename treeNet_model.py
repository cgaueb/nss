import os
import time
import data_stream
import kd_tree
import common
import callbacks
import numpy as np
import tensorflow as tf
import neural_spatial_split

class neural_kdtree :
    def __init__(self, config, model_name='') :
        self.config = config
        self.mPCSize = config['point_cloud_size']
        self.mBatchSize = config['batch_size']
        self.treeLevels = config['tree_levels']
        self.mModelName = model_name
        self.mName = config['name'] + '_' + self.mModelName
        self.mWeightFile = None
        self.mRootFolder = os.path.join(os.getcwd(), 'metadata')
        self.checkpoint_window = config['checkpoint_window']
        self.mModel = None
        self.mRootFolder = os.path.join(self.mRootFolder, self.mName)

        if not os.path.exists(self.mRootFolder) :
            os.mkdir(self.mRootFolder)

        self.mModel = neural_spatial_split.spatialSplit_Model(pConfig=config)
        self.mModel.compile()

    def train(self, pIsCheckpoint=False) :
        test_ds = data_stream.pointcloud_stream(self.config, self.config['test_dir'], self.config['test_csv'], self.mBatchSize)
        train_ds = data_stream.pointcloud_stream(self.config, self.config['train_dir'], self.config['train_csv'], self.mBatchSize)
        valid_ds = None
    
        self.train_dataset = train_ds.init_dataset(True, True)
        test_ds.init_dataset(False, True)

        if self.config['valid_dir'] is not None :
            valid_ds = data_stream.pointcloud_stream(self.config, self.config['valid_dir'], self.config['valid_csv'], self.mBatchSize)
            valid_ds.init_dataset(False, True)

        train_cb = callbacks.recur_trainLog(self.config,
            train_ds, test_ds, valid_ds,
            self.mModel, self.mName, pIsCheckpoint)

        train_cb.on_train_begin()
        numEpochs = self.config['epochs']
        
        for epoch in range(numEpochs) :
            global_loss_log = {}

            for step, (names, point_clouds) in enumerate(self.train_dataset) :
                batch = point_clouds

                print('Epoch {0}/{1} - batch {2}/{3} - '.format(epoch + 1, numEpochs,
                    step + 1, len(self.train_dataset),), end='', flush=True)

                t0 = time.time()

                batch_loss, batch_log = self.mModel.train_step(epoch + 1, step, batch)

                for key, value in batch_loss.items() :
                    if key in global_loss_log :
                        global_loss_log[key] += batch_loss[key]
                    else :
                        global_loss_log[key] = batch_loss[key]

                for key, value in batch_log.items() :
                    if key in global_loss_log :
                        global_loss_log[key] += batch_log[key]
                    else :
                        global_loss_log[key] = batch_log[key]

                print('elapsed time: {0:.2f} - '.format(time.time() - t0), end='', flush=False)
                for key, value in batch_loss.items() :
                    print('{0}: {1:.4f} - '.format(key, value), end='', flush=False)
                print('', flush=True)

            for key, value in global_loss_log.items() :
                global_loss_log[key] /= len(self.train_dataset)

            if (epoch + 1) % self.checkpoint_window == 0 :
                print('Exporting checkpoint')
                self.__save_model()

            print('Evaluating test set... ', end='', flush=True)
            t0 = time.time()
            train_cb.on_epoch_end(epoch + 1, global_loss_log)
            print('elapsed time {0}'.format(time.time() - t0))

        train_cb.on_train_end()
        self.__save_model()

    def __save_model(self) :
        self.mModel.save_weights(os.path.join(self.mRootFolder, 'model_weights'))
        opt_weights = np.asanyarray(self.mModel.optimizer.get_weights(), dtype='object')
        np.save(os.path.join(self.mRootFolder, 'opt_state'), opt_weights)

    def save_variables(self) :
        outDir = os.path.join(self.mRootFolder, 'binary weights')
        common.clear_make_dir(outDir)
        
        for layer in self.mModel.layers :
            layer.export(outDir)

    def load_trained_model(self, load_optimizer=True) :

        if load_optimizer :
            zero_input = tf.zeros(shape=(self.mBatchSize, self.mPCSize, 3), dtype=tf.float32)
            self.mModel.predict_step(zero_input)
            opt_weights = np.load(os.path.join(self.mRootFolder, 'opt_state') + '.npy', allow_pickle=True)
            model_vars = self.mModel.trainable_weights
            zero_grads = [tf.zeros_like(w) for w in model_vars]
            saved_vars = [tf.identity(w) for w in model_vars]
            self.mModel.optimizer.apply_gradients(zip(zero_grads, model_vars))
            [x.assign(y) for x,y in zip(model_vars, saved_vars)]
            self.mModel.optimizer.set_weights(opt_weights)

            print('Num params {0}'.format(self.mModel.get_num_params().numpy()))

        self.mModel.load_weights(os.path.join(self.mRootFolder, 'model_weights'))

    def continue_training(self) :
        self.load_trained_model()
        self.train(True)

    def predict(self, point_clouds, useGreedyInference) :
        input_pc = np.array(point_clouds)

        if len(input_pc.shape) == 2 :
            input_pc = input_pc[np.newaxis, ...]

        for i, pc in enumerate(input_pc) :
            input_pc[i] = common.applyNormalization(pc, common.getAABBox(pc), 1.0)
        
        input_pc = tf.convert_to_tensor(input_pc)
        t0 = time.time()
        if useGreedyInference :
            pred_costs, pred_trees = self.mModel.greedy_predict_step(input_pc)
        else :
            pred_costs, pred_trees = self.mModel.predict_step(input_pc)
        
        elapsed_time = (time.time() - t0) * 1000.0
        pred_trees = pred_trees.numpy()
        has_extra_dim = self.config['train_unbalanced'] or useGreedyInference

        for tree_i in range(pred_trees.shape[0]) :
            for plane_i in range(pred_trees.shape[1]) :
                pred_trees[tree_i, plane_i, 4 if has_extra_dim else 3] -= 1.0

        return pred_costs.numpy(), pred_trees, elapsed_time
    
    def predict_tree(self, point_clouds, useGreedyInference) :
        input_pc = np.array(point_clouds)

        if len(input_pc.shape) == 2 :
            input_pc = input_pc[np.newaxis, ...]

        for i, pc in enumerate(input_pc) :
            input_pc[i] = common.applyNormalization(pc, common.getAABBox(pc), 1.0)
        
        input_pc = tf.convert_to_tensor(input_pc)
        t0 = time.time()
        if useGreedyInference :
            pred_trees = self.mModel.greedy_predict_tree(input_pc)
        else :
            _, pred_trees = self.mModel.predict_step(input_pc)
        
        elapsed_time = (time.time() - t0) * 1000.0
        
        pred_trees = pred_trees.numpy()
        for tree_i in range(pred_trees.shape[0]) :
            for plane_i in range(pred_trees.shape[1]) :
                pred_trees[tree_i, plane_i, 4 if self.config['train_unbalanced'] else 3] -= 1.0
                
        return pred_trees, elapsed_time
    
    def test_model(self, pDataCSV, pc_dir) :
        valid_ds = data_stream.pointcloud_stream(pc_dir, pDataCSV, self.mBatchSize, False)

        validGen = valid_ds.get_dataset()

        sah_tree = kd_tree.kd_tree(pMaxLevels=self.treeLevels,
            pMaxLeafCapacity=0,
            pStrategy=kd_tree.strategy.SURFACE_HEURISTIC_RECURSIVE)

        sum_per_err = 0.0
        sum_per_true_cost = 0.0
        ret = []

        for step, (names, point_clouds) in enumerate(validGen) :
            pred_costs, pred_trees = self.mModel.predict_step(point_clouds)
            point_clouds = point_clouds.numpy()
            result = list(zip(names, point_clouds, pred_costs, pred_trees))
            ret += result

            for names, point_cloud, pred_cost, pred_tree in result :
                t_cost = np.zeros_like(pred_cost)

                err, abs_diff, percentage_err, eval_cost = sah_tree.abs_diff_pre_order(
                    point_cloud, pred_tree, pred_cost, t_cost,
                    normalize_cost=False,
                    allow_empty_nodes=True,
                    allow_out_of_bounds_nodes=True)

                sum_per_err += eval_cost

            norm_factor = 1.0 / (sah_tree.intersection_cost * point_cloud.shape[0])

        return ret, sum_per_err / valid_ds.x, (sum_per_err * norm_factor) / valid_ds.x,