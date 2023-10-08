import os
import common
import numpy as np
import pandas as pd
import tensorflow as tf

class pointcloud_stream :
    def __init__(self, pConfig, pPointCloudRootFolder, pCSV, pBatchSize) :
        self.config = pConfig
        self.pc_rootfolder = pPointCloudRootFolder
        self.df_files = pd.read_csv(pCSV)
        self.x = len(self.df_files.index)
        self.batchSize = self.x if self.x < pBatchSize else pBatchSize
        self.ref_point_clouds = None
        self.ref_names = None

    def __load_trees(self) :
        print('Caching trees...')
        point_clouds = []
        names = []

        for df_file in self.df_files.iterrows() :
            name = df_file[1]['samples']

            with np.load(os.path.join(self.pc_rootfolder, name) + '.npz', allow_pickle=True) as f :
                pc = f['a']

            s = np.random.normal(0, 0.0001, pc.shape[0] * 3)
            points = pc + np.reshape(s, pc.shape)
            points = pc
            if(common.volume(points) < 1.e-4) :
                continue

            points = common.applyNormalization(pc, common.getAABBox(pc), 1.0, self.config['gamma'])
            point_clouds += [points,]
            names += [name,]
        print('Tree caching finished')

        return np.array(names), np.array(point_clouds, dtype=np.float32)

    def init_dataset(self, pDropReminder=False, pShuffle = True) :
        names, point_clouds = self.__load_trees()
        self.ref_point_clouds = point_clouds
        self.ref_names = names

        self.dataset = tf.data.Dataset.from_tensor_slices((names, point_clouds))
        self.dataset = self.dataset.shuffle(buffer_size=self.x, reshuffle_each_iteration=pShuffle)
        self.dataset = self.dataset.batch(batch_size=self.batchSize, drop_remainder=pDropReminder)
        self.dataset = self.dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return self.dataset