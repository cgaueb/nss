import numpy as np
import os
import kd_tree
import common
import treeNet_model
import tensorflow as tf
import global_config

from tensorflow.keras import backend
from pathlib import Path

def export_structure_sah(inputFolder, outputFolder, greedyInference=False) :
    config = global_config.sah_config.copy()

    print('Loading network...')
    net = treeNet_model.neural_kdtree(config, 'test_tree')
    net.load_trained_model(load_optimizer=False)
    scene_name = Path(inputFolder).stem
    
    print('Loading scene {0}...'.format(scene_name))
    mesh = common.populate_primitive_buffer(os.path.join(inputFolder, '*'))
    aabb = common.getAABBox(mesh.vertices)
    input_points = common.sample_triangles(mesh.faces[:, np.newaxis, :], mesh.vertices, config['point_cloud_size'] - 2)
    input_points = np.concatenate((input_points, aabb), axis=0)
    input_points = common.applyNormalization(input_points, common.getAABBox(input_points), translation=1.0).astype(np.float32)

    print('Predicting tree...')
    tree_structure, _ = net.predict_tree(input_points.astype(np.float32), useGreedyInference=greedyInference)
    
    tree_structure = tree_structure[0]

    if not greedyInference :
        tree_structure = kd_tree.kd_tree.preOrder_to_lvlOrder(config['tree_levels'], tree_structure)

    print('Exporting tree to {0}'.format(outputFolder))
    tree_structure.tofile(os.path.join(outputFolder, scene_name + ('_greedy' if greedyInference else '_recur')))

def main() :
    inFolder = os.path.join(os.path.join(os.getcwd(), 'datasets'), 'custom_scenes', 'scenes', 'living room 1')

    outFolder = os.path.join(os.getcwd(), 'plots', 'predict hierarchy')

    if not os.path.exists(outFolder) :
        os.mkdir(outFolder)

    export_structure_sah(inFolder, outFolder, False)
    export_structure_sah(inFolder, outFolder, True)

if __name__ == "__main__" :
    print(tf.__version__)
    backend.clear_session()
    tf.config.run_functions_eagerly(False)
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.optimizer.set_jit('autoclustering')
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

    if not os.path.exists(os.path.join(os.getcwd(), 'metadata')) :
        os.mkdir(os.path.join(os.getcwd(), 'metadata'))

    if not os.path.exists(os.path.join(os.getcwd(), 'plots')) :
        os.mkdir(os.path.join(os.getcwd(), 'plots'))

    main()