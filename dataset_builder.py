import os
import glob

import common
import kd_tree
import trimesh
from scipy.spatial.transform import Rotation

import numpy as np
import pandas as pd

import multiprocessing

DATASET_DIR = os.path.join(os.getcwd(), 'datasets')

def pointclouds_to_csv(inDataFolder, outCsvFolder, suffix, ext='*.npz') :
    path = os.path.join(outCsvFolder, suffix + '.csv')
    trees = glob.glob(os.path.join(inDataFolder, ext))

    tree_dict = dict()
    tree_dict['samples'] = list()

    for tree in trees :
        treeName = os.path.splitext(os.path.basename(tree))[0]
        tree_dict['samples'] += [treeName,]

    tree_df = pd.DataFrame.from_dict(tree_dict)
    tree_df.to_csv(path, index=False)

def pointclouds_to_csv_from_folder(inDataFolder, outCsvFolder, suffix, ext='*.npz') :
    path = os.path.join(outCsvFolder, suffix + '.csv')
    
    tree_dict = dict()
    tree_dict['samples'] = list()
    
    for folder in glob.glob(os.path.join(inDataFolder, '*')) :
        trees = glob.glob(os.path.join(folder, ext))
        folder_name = os.path.splitext(os.path.basename(folder))[0]

        for tree in trees :
            treeName = os.path.splitext(os.path.basename(tree))[0]
            tree_dict['samples'] += [os.path.join(folder_name, treeName),]

    tree_df = pd.DataFrame.from_dict(tree_dict)
    tree_df.to_csv(path, index=False)

def custom_hierarchical_scene_extractor_sah(scenes, inputFolder, outputFolder, point_samples) :
    for scene in scenes :
        print('Loading scene {0}... '.format(scene), end='', flush=False)
        mesh = common.populate_primitive_buffer(os.path.join(inputFolder, scene, '*'))
        centroids = common.populate_centroids(mesh)
        face_indices = np.arange(mesh.faces.shape[0])[:, np.newaxis]

        print('Done')
        print('Building tree {0}... '.format(scene), end='', flush=False)
        tree = kd_tree.kd_tree(pMaxLevels=6, pNumBins=12,
            pStrategy=kd_tree.strategy.SURFACE_HEURISTIC_GREEDY,
            pMaxLeafCapacity=8,)

        tree.build(centroids, pForceBalancedTree=False, pKeepPointInNodes=True, indices=face_indices)
        print('Done')

        print('Sampling from tree {0}... '.format(scene), end='', flush=False)
        for lvl_i in range(1, tree.levels) :
            list_indices = tree.getIndicesFromLevel(lvl_i)
            
            for node_i, node_indices in enumerate(list_indices) :
                batch = common.sample_triangles(mesh.faces[node_indices, :], mesh.vertices, point_samples)
                batch_normalized = common.applyNormalization(batch, common.getAABBox(batch), 1.0)

                np.savez_compressed(os.path.join(outputFolder, scene + '_b_{0}{1}{2}'.format(lvl_i, node_i, 0)),
                    a=batch_normalized)

                for rot_i in range(2) :
                    rot_batch = batch @ Rotation.random().as_matrix()
                    rot_batch = common.applyNormalization(rot_batch, common.getAABBox(rot_batch), 1.0)

                    np.savez_compressed(os.path.join(outputFolder, scene + '_b_{0}{1}{2}'.format(lvl_i, node_i, rot_i + 1)),
                        a=rot_batch)
        print('Done')

def custom_hierarchical_scene_extractor_vh(scenes, inputFolder, outputFolder, point_samples) :
    for scene in scenes :
        print('Loading scene {0}...'.format(scene))
        mesh = common.populate_point_buffer(os.path.join(inputFolder, scene, '*'))
        face_indices = np.arange(mesh.vertices.shape[0])[:, np.newaxis]

        print('Building tree {0}...'.format(scene))
        tree = kd_tree.kd_tree(pMaxLevels=6, pNumBins=12,
            pStrategy=kd_tree.strategy.VOLUME_HEURISTIC_GREEDY,
            pMaxLeafCapacity=64,)

        tree.build(mesh.vertices, pForceBalancedTree=False, pKeepPointInNodes=True, indices=face_indices)

        print('Sampling from tree {0}...'.format(scene))
        for lvl_i in range(1, tree.levels) :
            list_indices = tree.getIndicesFromLevel(lvl_i)

            for node_i, node_indices in enumerate(list_indices) :
                v = mesh.vertices[node_indices.flatten(), :]
                batch = common.uniform_point_samples(v, point_samples)
                np.savez_compressed(os.path.join(outputFolder, scene + '_b_{0}{1}{2}'.format(lvl_i, node_i, 0)),
                    a=common.applyNormalization(batch, common.getAABBox(batch), 1.0))
                
                for rot_i in range(2) :
                    rot_batch = batch @ Rotation.random().as_matrix()
                    rot_batch = common.applyNormalization(rot_batch, common.getAABBox(rot_batch), 1.0)
                    np.savez_compressed(os.path.join(outputFolder, scene + '_b_{0}{1}{2}'.format(lvl_i, node_i, rot_i + 1)),
                        a=rot_batch)

def custom_scene_subsamples_extractor_sah(scenes, inputFolder, outputFolder, point_samples) :
    for scene in scenes :
        outputSceneFolder = os.path.join(outputFolder, scene)
        common.clear_make_dir(outputSceneFolder)

        print('Extracting samples from {0}...'.format(scene))
        mesh = common.populate_primitive_buffer(os.path.join(inputFolder, scene, '*'))
        
        aabb = common.getAABBox(mesh.vertices)
        input_points = []

        for i in range(64) :
            batch = common.sample_triangles(mesh.faces[:, np.newaxis, :], mesh.vertices, point_samples - 2)
            batch = np.concatenate((batch, aabb), axis=0)
            input_points += [batch,]

        for index, batch in enumerate(input_points) :
            batch_norm = common.applyNormalization(batch, common.getAABBox(batch), 1.0).astype(np.float32)
            np.savez_compressed(os.path.join(outputSceneFolder, scene + '_b' + str(index)), a=batch_norm)

def custom_scene_subsamples_extractor_vh(scenes, inputFolder, outputFolder, point_samples) :
    for scene in scenes :
        outputSceneFolder = os.path.join(outputFolder, scene)
        common.clear_make_dir(outputSceneFolder)
        print('Extracting samples from {0}...'.format(scene))
        mesh = common.populate_point_buffer(os.path.join(inputFolder, scene, '*'))
        aabb = common.getAABBox(mesh.vertices)
        input_points = []

        for i in range(64) :
            batch = common.uniform_point_samples(mesh.vertices, point_samples - 2)
            batch = np.concatenate((batch, aabb), axis=0)
            input_points += [batch,]

        for index, batch in enumerate(input_points) :
            np.savez_compressed(os.path.join(outputSceneFolder, scene + '_b' + str(index)),
                a=common.applyNormalization(batch, common.getAABBox(batch), 1.0))
            
def custom_scene_extractor(scenes, inputFolder, outputFolder) :
    triangle_batch_frac = 0.1
    triangle_stride_frac = 0.1
    iterations = 1
    sample_size = 1024

    for scene in scenes :
        print('Extracting samples from {0}...'.format(scene))
        batches = common.centroid_batches(os.path.join(inputFolder, scene, '*'),
            triangle_batch_frac, sample_size, triangle_stride_frac, iterations)

        for index, batch in enumerate(batches) :
            np.savez_compressed(os.path.join(outputFolder, scene + '_b' + str(index)),
                a=common.applyNormalization(batch, common.getAABBox(batch), 1.0))

            for i in range(2) :
                rot_batch = batch @ Rotation.random().as_matrix()
                rot_batch = common.applyNormalization(rot_batch, common.getAABBox(rot_batch), 1.0)
                np.savez_compressed(os.path.join(outputFolder, scene + '_b' + str(index) + '_rot' + str(i + 1)),
                    a=rot_batch)

def par_custom_scene_extractor(outFolder, fn, basedir, inputFolder, point_samples) :
    inputFolder = os.path.join(basedir, inputFolder)
    outputFolder = os.path.join(basedir, outFolder)
    common.clear_make_dir(outputFolder)

    root, scenes, files = next(os.walk(inputFolder))

    worker_count = multiprocessing.cpu_count() - 2
    scene_slices = np.array_split(scenes, worker_count)
    batches = len(scene_slices)
    args = list(zip(scene_slices, [inputFolder,]*batches, [outputFolder,]*batches, [point_samples,]*batches))
    common.launch_par_processes(worker_count, fn, args)

def build_sah_dataset(point_samples, prefix_label='') :
    build_train = False
    build_test = True
    build_valid = False
    
    if build_train :
        par_custom_scene_extractor('train_fragments_{1}_{0}sah'.format(prefix_label, point_samples), custom_hierarchical_scene_extractor_sah,
            os.path.join(DATASET_DIR, 'custom_scenes'), 'scenes', point_samples)

        pointclouds_to_csv(
            os.path.join(DATASET_DIR, 'custom_scenes', 'train_fragments_{1}_{0}sah'.format(prefix_label, point_samples)),
            os.path.join(DATASET_DIR, 'custom_scenes'),
            'train_fragments_{1}_{0}sah'.format(prefix_label, point_samples), ext='*.npz')

    if build_test :
        par_custom_scene_extractor('test_fragments_{1}_{0}sah'.format(prefix_label, point_samples), custom_scene_subsamples_extractor_sah,
            os.path.join(DATASET_DIR, 'custom_scenes'), 'scenes', point_samples)
    
        pointclouds_to_csv_from_folder(
            os.path.join(DATASET_DIR, 'custom_scenes', 'test_fragments_{1}_{0}sah'.format(prefix_label, point_samples)),
            os.path.join(DATASET_DIR, 'custom_scenes'),
            'test_fragments_{1}_{0}sah'.format(prefix_label, point_samples), ext='*.npz')

    if build_valid :
        par_custom_scene_extractor('valid_fragments_{1}_{0}sah'.format(prefix_label, point_samples), custom_scene_subsamples_extractor_sah,
            os.path.join(DATASET_DIR, 'custom_scenes'), 'extra_scenes', point_samples)
    
        pointclouds_to_csv_from_folder(
            os.path.join(DATASET_DIR, 'custom_scenes', 'valid_fragments_{1}_{0}sah'.format(prefix_label, point_samples)),
            os.path.join(DATASET_DIR, 'custom_scenes'),
            'valid_fragments_{1}_{0}sah'.format(prefix_label, point_samples), ext='*.npz')
        
def build_pc_from_scenes(scenes, inputFolder, outputFolder, point_samples) :
    for scene in scenes :
        outputSceneFolder = os.path.join(outputFolder, scene)
        common.clear_make_dir(outputSceneFolder)
        print('Extracting samples from {0}...'.format(scene))
        mesh = common.populate_primitive_buffer(os.path.join(inputFolder, scene, '*'))
        points = mesh.sample(point_samples)
        submesh = trimesh.Trimesh(vertices=points)
        submesh.export(file_obj=os.path.join(outputSceneFolder, scene + ".ply"), file_type='ply')

def build_vh_dataset(point_samples) :
    build_train = True
    build_test = True
    build_valid = False
    
    if build_train :
        par_custom_scene_extractor('train_fragments_{0}_vh'.format(point_samples), custom_hierarchical_scene_extractor_vh,
            os.path.join(DATASET_DIR, 'custom_scenes'), 'scenes_pc', point_samples)

        pointclouds_to_csv(
            os.path.join(DATASET_DIR, 'custom_scenes', 'train_fragments_{0}_vh'.format(point_samples)),
            os.path.join(DATASET_DIR, 'custom_scenes'),
            'train_fragments_{0}_vh'.format(point_samples), ext='*.npz')
        
    if build_test :
        par_custom_scene_extractor('test_fragments_{0}_vh'.format(point_samples), custom_scene_subsamples_extractor_vh,
            os.path.join(DATASET_DIR, 'custom_scenes'), 'scenes_pc', point_samples)

        pointclouds_to_csv_from_folder(
            os.path.join(DATASET_DIR, 'custom_scenes', 'test_fragments_{0}_vh'.format(point_samples)),
            os.path.join(DATASET_DIR, 'custom_scenes'),
            'test_fragments_{0}_vh'.format(point_samples), ext='*.npz')

    if build_valid :
        par_custom_scene_extractor('valid_fragments_{0}_vh'.format(point_samples), custom_scene_subsamples_extractor_vh,
            os.path.join(DATASET_DIR, 'custom_scenes'), 'extra_scenes_pc', point_samples)

        pointclouds_to_csv_from_folder(
            os.path.join(DATASET_DIR, 'custom_scenes', 'valid_fragments_{0}_vh'.format(point_samples)),
            os.path.join(DATASET_DIR, 'custom_scenes'),
            'valid_fragments_{0}_vh'.format(point_samples), ext='*.npz')
        
def build_pc() :
    par_custom_scene_extractor('scenes_pc', build_pc_from_scenes,
        os.path.join(DATASET_DIR, 'custom_scenes'), 'scenes', 1000000)

def main() :
    build_pc()
    build_vh_dataset(2048)
    build_sah_dataset(2048)

if __name__ == "__main__" :
    main()