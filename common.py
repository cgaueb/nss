import math
import numpy as np
import glob
import trimesh
import os
import shutil

from concurrent.futures import ProcessPoolExecutor

def make_dir(pFolderDir) :
    if not os.path.exists(pFolderDir) :
        os.mkdir(pFolderDir)

def clear_make_dir(pFolderDir) :
    if not os.path.exists(pFolderDir) :
        os.mkdir(pFolderDir)
    else :
        for file in glob.glob(os.path.join(pFolderDir, '*')) :
            if os.path.isdir(file) :
                shutil.rmtree(file)
            else :
                os.remove(file)
        #shutil.rmtree(pFolderDir)
        #os.mkdir(pFolderDir)

def sumPowerSeries(alpha, n) :
    return int((math.pow(alpha, n+1) - 1) / (alpha - 1))

def num_splits_nodes(pTreeLevels) :
    return sumPowerSeries(2, pTreeLevels - 1), sumPowerSeries(2, pTreeLevels)

def getAABBox(pPoints) :
    aabb = np.empty((2, 3))

    aabb[0, 0] = np.min(pPoints[:, 0])
    aabb[0, 1] = np.min(pPoints[:, 1])
    aabb[0, 2] = np.min(pPoints[:, 2])

    aabb[1, 0] = np.max(pPoints[:, 0])
    aabb[1, 1] = np.max(pPoints[:, 1])
    aabb[1, 2] = np.max(pPoints[:, 2])

    return aabb

def areaTriangle(a, b, c) :
    return np.linalg.norm(np.cross(b-a, c-a)) / 2

def volume(points) :
    aabb = getAABBox(points)
    diag = aabb[1, :] - aabb[0, :]
    return diag[0] * diag[1] * diag[2]

def applyNormalization(pPoints, pAABB, translation = 0.0, gamma = 1.0) :
    diag = pAABB[1] - pAABB[0]
    minP = np.min(diag)
    maxP = np.max(diag)
    scale = np.where(maxP > 0.0, 1.0 / maxP, 0.0)
    normPoints = (pPoints - pAABB[0]) * scale
    normPoints = normPoints * gamma + translation

    normAABB = (pAABB - pAABB[0]) * scale
    normAABB = normAABB + translation
    normCent = (normAABB[1] + normAABB[0]) * 0.5
    
    return normPoints

def sample_triangles(faces, vertices, num_samples) :
    sample_faces = uniform_point_samples(faces, num_samples)
    batch = np.zeros(shape=(num_samples, 3), dtype=np.float32)
    for sample_face_i in range(sample_faces.shape[0]) :
        v0 = vertices[sample_faces[sample_face_i, 0, 0], :]
        v1 = vertices[sample_faces[sample_face_i, 0, 1], :]
        v2 = vertices[sample_faces[sample_face_i, 0, 2], :]
        r1 = np.random.uniform()
        r2 = np.random.uniform()
        su0 = math.sqrt(r1)
        bx = 1 - su0
        by = r2 * su0
        batch[sample_face_i, :] = v0 * bx + v1 * by + v2 * (1.0 - bx - by)
    return batch

def refit_aabb(aabb) :
    bmin = np.copy(aabb[0, :])
    bmax = np.copy(aabb[1, :])
    aabb[0, :] = np.minimum(bmin, bmax)
    aabb[1, :] = np.maximum(bmin, bmax)
    return aabb

def isect_point_AABB(point, aabb) :
    return np.all(point >= aabb[0, :]) and np.all(point <= aabb[1, :])

def populate_primitive_buffer(pFolder) :
    for i, file in enumerate(glob.glob(pFolder)) :
        model = trimesh.load(file, force='mesh', skip_materials=True)

        if i == 0 :
            vertices = model.vertices[model.faces.flatten(), :]
            faces = np.reshape(np.arange(vertices.shape[0]), [-1, 3])
        else :
            v = model.vertices[model.faces.flatten(), :]

            indices = np.arange(vertices.shape[0], vertices.shape[0] + v.shape[0])
            indices = np.reshape(indices, [-1, 3])

            vertices = np.concatenate([vertices, v], axis=0)
            faces = np.concatenate([faces, indices], axis=0)

    return trimesh.Trimesh(vertices=vertices, faces=faces)

def launch_par_processes(worker_count, fn, args) :
    if len(args) != worker_count :
        raise ValueError('worker count != len(args)')

    print('Running {0} parallel processes...'.format(worker_count))

    if worker_count == 1 :
        fn(*(args[0]))
    else :
        with ProcessPoolExecutor(max_workers=worker_count) as executor :
            for batch in range(worker_count) :
                future = executor.submit(fn, *(args[batch]))
                future.add_done_callback(lambda f : print('Batch process finished.'))

def strided_arr3D(arr, L, S ):
    nrows = ((arr.shape[0] - L) // S) + 1
    elem_size = arr.itemsize

    tiles = np.lib.stride_tricks.as_strided(arr, shape=(nrows, L, 3),
        strides=(S * 3 * elem_size, 3 * elem_size, elem_size))

    return tiles

def uniform_point_samples(points, sample_size) :
    indices = np.random.choice(points.shape[0], sample_size,
        replace=False if sample_size < points.shape[0] else True)
    return points[indices, :]

def farthest_point_samples(points, sample_size):
    if len(points.shape) == 1 :
        points = points[:, np.newaxis]

    calc_dist = lambda p0, points : ((p0 - points)**2).sum(axis=1)

    farthest_pts = np.zeros((sample_size, points.shape[1]))
    farthest_pts[0] = points[np.random.randint(points.shape[0])]
    distances = calc_dist(farthest_pts[0], points)

    for i in range(1, sample_size):
        farthest_pts[i] = points[np.argmax(distances)]
        distances = np.minimum(distances, calc_dist(farthest_pts[i], points))

    return np.squeeze(farthest_pts)

def clip_plane_aabb(plane, aabb) :
    rect = np.zeros((2,3))
    b = plane[3]

    if plane[0] == 1 :
        rect[0, :] = [b, aabb[0, 1], aabb[0, 2]]
        rect[1, :] = [b, aabb[1, 1], aabb[1, 2]]
    elif plane[1] == 1 :
        rect[0, :] = [aabb[0, 0], b, aabb[0, 2]]
        rect[1, :] = [aabb[1, 0], b, aabb[1, 2]]
    else :
        rect[0, :] = [aabb[0, 0], aabb[0, 1], b]
        rect[1, :] = [aabb[1, 0], aabb[1, 1], b]

    return rect

def populate_point_buffer(pFolder) :
    for i, file in enumerate(glob.glob(pFolder)) :
        model = trimesh.load(file, file_type='ply', skip_materials=True)

        if i == 0 :
            vertices = model.vertices
        else :
            v = model.vertices
            vertices = np.concatenate([vertices, v], axis=0)

    return trimesh.Trimesh(vertices=vertices)

def populate_centroids(pMesh) :
    centroids = np.zeros(shape=(pMesh.faces.shape[0], 3), dtype=np.float32)

    for i, face in enumerate(pMesh.faces) :
        tr = [
            pMesh.vertices[face[0], :],
            pMesh.vertices[face[1], :],
            pMesh.vertices[face[2], :],]

        centroids[i] = (tr[0] + tr[1] + tr[2]) * 0.33

    return centroids