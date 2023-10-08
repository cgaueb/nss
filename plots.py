import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_aabb(ax, aabb, alpha=0.1, facecolor=(0, 0, 1)) :
    corner1 = (aabb[0, 0], aabb[0, 1], aabb[0, 2])
    corner2 = (aabb[0, 0], aabb[1, 1], aabb[0, 2])
    corner3 = (aabb[1, 0], aabb[0, 1], aabb[0, 2])
    corner4 = (aabb[0, 0], aabb[0, 1], aabb[1, 2])
    cube_definition = [corner1, corner2, corner3, corner4]

    cube_definition_array = [np.array(list(item)) for item in cube_definition]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]]

    faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
    #faces.set_facecolor(facecolor)
    faces.set_alpha(alpha)
    ax.add_collection3d(faces)
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0)
    return ax

def plot_plane(ax, plane, clr = (0, 0, 1, 0.3)) :
    normal = plane[0:3]
    d = plane[3]
    a, b = np.meshgrid(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1))

    if normal[0] != 0 :
        c = (d + a * normal[1] + b * normal[2]) / normal[0]
        ax.plot_surface(c, a, b, color=clr)
    elif normal[1] != 0 :
        c = (d + a * normal[0] + b * normal[2]) / normal[1]
        ax.plot_surface(a, c, b, color=clr)
    else :
        c = (d + a * normal[0] + b * normal[1]) / normal[2]
        ax.plot_surface(a, b, c, color=clr)

    return ax