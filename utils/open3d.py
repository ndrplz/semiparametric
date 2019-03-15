from pathlib import Path
from typing import Dict
from typing import Sequence
from typing import Tuple
from typing import Union

from open3d import *

from datasets.interop import pascal_idx_to_kpoint
from datasets.interop import pascal_stick_planes


Kpoint3D = Sequence[float]


def color_mesh_from_obj(mesh: TriangleMesh,
                        obj_name: Union[str, Path],
                        mtl_to_color: dict,
                        base_color=(0.2, 0.2, 0.2)):
    """
    Color a open3d TriangleMesh according to material information in obj file
    """

    mesh.paint_uniform_color(base_color)

    vertices_colors = np.asarray(mesh.vertex_colors)

    with open(obj_name, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    # Row-index in the file where each new material begins
    mtl_start_indexes = [i for i, l in enumerate(lines) if 'usemtl' in l]

    for i in range(len(mtl_start_indexes)):
        index = mtl_start_indexes[i]
        name = lines[index].split(' ')[-1]
        if name not in mtl_to_color.keys():
            continue
        if index == mtl_start_indexes[-1]:
            lines_mat = lines[index + 2:]  # remove also material header
        else:
            lines_mat = lines[index + 2: mtl_start_indexes[i + 1]]
        for l in lines_mat:
            l = l[1:]  # remove `f`
            l = l.replace('//', ' ')  # separate just with space
            face_vertices = np.fromstring(l, dtype=np.int, sep=' ')
            face_vertices = face_vertices[::2] - 1  # obj vertices start from 1

            # Assign vertex color to all vertices belonging to current face
            for index_v in face_vertices:
                vertices_colors[index_v] = mtl_to_color[name]

    mesh.vertex_colors = Vector3dVector(vertices_colors)


def stick_line_sets(kpoint_array: np.ndarray, pascal_class: str):

    kpoints_3d_dict = {
        pascal_idx_to_kpoint[pascal_class][i]: kpoint_array[i] for i in range(kpoint_array.shape[0])
    }

    s_planes = pascal_stick_planes[pascal_class]
    l1 = draw_segments(s_planes['left'], kpoints_3d_dict, color=(0, 255, 0))
    l2 = draw_segments(s_planes['right'], kpoints_3d_dict, color=(255, 0, 0))
    l3 = draw_segments(s_planes['wheel'], kpoints_3d_dict, color=(0, 0, 255))
    l4 = draw_segments(s_planes['light'], kpoints_3d_dict, color=(0, 0, 255))
    l5 = draw_segments(s_planes['roof'], kpoints_3d_dict, color=(0, 0, 255))
    return [l1, l2, l3, l4, l5]


def draw_segments(segments: Sequence[Tuple[str, str]],
                  kpoints_3d: Dict[str, Kpoint3D],
                  color: Tuple[int, int, int])->LineSet:
    line_set = LineSet()
    n_lines = len(segments)
    line_set.colors = Vector3dVector([color] * n_lines)

    vertices_names_uniques = np.unique(np.asarray(segments))

    points = []
    for v_name in vertices_names_uniques:
        points.append(kpoints_3d[v_name])
    line_set.points = Vector3dVector(points)

    lines = []
    for vertex_a, vertex_b in segments:
        start = int(np.where(vertices_names_uniques == vertex_a)[0])
        end = int(np.where(vertices_names_uniques == vertex_b)[0])
        lines.append([start, end])
    line_set.lines = Vector2iVector(lines)
    return line_set
