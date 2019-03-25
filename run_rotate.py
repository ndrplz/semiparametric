"""
Open3D-based script to tinker with model predictions interactively.
"""
import argparse
import collections
from pathlib import Path

import cv2
import torch
from open3d import *  # watch out, this also contains `Image`
from PIL import Image
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor

from datasets.dataset_texture import TextureDatasetWithNormal
from datasets.dataset_texture import get_planes
from datasets.dataset_texture import warp_unwarp_planes
from datasets.interop import pascal_idx_to_kpoint
from datasets.interop import pascal_parts_colors
from model.von import G_Resnet
from utils.geometry import intrinsic_matrix
from utils.geometry import pascal_vpoint_to_extrinsics
from utils.geometry import project_points
from utils.normalization import to_image
from utils.open3d import color_mesh_from_obj


def align_view(vis: VisualizerWithKeyCallback,
               focal: int, extrinsic: np.ndarray):
    """ Implement look-at to the origin """

    pinhole_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

    # Get view controller intrinsics
    intrinsic = pinhole_params.intrinsic
    w, h = intrinsic.width, intrinsic.height
    cx, cy = intrinsic.get_principal_point()
    intrinsic.set_intrinsics(w, h, focal, focal, cx, cy)

    # Use current camera extrinsics to update view
    pinhole_params.extrinsic = np.concatenate([extrinsic, np.asarray([[0, 0, 0, 1]])])
    vis.get_view_control().convert_from_pinhole_camera_parameters(pinhole_params)
    vis.update_geometry()


class Geometries(dict):
    def __init__(self):
        super(Geometries, self).__init__()

    def as_list(self):
        l = []
        for v in self.values():
            if isinstance(v, collections.Iterable):
                l.extend(v)
            else:
                l.append(v)
        return l


class Callbacks(object):
    def __init__(self, key: int):
        self.key = key

    def __call__(self, vis):

        global state
        global kpoints_3d
        global normal_vertex_colors
        global part_vertex_colors
        global texture_src

        # Do nothing
        if self.key == 0:
            pass
        # Dump current output window
        elif self.key == ord('X'):
            args.dump_dir.mkdir(exist_ok=True)
            el = int(state['angle_y'])
            az = int(state['angle_z'])
            rad = int(state['radius'])
            d_id = state['dump_id']
            id_str = f'{d_id:03d}_el_{el:03d}_az_{az:03d}_rad_{rad:03d}'
            cv2.imwrite(str(args.dump_dir / f'{id_str}.png'),
                        state['dump_image'])
            state['dump_id'] += 1
        # Override open3D reset
        elif self.key == ord('R'):
            pass
        # Rotation around Y axis
        elif self.key == ord('F'):
            state['angle_y'] += 5
        elif self.key == ord('D'):
            state['angle_y'] -= 5
        # Rotation around Z axis
        elif self.key == ord('S'):
            state['angle_z'] += 5
        elif self.key == ord('A'):
            state['angle_z'] -= 5
        # Distance from origin (+)
        elif self.key == ord('H'):
            state['radius'] += 0.05
        # Distance from origin (-)
        elif self.key == ord('G'):
            state['radius'] -= 0.05
        # Next dataset example
        elif self.key == ord(' '):
            texture_src = state['dataset'][state['dataset_index']]
            state['dataset_index'] += 1
        # Next CAD model
        elif self.key == ord('N'):
            state['cad_idx'] += 1
            if state['cad_idx'] == 10:
                state['cad_idx'] = 0

            # --------------- Update model and Kpoints 3D ----------------------
            cad_idx = state['cad_idx']
            model_path = args.CAD_root / f'pascal_car_cad_{cad_idx:03d}.ply'
            kpoints_3d = state['car_cad_kpoints_3D'][cad_idx].astype(np.float32)
            car_mesh = read_triangle_mesh(str(model_path))

            # ------------------- Compute normal Colors ----------------------
            car_mesh.compute_vertex_normals()
            normal_vertex_colors = (np.asarray(car_mesh.vertex_normals) + 1) / 2.

            # -------------------- Compute Parts Colors ------------------------
            color_dict = pascal_parts_colors['car']
            obj_path = model_path.parent / (model_path.stem + '.obj')
            color_mesh_from_obj(car_mesh, obj_name=obj_path, mtl_to_color=color_dict,
                                base_color=color_dict['car'])
            part_vertex_colors = np.asarray(car_mesh.vertex_colors).copy()
            # Reset colors to normals
            car_mesh.vertex_colors = Vector3dVector(normal_vertex_colors)

            if 'car_mesh' not in state['geometries']:
                state['geometries']['car_mesh'] = car_mesh
            else:
                state['geometries']['car_mesh'].vertices = car_mesh.vertices
                state['geometries']['car_mesh'].vertex_colors = car_mesh.vertex_colors
                state['geometries']['car_mesh'].vertex_normals = car_mesh.vertex_normals
                state['geometries']['car_mesh'].triangles = car_mesh.triangles

        else:
            raise NotImplementedError()

        # -------------------------- Move Camera ------------------------------
        angle_y = np.clip(state['angle_y'], -90, 90)
        radius = np.clip(state['radius'], 0, state['radius'])
        pascal_az = (state['angle_z'] + 90) % 360

        intrinsic = intrinsic_matrix(state['focal'], cx=img_w/2, cy=img_h/2)

        if args.verbose:
            print(f'Azimuth:{pascal_az} Elevation:{angle_y} Radius:{radius}')

        extrinsic = pascal_vpoint_to_extrinsics(az_deg=pascal_az,
                                                el_deg=90 - angle_y,
                                                radius=radius)

        # ---------------------- Start Rendering ------------------------------
        if not vis.get_render_option() or not vis.get_view_control():
            vis.update_geometry()  # we don't have anything, return
            return
        align_view(vis, focal=state['focal'], extrinsic=extrinsic)
        vis.get_render_option().mesh_color_option = MeshColorOption.Color
        vis.get_render_option().light_on = False
        vis.get_render_option().background_color = (0, 0, 0)

        # ---------------------- Normal ---------------------------------------
        src_shaded = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        src_shaded = (src_shaded * 255).astype(np.uint8)
        if args.LAB:
            src_shaded = cv2.cvtColor(src_shaded, cv2.COLOR_RGB2LAB)
        else:
            src_shaded = cv2.cvtColor(src_shaded, cv2.COLOR_RGB2BGR)

        # ---------------------- Parts ----------------------------------------
        state['geometries']['car_mesh'].vertex_colors = Vector3dVector(part_vertex_colors)
        vis.update_geometry()
        src_parts = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        src_parts = (src_parts * 255).astype(np.uint8)
        if args.LAB:
            src_parts = cv2.cvtColor(src_parts, cv2.COLOR_RGB2LAB)
        else:
            src_parts = cv2.cvtColor(src_parts, cv2.COLOR_RGB2BGR)

        state['geometries']['car_mesh'].vertex_colors = Vector3dVector(normal_vertex_colors)
        vis.update_geometry()

        # Project model kpoints in 2D
        kpoints_2d_step = project_points(kpoints_3d, intrinsic, extrinsic)
        kpoints_2d_step /= (img_w, img_h)
        kpoints_2d_step = np.clip(kpoints_2d_step, -1, 1)
        kpoints_2d_step_dict = {pascal_idx_to_kpoint['car'][i]: kpoints_2d_step[i]
                                for i in range(kpoints_2d_step.shape[0])}

        meta = {'kpoints_2d': kpoints_2d_step_dict,
                'vpoint': [pascal_az]}

        _, dst_kpoints_planes, dst_visibilities = get_planes(np.zeros((img_h, img_w, 3)),
                                                             meta=meta, pascal_class='car')
        src_planes = np.asarray([to_image(i, from_LAB=args.LAB) for i in texture_src['planes']])
        src_kpoints_planes = texture_src['src_kpoints_planes']
        src_visibilities = texture_src['src_vs']

        planes_warped, planes_unwarped = warp_unwarp_planes(src_planes=src_planes,
                                                            src_planes_kpoints=src_kpoints_planes,
                                                            dst_planes_kpoints=dst_kpoints_planes,
                                                            src_visibilities=src_visibilities,
                                                            dst_visibilities=dst_visibilities,
                                                            pascal_class='car')

        planes_warped = TextureDatasetWithNormal.planes_to_torch(planes_warped, to_LAB=args.LAB)
        planes_warped = planes_warped.reshape(1, planes_warped.shape[0] * planes_warped.shape[1],
                                              planes_warped.shape[2], planes_warped.shape[3])

        src_shaded_input = Normalize(mean=[0.5]*3, std=[0.5]*3)(ToTensor()((Image.fromarray(src_shaded))))
        src_parts_input = Normalize(mean=[0.5]*3, std=[0.5]*3)(ToTensor()((Image.fromarray(src_parts))))
        src_central = texture_src['src_central']

        gen_in_src = torch.cat([src_shaded_input.unsqueeze(0), src_parts_input.unsqueeze(0),
                                src_central.unsqueeze(0), planes_warped], dim=1)
        net_image = to_image(state['net'](gen_in_src.to(args.device))[0], from_LAB=args.LAB)
        out_image = np.concatenate([to_image(texture_src['src_image'],
                                                        from_LAB=args.LAB),
                                    to_image(src_shaded_input, from_LAB=args.LAB),
                                    to_image(src_parts_input, from_LAB=args.LAB),
                                    to_image(src_central, from_LAB=args.LAB),
                                    net_image], axis=1)
        state['dump_image'] = out_image

        cv2.imshow('Output', out_image)
        cv2.waitKey(20)
        vis.update_geometry()


def run(args: argparse.Namespace):

    global state

    # Initialize state
    state = {
        'angle_y': 90.,
        'angle_z': 0.,
        'cad_idx': 0,
        'car_cad_kpoints_3D': np.load(args.CAD_root / 'cad_car.npz')['kpoints'],
        'dataset_index': 0,
        'dump_id': 0,
        'focal': 1000,
        'geometries': Geometries(),
        'radius': 7
    }

    # Load pre-trained model
    net = G_Resnet(input_nc=24).to(args.device)
    net.load_state_dict(torch.load(args.model_path))
    net.eval()

    state['net'] = net

    # Load test dataset
    dataset = TextureDatasetWithNormal(folder=args.texture_dataset_dir,
                                       resize_factor=0.5,
                                       demo_mode=args.demo,
                                       use_LAB=args.LAB)
    dataset.eval()
    state['dataset'] = dataset

    key_callbacks = {
        ord('F'): Callbacks(ord('F')),
        ord('D'): Callbacks(ord('D')),
        ord('A'): Callbacks(ord('A')),
        ord('S'): Callbacks(ord('S')),
        ord('H'): Callbacks(ord('H')),
        ord('G'): Callbacks(ord('G')),
        ord(' '): Callbacks(ord(' ')),
        ord('N'): Callbacks(ord('N')),
        ord('O'): Callbacks(ord('O')),
        ord('P'): Callbacks(ord('P')),
        ord('X'): Callbacks(ord('X')),
    }

    # Init callbacks
    Callbacks(ord('N'))(Visualizer())  # init model
    Callbacks(ord(' '))(Visualizer())  # init appearance
    Callbacks(0)(Visualizer())
    draw_geometries_with_key_callbacks(state['geometries'].as_list(),
                                       key_callbacks,
                                       width=img_w, height=img_h,
                                       left=50, top=1080//4)
    cv2.namedWindow('Projection')


if __name__ == '__main__':

    img_h = img_w = 128

    parser = argparse.ArgumentParser()
    parser.add_argument('texture_dataset_dir', type=Path,
                        help='Texture dataset directory')
    parser.add_argument('model_path', type=Path,
                        help='Path to pre-trained model')
    parser.add_argument('CAD_root', type=Path,
                        help='Directory containing 3D CAD')
    parser.add_argument('--dump_dir', type=Path, default=Path('/tmp'),
                        help='Directory to save output')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda',
                        help='Device used for model inference')
    parser.add_argument('--demo', action='store_true',
                        help='Load a subset of dataset - faster to load.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print spherical coordinates')
    args = parser.parse_args()

    args.LAB = True  # Expected arg for released model

    # Load only the first 100 examples - faster to load. Keep to False to
    #  iterate over the whole test set.
    args.demo = True

    # Sanity-checks
    if not args.model_path.is_file():
        raise OSError('Please provide a valid file for pretrained weights.')
    if not args.CAD_root.is_dir():
        raise OSError('Please provide a valid CAD root.')

    # Print help to console
    with open('./help.txt') as help_file:
        print(help_file.read())

    # Start the GUI
    run(args)
