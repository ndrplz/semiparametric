"""
Miscellaneous geometric functions that have been helpful from time to time.
"""
from typing import Tuple
from typing import Union

import numpy as np
import scipy
import torch


def geodesic_distance(R1: np.ndarray,
                      R2: np.ndarray
                      ) -> float:
    """
    Return the geodesic distance between 3D rotation matrices

    See Also: //www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf

    Parameters
    ----------
    R1: ndarray
        Target rotation (shape 3x3)
    R2: ndarray
        Predicted rotation (shape 3x3)
    Returns
    -------
    d: float
        geodesic distance between R1 and R2
    """
    return np.linalg.norm(scipy.linalg.logm(R1.T @ R2), 'fro') / np.sqrt(2.)


def angles_from_zxz_dcm(dcm: np.ndarray,
                        clockwise: bool=False
                        ) -> Tuple[float, float, float]:
    """
    Porting of MatLab `dcm2angle(x, 'ZXZ')` function to get rotation
    angles from a 3x3 direction cosine matrix encoded using 'ZXZ' convention.

    :param dcm: Rotation matrix (DCM matrix) of shape 3x3 encoded in 'ZXZ'
    :param clockwise: Whether clockwise or counter-clockwise rotation is used. 
    :return (r1, r2, r3): Rotation angles around each axis (in radians). 
    """
    r11 = dcm[2, 0]
    r12 = dcm[2, 1]
    r21 = dcm[2, 2]
    r31 = dcm[0, 2]
    r32 = -dcm[1, 2]

    if clockwise:
        r12 *= -1
        r32 *= -1

    r1 = np.arctan2(r11, r12)
    r2 = np.arccos(r21)
    r3 = np.arctan2(r31, r32)

    return r1, r2, r3


def x_rot(alpha: float,
          clockwise: bool=False,
          pytorch: bool=False
          ) -> Union[np.ndarray, torch.Tensor]:
    """
    Compose a rotation matrix around X axis (default: counter-clockwise).

    :param alpha: Rotation angle in radians.
    :param clockwise: Default rotation convention is counter-clockwise.
     In case the `clockwise` flag is set, the sign of `sin(alpha)` is reversed
     to rotate clockwise.
    :param pytorch: In case the `pytorch` flag is set, all operation are
     between torch tensors and a torch.Tensor is returned .
    :return rot: Rotation matrix around X axis.
    """
    if pytorch:
        cx = torch.cos(alpha)
        sx = torch.sin(alpha)
    else:
        cx = np.cos(alpha)
        sx = np.sin(alpha)

    if clockwise:
        sx *= -1

    if pytorch:
        zero = torch.zeros(1)
        one = torch.ones(1)
        rot = torch.cat([torch.stack([one, zero, zero], dim=1),
                         torch.stack([zero, cx, -sx], dim=1),
                         torch.stack([zero, sx, cx], dim=1)], dim=0)
    else:
        rot = np.asarray([[1., 0., 0.],
                          [0., cx, -sx],
                          [0., sx, cx]], dtype=np.float32)
    return rot


def y_rot(alpha: float,
          clockwise: bool=False,
          pytorch: bool=False
          ) -> Union[np.ndarray, torch.Tensor]:
    """
    Compose a rotation matrix around Y axis (default: counter-clockwise).

    :param alpha: Rotation angle in radians.
    :param clockwise: Default rotation convention is counter-clockwise.
     In case the `clockwise` flag is set, the sign of `sin(alpha)` is reversed
     to rotate clockwise.
    :param pytorch: In case the `pytorch` flag is set, all operation are
     between torch tensors and a torch.Tensor is returned .
    :return rot: Rotation matrix around Y axis.
    """
    if pytorch:
        cy = torch.cos(alpha)
        sy = torch.sin(alpha)
    else:
        cy = np.cos(alpha)
        sy = np.sin(alpha)

    if clockwise:
        sy *= -1

    if pytorch:
        zero = torch.zeros(1)
        one = torch.ones(1)
        rot = torch.cat([torch.stack([cy, zero, sy], dim=1),
                         torch.stack([zero, one, zero], dim=1),
                         torch.stack([-sy, zero, cy], dim=1)], dim=0)
    else:
        rot = np.asarray([[cy, 0., sy],
                          [0., 1., 0.],
                          [-sy, 0., cy]], dtype=np.float32)
    return rot


def z_rot(alpha: float,
          clockwise: bool=False,
          pytorch: bool=False
          ) -> Union[np.ndarray, torch.Tensor]:
    """
    Compose a rotation matrix around Z axis (default: counter-clockwise).

    :param alpha: Rotation angle in radians.
    :param clockwise: Default rotation convention is counter-clockwise.
     In case the `clockwise` flag is set, the sign of `sin(alpha)` is reversed
     to rotate clockwise.
    :param pytorch: In case the `pytorch` flag is set, all operation are
     between torch tensors and a torch.Tensor is returned .
    :return rot: Rotation matrix around Z axis.
    """
    if pytorch:
        cz = torch.cos(alpha)
        sz = torch.sin(alpha)
    else:
        cz = np.cos(alpha)
        sz = np.sin(alpha)

    if clockwise:
        sz *= -1

    if pytorch:
        zero = torch.zeros(1)
        one = torch.ones(1)
        rot = torch.cat([torch.stack([cz, -sz, zero], dim=1),
                         torch.stack([sz, cz, zero], dim=1),
                         torch.stack([zero, zero, one], dim=1)], dim=0)
    else:
        rot = np.asarray([[cz, -sz, 0.],
                          [sz, cz, 0.],
                          [0., 0., 1.]], dtype=np.float32)

    return rot


def viewpoint_to_rot(viewpoint):
    """
    Black magic to transform Pascal3D+ viewpoint into rotation matrix.
    (See: readVpsDataPascal.m in Viewpoints and Keypoints repository)

    Note that this is NOT the matrix currently used to encode the ground truth.
    """
    azimuth, elevation, theta = map(np.radians, viewpoint[:-1])

    angles = [theta, elevation - np.pi / 2, -azimuth]

    c_ang = np.cos(angles)
    s_ang = np.sin(angles)

    R = np.full(shape=(3, 3), fill_value=np.nan)
    R[0, 0] = -s_ang[0] * c_ang[1] * s_ang[2] + c_ang[0] * c_ang[2]
    R[0, 1] = c_ang[0] * c_ang[1] * s_ang[2] + s_ang[0] * c_ang[2]
    R[0, 2] = s_ang[1] * s_ang[2]
    R[1, 0] = -s_ang[0] * c_ang[2] * c_ang[1] - c_ang[0] * s_ang[2]
    R[1, 1] = c_ang[0] * c_ang[2] * c_ang[1] - s_ang[0] * s_ang[2]
    R[1, 2] = s_ang[1] * c_ang[2]
    R[2, 0] = s_ang[0] * s_ang[1]
    R[2, 1] = -c_ang[0] * s_ang[1]
    R[2, 2] = c_ang[1]

    return R


def project_points(points_3d: np.array,
                   intrinsic: np.array,
                   extrinsic: np.array
                   ) -> np.array:
    """
    Project 3D points in 2D according to pinhole camera model.
    
    :param points_3d: 3D points to be projected (n_points, 3) 
    :param intrinsic: Intrinsics camera matrix
    :param extrinsic: Extrinsics camera matrix
    :return projected: 2D projected points (n_points, 2) 
    """
    n_points = points_3d.shape[0]

    assert points_3d.shape == (n_points, 3)
    assert extrinsic.shape == (3, 4) or extrinsic.shape == (4, 4)
    assert intrinsic.shape == (3, 3)

    if extrinsic.shape == (4, 4):
        if not np.all(extrinsic[-1, :] == np.asarray([0, 0, 0, 1])):
            raise ValueError('Format for extrinsic not valid')
        extrinsic = extrinsic[:3, :]

    points3d_h = np.concatenate([points_3d, np.ones(shape=(n_points, 1))], 1)

    projected = intrinsic @ extrinsic @ points3d_h.T
    projected /= projected[2, :]
    projected = projected.T
    return projected[:, :2]


def intrinsic_matrix(focal: float, cx: float, cy: float) -> np.ndarray:
    """
    Return intrinsics camera matrix with square pixel and no skew.

    :param focal: Focal length
    :param cx: X coordinate of principal point
    :param cy: Y coordinate of principal point
    :return K: intrinsics matrix of shape (3, 3)
    """
    return np.asarray([[focal, 0., cx],
                       [0., focal, cy],
                       [0., 0., 1.]])


def pascal_vpoint_to_extrinsics(az_deg: float,
                                el_deg: float,
                                radius: float):
    """
    Convert Pascal viewpoint to a camera extrinsic matrix which
     we can use to project 3D points from the CAD

    :param az_deg: Angle of rotation around X axis (degrees)
    :param el_deg: Angle of rotation around Y axis (degrees)
    :param radius: Distance from the origin
    :return extrinsic: Extrinsic matrix of shape (4, 4)
    """
    az_ours = np.radians(az_deg - 90)
    el_ours = np.radians(90 - el_deg)

    # Compose the rototranslation for a camera with look-at at the origin
    Rc = z_rot(az_ours) @ y_rot(el_ours)
    Rc[:, 0], Rc[:, 1] = Rc[:, 1].copy(), Rc[:, 0].copy()
    z_dir = Rc[:, -1] / np.linalg.norm(Rc[:, -1])
    Rc[:, -1] *= -1  # right-handed -> left-handed
    t = np.expand_dims(radius * z_dir, axis=-1)

    # Invert camera roto-translation to get the extrinsic
    #  see: http://ksimek.github.io/2012/08/22/extrinsic/
    extrinsic = np.concatenate([Rc.T, -(Rc.T) @ t], axis=1)
    return extrinsic
