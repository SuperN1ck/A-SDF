from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np

import trimesh
import pyrender
import pdb
import re
import math

from skimage.transform import rescale, resize
from scipy.ndimage import median_filter

### Copied From the notebook

def surface_normal(points, sH, sW):
    # These lookups denote y,x offsets from the anchor point for 8 surrounding
    # directions from the anchor A depicted below.
    #  -----------
    # | 7 | 6 | 5 |
    #  -----------
    # | 0 | A | 4 |
    #  -----------
    # | 1 | 2 | 3 |
    #  -----------
    d = 2
#     lookups = {0:(-d,0),1:(-d,d),2:(0,d),3:(d,d),4:(d,0),5:(d,-d),6:(0,-d),7:(-d,-d)}

    lookups = {0:(0,-d),1:(d,-d),2:(d,0),3:(d,d),4:(0,d),5:(-d,d),6:(-d,0),7:(-d,-d)}

    surface_normals = np.zeros((sH,sW,3))
    for i in range(sH):
        for j in range(sW):
            min_diff = None
            point1 = points[i,j,:3]
             # We choose the normal calculated from the two points that are
             # closest to the anchor points.  This helps to prevent using large
             # depth disparities at surface borders in the normal calculation.
            for k in range(8):
                try:
                    point2 = points[i+lookups[k][0],j+lookups[k][1],:3]
                    point3 = points[i+lookups[(k+2)%8][0],j+lookups[(k+2)%8][1],:3]
                    diff = np.linalg.norm(point2 - point1) + np.linalg.norm(point3 - point1)
                    if min_diff is None or diff < min_diff:
                        normal = normalize(np.cross(point2-point1,point3-point1))
                        min_diff = diff
                except IndexError:
                    continue
            surface_normals[i,j,:3] = normal
    return surface_normals

def normalize(v):
    return v/np.linalg.norm(v)

def depth_to_surface_normal_opencv_projection(depth, intrinsics, extrinsics, cls, seq, center, scale=0.25):
    depth_map = depth.copy()
    H, W = depth.shape
    sH, sW = int(scale*H), int(scale*W)
    depth_map[depth < 0.0001] = 50.0

    # Each 'pixel' containing the 3D point in camera coords
    depth_in_world = depth2world(depth_map, intrinsics, extrinsics, cls, seq, center, True)[:,:3].reshape(H,W,3)
    surface_normals = surface_normal(depth_in_world[::int(1/scale),::int(1/scale),:], sH, sW)
    surface_normals = resize(surface_normals, (H, W), anti_aliasing=True)
    return surface_normals

def R_z(theta):
    theta = theta/180*np.pi
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0,1]])
def R_y(theta):
    theta = theta/180*np.pi
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta),0,np.cos(theta)]])
def R_x(theta):
    theta = theta/180*np.pi
    return np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0,np.sin(theta), np.cos(theta)]])

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def quat2rot(quat):
    mtx = np.zeros((4,4))
    mtx[3,3] = 1
    r = R.from_quat(quat[3:])
    rot = r.as_matrix()
    mtx[:3,:3] = rot
    mtx[:3,3] =  quat[:3]
    return mtx

def ext_test(cam_ext_file, time):
    # ['/map', 'CreditCardWith4Markers']
    # ['/map', 'AsusXtionCameraFrame']
    # ['/map', 'TransparentBoxWith4Markers']

    # ['/AsusXtionCameraFrame', '/camera_link']
    # ['/camera_link', '/camera_rgb_frame']
    # ['/camera_link', '/camera_depth_frame']
    # ['/camera_depth_frame', '/camera_depth_optical_frame']

    quat = np.zeros(7)

    Asus2cam_quat_list = list()
    cam2depth_quat_list = list()
    cam2rgb_quat_list = list()
    depth2optical_quat_list = []
    map2Asus_quat_list = []

    Asus2cam_time_list = []
    cam2depth_time_list = []
    cam2rgb_time_list = []
    depth2optical_time_list = []
    map2Asus_time_list = []

    cams = ['field.transforms0.header.frame_id', 'field.transforms0.child_frame_id']
    
    pose = ['field.transforms0.transform.translation.x','field.transforms0.transform.translation.y',
            'field.transforms0.transform.translation.z','field.transforms0.transform.rotation.x',
            'field.transforms0.transform.rotation.y','field.transforms0.transform.rotation.z',
            'field.transforms0.transform.rotation.w']
    
    with open(cam_ext_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for i, p in enumerate(pose):
                quat[i] = float(row[p])

            if row[cams[0]]=='/AsusXtionCameraFrame' and row[cams[1]]=='/camera_link':
                Asus2cam_quat_list.append(np.array(list(quat)))
                Asus2cam_time_list.append(float(row['%time']))
            elif row[cams[0]]=='/camera_link' and row[cams[1]]=='/camera_depth_frame':
                cam2depth_quat_list.append(np.array(list(quat)))
                cam2depth_time_list.append(float(row['%time']))
            elif row[cams[0]]=='/camera_link' and row[cams[1]]=='/camera_rgb_frame':
                cam2rgb_quat_list.append(np.array(list(quat)))
                cam2rgb_time_list.append(float(row['%time']))
            elif row[cams[0]]=='/camera_depth_frame' and row[cams[1]]=='/camera_depth_optical_frame':
                depth2optical_quat_list.append(np.array(list(quat)))
                depth2optical_time_list.append(float(row['%time']))
            elif row[cams[0]]=='/map' and row[cams[1]]=='AsusXtionCameraFrame':
                map2Asus_quat_list.append(np.array(list(quat)))
                map2Asus_time_list.append(float(row['%time']))
            else:
                continue
            

    idx = find_nearest(Asus2cam_time_list, time)
    Asus2cam = quat2rot(Asus2cam_quat_list[idx])
    
    idx = find_nearest(cam2depth_time_list, time)
    cam2depth = quat2rot(cam2depth_quat_list[idx])
    
    idx = find_nearest(cam2rgb_time_list, time)
    cam2rgb = quat2rot(cam2rgb_quat_list[idx])
    
    idx = find_nearest(depth2optical_time_list, time)
    depth2optical = quat2rot(depth2optical_quat_list[idx])
    
    idx = find_nearest(map2Asus_time_list, time)
    map2Asus = quat2rot(map2Asus_quat_list[idx])
    
    map2optical = map2Asus@Asus2cam@cam2depth@depth2optical

    return np.linalg.inv(map2optical)

        
def pose_test(pose_file, time):
    marker0_list = np.zeros(7)
    marker1_list = np.zeros(7)
    marker2_list = np.zeros(7)

    marker0 = np.zeros(7)
    marker1 = np.zeros(7)
    marker2 = np.zeros(7)
    
    time_list = []

    pose = ['pose.position.x', 'pose.position.y', 'pose.position.z',
                     'pose.orientation.x', 'pose.orientation.y', 'pose.orientation.z', 'pose.orientation.w']

    with open(pose_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for j, row in enumerate(reader):
            if j<635:
                for i, p in enumerate(pose):

                    marker0[i] = float(row['field.markers0.' + p])
                    marker1[i] = float(row['field.markers1.' + p])
                    marker2[i] = float(row['field.markers2.' + p])
                marker0_list = np.vstack((marker0_list, marker0))
                marker1_list = np.vstack((marker1_list, marker1))
                marker2_list = np.vstack((marker2_list, marker2))
                time_list.append(float(row['%time']))
        marker0_list = marker0_list[1:,:]
        marker1_list = marker1_list[1:,:]
        marker2_list = marker2_list[1:,:]

    idx = find_nearest(time_list, time)
    map2obj = quat2rot(marker2_list[idx])
    
    return map2obj


def int_test(cam_int_file):
    
    K = np.zeros(9)
    K_items = ['field.K0', 'field.K1', 'field.K2', 'field.K3', 'field.K4', 'field.K5',
         'field.K6', 'field.K7', 'field.K8']
    
    with open(cam_int_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for i, K_item in enumerate(K_items):
                K[i] = float(row[K_item])
            return K.reshape(3,3)

def js_test(js_int_file, time):

    time_list = []
    js_list = []
    
    with open(js_int_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            js = int(float(row['field.position0'])/np.pi*180)
            t = float(row['%time'])
            js_list.append(js)
            time_list.append(t)
        idx = find_nearest(time_list, time)
    return js_list[idx]
    
def depth2world(depth_map, intrinsic_param, extrinsic_param, cls, seq, center, return_full=False):

    # Get world coords
    H, W = depth_map.shape

    WS = np.repeat(np.linspace(1 / (2 * W), 1 - 1 / (2 * W), W).reshape([1, -1]), H, axis=0)
    HS = np.repeat(np.linspace(1 / (2 * H), 1 - 1 / (2 * H), H).reshape([-1, 1]), W, axis=1)

    pixel_coords = np.stack([WS*W, HS*H, np.ones(depth_map.shape)], 2)
    pixel_coords = pixel_coords.reshape(-1, 3).T
    depth_map = depth_map.reshape(-1,1).T
    
    cam_coords = np.linalg.inv(intrinsic_param)@(pixel_coords)
    cam_coords *= depth_map
    
    cam_coords = np.vstack([cam_coords, np.ones((1,cam_coords.shape[1]))])
    world_coords = np.linalg.inv(extrinsic_param)@cam_coords
    
    world_coords = world_coords.T

    if return_full==False:
        mask = np.repeat(depth_map.copy(), 4, axis=0).T
        world_coords = world_coords[mask>0].reshape(-1,4)
        world_coords = alignment(cls, seq, center, world_coords)
    else:
        world_coords = alignment(cls, seq, center, world_coords)

    return world_coords


# specify alignment
def alignment(cls, seq, center, world_coords):
    align = np.eye(4)

    # scale to center
#     if center is None:
#         c = np.mean(world_coords, axis = 0)
#     else:
#         c = center[seq]
    if seq not in center.keys():
        c = np.mean(world_coords, axis = 0)
        center[seq] = c
    else:
        c = center[seq]
    print(np.mean(world_coords, axis = 0))
    print("c: ", c)
    world_coords = world_coords - c
#     scale = 1/np.max(np.abs(world_coords))
    
    if cls == 'laptop':
        world_coords *= 2.5
        world_coords = world_coords.T
        x = -5
        y = 310
        z = 120
        align[:3,:3] = R_z(z)@R_y(y)@R_x(x)
        world_coords = align@world_coords
        world_coords = world_coords.T
        world_coords[:,0] += 0.1
        world_coords[:,1] += 0.0
        world_coords[:,2] -= 0.25

    return world_coords


def generate_SDF(world_coords, surface_normals, eta=0.025):
    pos_pts_world = world_coords.copy()
    neg_pts_world = world_coords.copy()
    # purturb with surface normal
    for idx in range(world_coords.shape[0]):
        pos_pts_world[idx] += eta * np.array(surface_normals[idx])
        neg_pts_world[idx] -= eta * np.array(surface_normals[idx])

    eta_vec = np.ones((pos_pts_world.shape[0], 1)) * eta
    part = np.zeros((pos_pts_world.shape[0], 1))
    pos = np.hstack([pos_pts_world, eta_vec, part])
    neg = np.hstack([neg_pts_world, -eta_vec, part])
    
    return pos, neg
