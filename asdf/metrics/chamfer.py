#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh
import torch
import pytorch3d


def compute_trimesh_chamfer(gt_points, gen_mesh, offset, scale, num_mesh_samples=30000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """

    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    gt_points_np = gt_points.vertices
    gt_points_np = (gt_points_np + offset) * scale

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    print(gt_to_gen_chamfer + gen_to_gt_chamfer, gt_to_gen_chamfer, gen_to_gt_chamfer)
    return gt_to_gen_chamfer + gen_to_gt_chamfer


def compute_pytorch3d_chamfer(
    gt_points, gen_mesh, offset, scale, num_mesh_samples=30000
):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """

    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    gen_points_sampled = gen_points_sampled / scale - offset

    # only need numpy array of points
    # gt_points_np = gt_points.vertices
    gt_points_np = gt_points.vertices

    gt_points_torch = torch.Tensor(gt_points_np).cuda().unsqueeze(0)

    # Not needed, we will sample as many points as we have in GT
    # gt_points_torch, _ = pytorch3d.ops.sample_farthest_points(
    #   gt_points_torch, K=num_mesh_samples
    # )

    pred_points_torch = torch.Tensor(gen_points_sampled).cuda().unsqueeze(0)
    cd, _ = pytorch3d.loss.chamfer_distance(gt_points_torch, pred_points_torch)
    cd = float(cd)
    # print(f"Pytorch: {cd}")
    return cd


def compute_depth_chamfer(gt_points, gen_mesh, num_mesh_samples=30000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """

    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    # gen_points_sampled = gen_points_sampled / scale - offset

    # only need numpy array of points
    # gt_points_np = gt_points.vertices
    gt_points_np = gt_points

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    print(gt_to_gen_chamfer + gen_to_gt_chamfer, gt_to_gen_chamfer, gen_to_gt_chamfer)
    # return gt_to_gen_chamfer + gen_to_gt_chamfer
    return gt_to_gen_chamfer
