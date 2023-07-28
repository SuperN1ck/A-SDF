#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data

import asdf.workspace as ws
import re
import tqdm
import math

from typing import List

import CARTO.Decoder
import pathlib


def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """ "Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename, articulation=False, num_atc_parts=1):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])
    if articulation == True:
        if num_atc_parts == 1:
            atc = torch.Tensor([float(re.split("/", filename)[-1][7:11])])
            instance_idx = int(re.split("/", filename)[-1][:4])
            return ([pos_tensor, neg_tensor], atc, instance_idx)
        if num_atc_parts == 2:
            atc1 = torch.Tensor([float(re.split("/", filename)[-1][7:11])])
            atc2 = torch.Tensor([float(re.split("/", filename)[-1][11:15])])
            instance_idx = int(re.split("/", filename)[-1][:4])
            return ([pos_tensor, neg_tensor], torch.Tensor([atc1, atc2]), instance_idx)
    else:
        return [pos_tensor, neg_tensor]


def read_sdf_samples_into_ram_rbo(filename, articulation=False, num_atc_parts=1):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])
    if articulation == True:
        if num_atc_parts == 1:
            atc = torch.Tensor([float(re.split("/", filename)[-1][-8:-4])])
            instance_idx = int(re.split("/", filename)[-1][:4])
            return ([pos_tensor, neg_tensor], atc, instance_idx)
    else:
        return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, subsample=None, articulation=False, num_atc_parts=1):
    # Entries
    # :, 0:3 xyz
    # :, 3 sdf
    # :, 4 part
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(min(subsample, pos_tensor.shape[0] + neg_tensor.shape[0]) / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    if articulation == True:
        if num_atc_parts == 1:
            atc = torch.Tensor([float(re.split("/", filename)[-1][7:11])])
            instance_idx = int(re.split("/", filename)[-1][:4])
            return (samples, atc, instance_idx)
        if num_atc_parts == 2:
            atc1 = torch.Tensor([float(re.split("/", filename)[-1][7:11])])
            atc2 = torch.Tensor([float(re.split("/", filename)[-1][11:15])])
            instance_idx = int(re.split("/", filename)[-1][:4])
            return (samples, torch.Tensor([atc1, atc2]), instance_idx)
    else:
        return samples


def unpack_sdf_samples_from_ram(
    data, subsample=None, articulation=False, joint_type=False, num_atc_parts=1
):
    if subsample is None:
        return data
    if articulation == True:
        pos_tensor = data[0][0]
        neg_tensor = data[0][1]
        atc = data[1]
        instance_idx = data[2]
        if joint_type:
            jt = data[3]
    else:
        pos_tensor = data[0]
        neg_tensor = data[1]

    # split the sample into half
    half = int(min(subsample, pos_tensor.shape[0] + neg_tensor.shape[0]) / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    # pos_start_ind = random.randint(0, pos_size - half)
    # sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if pos_size <= half:
        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    else:
        pos_start_ind = random.randint(0, pos_size - half)
        sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    if articulation == True:
        if not joint_type:
            return (samples, atc, instance_idx)
        else:
            return (samples, atc, instance_idx, jt)
    else:
        return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
        articulation=False,
        num_atc_parts=1,
    ):
        self.subsample = subsample

        self.data_source = data_source

        if not isinstance(split, List):
            split = [split]

        self.npyfiles = []
        for split_ in split:
            self.npyfiles.extend(get_instance_filenames(data_source, split_))

        self.articualtion = articulation
        self.num_atc_parts = num_atc_parts

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + str(data_source)
        )

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

                if self.articualtion == True:
                    if self.num_atc_parts == 1:
                        atc = torch.Tensor([float(re.split("/", filename)[-1][7:11])])
                        instance_idx = int(re.split("/", filename)[-1][:4])
                        self.loaded_data.append(
                            (
                                [
                                    pos_tensor[torch.randperm(pos_tensor.shape[0])],
                                    neg_tensor[torch.randperm(neg_tensor.shape[0])],
                                ],
                                atc,
                                instance_idx,
                            )
                        )
                    if self.num_atc_parts == 2:
                        atc1 = torch.Tensor([float(re.split("/", filename)[-1][7:11])])
                        atc2 = torch.Tensor([float(re.split("/", filename)[-1][11:15])])
                        instance_idx = int(re.split("/", filename)[-1][:4])
                        self.loaded_data.append(
                            (
                                [
                                    pos_tensor[torch.randperm(pos_tensor.shape[0])],
                                    neg_tensor[torch.randperm(neg_tensor.shape[0])],
                                ],
                                [atc1, atc2],
                                instance_idx,
                            )
                        )

                else:
                    self.loaded_data.append(
                        [
                            pos_tensor[torch.randperm(pos_tensor.shape[0])],
                            neg_tensor[torch.randperm(neg_tensor.shape[0])],
                        ],
                    )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(
                    self.loaded_data[idx],
                    self.subsample,
                    self.articualtion,
                    self.num_atc_parts,
                ),
                idx,
            )
        else:
            return (
                unpack_sdf_samples(
                    filename, self.subsample, self.articualtion, self.num_atc_parts
                ),
                idx,
            )


class OurSDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split_file_name,
        num_samp_per_scene,
        base_path,
        file_id_selection="train",
        load_ram=False,
        articulation=True,
        return_joint_type=False,
        return_pos_neg_split: bool = False,
    ):
        base_path = pathlib.Path(base_path)

        gen_cfg: CARTO.Decoder.config.GenerationConfig = CARTO.Decoder.utils.load_cfg(
            base_path / data_source, cfg_class=CARTO.Decoder.config.GenerationConfig
        )
        max_extents = gen_cfg.max_extent
        assert max_extents > 0.0, "max_extents incorrect"
        rescaler = CARTO.Decoder.data.dataset.Rescaler3D(scale=max_extents)

        split_dicts = CARTO.Decoder.data.dataset.get_dataset_split_dict(
            base_path / data_source, split_file_name
        )

        file_ids = split_dicts[file_id_selection]
        self.file_ids = [base_path / file_id for file_id in file_ids]

        self.our_sdf_dataset = CARTO.Decoder.data.dataset.SDFDataset(
            self.file_ids,
            subsample=num_samp_per_scene,
            cache_in_ram=load_ram,
            rescaler=rescaler,
        )
        self.articulation = articulation
        self.return_joint_type = return_joint_type
        self.return_pos_neg_split = return_pos_neg_split

        # Starting at 1 here as in the original A-SDF code we do -1
        self.object_index = CARTO.Decoder.utils.IndexDict(counter_start=1)
        for datapoint in tqdm.tqdm(self.our_sdf_dataset):
            self.object_index[datapoint.object_id]

    def __len__(self):
        return len(self.our_sdf_dataset)

    def __getitem__(self, index):
        datapoint: CARTO.Decoder.data.dataset.DataPoint = self.our_sdf_dataset[index]

        sdf_vals = np.expand_dims(datapoint.sdf_values, -1)
        points = datapoint.points
        parts = np.zeros_like(sdf_vals)

        if self.return_pos_neg_split:
            pos_mask = (sdf_vals > 0)[:, 0]

            pos_points = np.copy(points[pos_mask])
            neg_points = np.copy(points[~pos_mask])
            pos_sdf = np.copy(sdf_vals[pos_mask])
            neg_sdf = np.copy(sdf_vals[~pos_mask])
            pos_parts = np.copy(parts[pos_mask])
            neg_parts = np.copy(parts[~pos_mask])

            data_tensor = [
                torch.Tensor(np.concatenate([pos_points, pos_sdf, pos_parts], axis=-1)),
                torch.Tensor(np.concatenate([neg_points, neg_sdf, neg_parts], axis=-1)),
            ]
        else:
            data_tensor = torch.FloatTensor(
                np.concatenate([points, sdf_vals, parts], axis=-1)
            )
        joint_state = torch.FloatTensor([list(datapoint.zero_joint_config.values())[0]])
        # Bring everything to original A-SDF scale
        joint_state = joint_state / math.pi * 180

        if self.return_joint_type:
            joint_type = datapoint.joint_def[
                list(datapoint.zero_joint_config.keys())[0]
            ]["type"]
            joint_type_one_hot = CARTO.Decoder.utils.encode_joint_types(joint_type)[0]

        idx = self.object_index[datapoint.object_id]

        if self.articulation:
            return_tuple = (data_tensor, joint_state, idx)
            if self.return_joint_type:
                return_tuple += (joint_type_one_hot,)
            return return_tuple, index
        else:
            return data_tensor, index
