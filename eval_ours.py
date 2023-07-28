#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import logging
import json
import numpy as np
import os
import trimesh
import csv
import tqdm

import asdf
import asdf.workspace as ws
import glob
import re
import pathlib
from simnet import shape_pretraining_articulated as simnet
from simnet.shape_pretraining_articulated import utils
from simnet.lib.datasets import PartNetMobilityV0DB


def compute_chamfer_distance(chamfer_dist_file):
    chamfer_distance = []
    with open(chamfer_dist_file, newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=" ", quotechar="|")
        for idx, row in enumerate(spamreader):
            if idx > 0:
                chamfer_distance.append(float(row[-1]))
    print("avg chamfer distance: ", 1000 * np.mean(np.array(chamfer_distance)))


def evaluate(
    experiment_directory, checkpoint, data_dir, mode, specs, joint_type_selection="all"
):
    test_split_file = specs["TestSplit"]

    # sdf_dataset = asdf.data.OurSDFSamples(
    #             specs["DataSource"],
    #             test_split_file,
    #             1e9,
    #             specs["BasePathOurs"],
    #             file_id_selection="test",
    #             load_ram=True,
    #             # return_pos_neg_split=True
    #         )
    base_path = pathlib.Path(specs["BasePathOurs"])
    print(base_path)

    gen_cfg: simnet.config.GenerationConfig = simnet.utils.load_cfg(
        base_path / specs["DataSource"], cfg_class=simnet.config.GenerationConfig
    )
    max_extents = gen_cfg.max_extent
    assert max_extents > 0.0, "max_extents incorrect"
    rescaler = simnet.data.dataset.Rescaler3D(scale=max_extents)

    split_dicts = simnet.data.dataset.get_dataset_split_dict(
        base_path / specs["DataSource"], test_split_file
    )
    file_ids = [base_path / file_id for file_id in split_dicts["test"]]
    test_dataset = simnet.data.dataset.SimpleDataset(file_ids, rescaler=rescaler)

    chamfer_results = utils.AccumulatorDict()
    joint_errors = utils.AccumulatorDict()
    joint_type_errors = utils.AccumulatorDict()

    dataset = "ours"
    class_name = specs["Class"]

    print(mode)
    if mode == "recon_testset":
        all_names = sorted(
            glob.glob(
                os.path.join(
                    experiment_directory,
                    "Results_recon_testset",
                    checkpoint,
                    "Meshes",
                    dataset,
                    "*.ply",
                )
            )
        )
    elif mode == "recon_testset_ttt":
        all_names = sorted(
            glob.glob(
                os.path.join(
                    experiment_directory,
                    "Results_recon_testset_ttt",
                    checkpoint,
                    "Meshes",
                    dataset,
                    "*.ply",
                )
            )
        )
    elif mode == "inter_testset":
        all_names = sorted(
            glob.glob(
                os.path.join(
                    experiment_directory,
                    "Results_inter_testset",
                    checkpoint,
                    "Meshes",
                    dataset,
                    "*.ply",
                )
            )
        )
    elif mode == "inter_testset_ttt":
        all_names = sorted(
            glob.glob(
                os.path.join(
                    experiment_directory,
                    "Results_inter_testset_ttt",
                    checkpoint,
                    "Meshes",
                    dataset,
                    "*.ply",
                )
            )
        )
    elif mode == "generation":
        all_names = sorted(
            glob.glob(
                os.path.join(
                    experiment_directory,
                    "Results_generation",
                    checkpoint,
                    "Meshes",
                    dataset,
                    "*.ply",
                )
            )
        )
    elif mode == "generation_ttt":
        all_names = sorted(
            glob.glob(
                os.path.join(
                    experiment_directory,
                    "Results_generation_ttt",
                    checkpoint,
                    "Meshes",
                    dataset,
                    "*.ply",
                )
            )
        )
    else:
        print(f"Unknown mode {mode = }")

    logging.info("Num of files to be evaluated: {}".format(len(all_names)))

    # for instance_name in all_names:
    for ii, datapoint in tqdm.tqdm(enumerate(test_dataset)):
        instance_name = test_dataset.file_ids[ii].stem.split(".")[0]

        object_meta = PartNetMobilityV0DB.get_object_meta(
            datapoint.object_id
        )  # Actually ID
        category = object_meta["model_cat"]
        # print(instance_name)

        reconstructed_name = re.split("/", instance_name)[-1]

        logging.debug(
            "evaluating " + os.path.join(dataset, class_name, reconstructed_name)
        )

        # Get Joint Type, assuming single joint
        joint_type = datapoint.joint_def[list(datapoint.zero_joint_config.keys())[0]][
            "type"
        ]

        if mode == "recon_testset":
            reconstructed_mesh_filename = ws.get_recon_testset_mesh_filename(
                experiment_directory, checkpoint, dataset, class_name, instance_name
            )
            joint_error_filename = ws.get_recon_testset_error_file_name(
                experiment_directory, checkpoint, dataset, class_name, instance_name
            )
            if specs["JointTypeInput"]:
                joint_type_error_filename = (
                    ws.get_recon_testset_joint_type_error_file_name(
                        experiment_directory,
                        checkpoint,
                        dataset,
                        class_name,
                        instance_name,
                    )
                )
        elif mode == "recon_testset_ttt":
            reconstructed_mesh_filename = ws.get_recon_testset_ttt_mesh_filename(
                experiment_directory, checkpoint, dataset, class_name, instance_name
            )
            joint_error_filename = ws.get_recon_testset_ttt_error_file_name(
                experiment_directory, checkpoint, dataset, class_name, instance_name
            )
        elif mode == "inter_testset":
            reconstructed_mesh_filename = ws.get_inter_testset_mesh_filename(
                experiment_directory,
                checkpoint,
                dataset,
                class_name,
                reconstructed_name[:-4],
            )
        elif mode == "generation":
            reconstructed_mesh_filename = ws.get_generation_mesh_filename(
                experiment_directory,
                checkpoint,
                dataset,
                class_name,
                reconstructed_name[:-4],
            )
        elif mode == "generation_ttt":
            reconstructed_mesh_filename = ws.get_generation_ttt_mesh_filename(
                experiment_directory,
                checkpoint,
                dataset,
                class_name,
                reconstructed_name[:-4],
            )

        logging.debug('reconstructed mesh is "' + reconstructed_mesh_filename + '"')

        ground_truth_points = trimesh.PointCloud(datapoint.full_pc)

        reconstruction = trimesh.load(reconstructed_mesh_filename)

        chamfer_dist = asdf.metrics.chamfer.compute_pytorch3d_chamfer(
            ground_truth_points,
            reconstruction,
            0.0,  # normalization_params["offset"],
            1.0,  # normalization_params["scale"],
            num_mesh_samples=ground_truth_points.shape[0],
        )

        logging.debug("chamfer distance: " + str(chamfer_dist))

        chamfer_results.increment(category, [chamfer_dist])

        try:
            with open(joint_type_error_filename, "r") as f:
                joint_type_error = np.loadtxt(f)  # True when there is an error
            joint_type_errors.increment(category, [joint_type_error])
        except:
            joint_type_error = (
                0.0  # We just assume there was no error --> doesn't break normal evals
            )

        try:
            with open(joint_error_filename, "r") as f:
                joint_error = np.loadtxt(f)
            if (
                joint_type_selection == "all" or joint_type == joint_type_selection
            ) and joint_type_error == 0.0:  # No error!
                joint_errors.increment(category, [joint_error])
        except:
            logging.warn(f"Couldn't load {joint_error_filename}")

    chamfer_dist_file = os.path.join(
        ws.get_evaluation_dir(
            experiment_directory + "/Evaluation_with_Latent/", checkpoint, True
        ),
        "chamfer.csv",
    )

    # with open(os.path.join(chamfer_dist_file),"w") as f:
    #     f.write("shape, chamfer_dist\n")
    #     for result in chamfer_results:
    #         f.write("{}, {}\n".format(result[0], result[1]))

    print("**** Chamfer Distance ****")
    for category, cat_chamfer_distance in chamfer_results.items():
        print(
            f"{category} {np.array(cat_chamfer_distance).mean() * 1000} with {len(cat_chamfer_distance)}"
        )
    print(
        f"Instance mean {np.concatenate([np.array(cat_results) for cat_results in chamfer_results.values()]).mean() * 1000}"
    )
    print(
        f"Category mean {np.array([np.array(cat_results).mean() for cat_results in chamfer_results.values()]).mean() * 1000}"
    )

    print("**** Joint Errors ****")
    for category, cat_joint_errors in joint_errors.items():
        print(
            f"{category} Joint Errors Mean {np.array(cat_joint_errors).mean()} with {len(cat_joint_errors)}"
        )
    instance_mean = np.concatenate(
        [
            np.array(cat_results)
            for cat, cat_results in joint_errors.items()
            if cat != "Table"
        ]
    ).mean()
    print(f"Instance mean {instance_mean}")
    category_mean = np.array(
        [
            np.array(cat_results).mean()
            for cat, cat_results in joint_errors.items()
            if cat != "Table"
        ]
    ).mean()
    print(f"Category mean {category_mean}")

    print("**** Joint Accuracy ****")
    for category, cat_joint_type_errors in joint_type_errors.items():
        print(
            f"{category} Joint Type Accuracy {1 - np.array(cat_joint_type_errors).mean()} with {len(cat_joint_type_errors)}"
        )
    print(
        f"Instance mean {np.concatenate([1- np.array(cat_results) for cat_results in joint_type_errors.values()]).mean()}"
    )
    print(
        f"Category mean {np.array([1- np.array(cat_results).mean() for cat_results in joint_type_errors.values()]).mean()}"
    )

    return chamfer_dist_file


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Evaluate a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include experiment specifications in "
        + '"specs.json", and logging will be done in this directory as well.',
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint to test.",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        default="data",
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--mode",
        "-m",
        required=True,
        help="choose from recon_testset | inter_testset | genration",
    )

    arg_parser.add_argument(
        "--joint-type",
        "-jt",
        default="all",
        help="choose from prismatic | revolute | all",
    )

    asdf.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    asdf.configure_logging(args)

    specs_filename = os.path.join(args.experiment_directory, "specs.json")
    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )
    specs = json.load(open(specs_filename))

    chamfer_dist_file = evaluate(
        args.experiment_directory,
        args.checkpoint,
        args.data_source,
        args.mode,
        specs,
        joint_type_selection=args.joint_type,
    )

    compute_chamfer_distance(chamfer_dist_file)
