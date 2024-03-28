import argparse

from sleap.io.video import Video, HDF5Video
from sleap.io.dataset import Labels
from sleap.skeleton import Skeleton
from sleap.instance import Instance, Point, LabeledFrame

import os

import pandas as pd
from sleap.nn.training import Trainer
from sleap.nn.config import TrainingJobConfig
from sleap.util import get_package_file


def train(output_name, video_path, dataset_name, labels_path):

    print(output_name, video_path, labels_path, dataset_name)

    video = Video(backend=HDF5Video(filename=video_path, dataset=dataset_name, input_format="channels_first"))

    labels_df = pd.read_csv(labels_path, names=["frame_index", "instance_index", "joint_index", "x", "y", "visible"])

    skeleton = Skeleton()
    skeleton.add_nodes(["head", "neck", "tail"])
    nodes = skeleton.nodes

    edges = [["head", "neck"], ["neck", "tail"]]
    for node1, node2 in edges:
        skeleton.add_edge(node1, node2)

    labeled_frames = []

    for frame_index, cur_frame in labels_df.groupby("frame_index"):
        instances = []
        for instance_index, cur_fish in cur_frame.groupby("instance_index"):

            points = {}

            for joint_index, cur_joint in cur_fish.groupby("joint_index"):
                x = cur_joint["x"].to_numpy()[0]
                y = cur_joint["y"].to_numpy()[0]
                visible = cur_joint["visible"].to_numpy()[0]
                points[nodes[joint_index]] = Point(x, y, visible=visible, complete=True)

            inst = Instance(skeleton=skeleton, points=points)
            instances.append(inst)

        labeled_frames.append(LabeledFrame(video, frame_idx=frame_index, instances=instances))

    labels = Labels(labeled_frames=labeled_frames)

    job_filename = "baseline.bottomup.json"
    profile_dir = get_package_file("sleap/training_profiles")
    if os.path.exists(os.path.join(profile_dir, job_filename)):
        job_filename = os.path.join(profile_dir, job_filename)
    job_config = TrainingJobConfig.load_json(job_filename)

    job_config.model.heads.multi_instance.confmaps.sigma = 2.5
    job_config.model.heads.multi_instance.confmaps.output_stride = 2
    job_config.model.heads.multi_instance.pafs.output_stride = 4
    job_config.model.backbone.unet.max_stride = 16
    job_config.model.backbone.unet.output_stride = 2
    job_config.model.backbone.unet.filters = 16
    job_config.model.backbone.unet.filters_rate = 2.
    job_config.outputs.run_name = output_name

    trainer = Trainer.from_config(job_config, training_labels=labels, validation_labels=0.1)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_name", type=str)
    parser.add_argument("video_path", type=str)
    parser.add_argument("labels_path", type=str)
    parser.add_argument("--dataset_name", type=str, default="frames")
    args = parser.parse_args()

    train(**vars(args))
