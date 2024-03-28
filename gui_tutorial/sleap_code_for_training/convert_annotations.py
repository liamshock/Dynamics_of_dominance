import h5py
import argparse
import numpy as np
import csv
import os
import shutil
import glob


def make_crop(center, crop_size):
    corner = (int(center[0] - crop_size / 2), int(center[1] - crop_size / 2))
    crop_bb = np.s_[corner[0] : corner[0] + crop_size, corner[1] : corner[1] + crop_size]
    return crop_bb


def extract_annotations(annotations_file: str, crop_size: int):

    results = {}

    with h5py.File(annotations_file, "r") as f:
        frames_names = sorted([x for x in f if x.startswith("frames")])
        labels_names = sorted([x for x in f if x.startswith("labels")])

        for frames_name, labels_name in zip(frames_names, labels_names):

            results_frames = []
            results_labels = []

            frames = f[frames_name][:]
            labels = f[labels_name][:]

            labels = np.flip(labels, axis=-1)  # swap XY for numpy indexing

            for frame_index in range(len(frames)):
                cur_frame = frames[frame_index]
                cur_labels = labels[frame_index]

                for fish_index in range(len(cur_labels)):
                    cur_fish_labels = cur_labels[fish_index]
                    average_position = np.mean(cur_fish_labels, axis=0)

                    crop_bb = make_crop(average_position, crop_size)
                    crop_frame = cur_frame[crop_bb]
                    crop_frame = crop_frame[np.newaxis, :, :]  # add channel axis

                    cur_labels_cropped = cur_labels - (crop_bb[0].start, crop_bb[1].start)

                    crop_index = len(results_frames)
                    crop_results = []

                    for i, label_cropped in enumerate(cur_labels_cropped):
                        for joint_index in range(len(label_cropped)):
                            cur_point = label_cropped[joint_index]
                            in_crop = 0 <= cur_point[0] < crop_size and 0 <= cur_point[1] < crop_size
                            if in_crop:
                                crop_results.append([crop_index, i, joint_index, cur_point[0], cur_point[1], True]) #always visible=True for now

                    results_frames.append(crop_frame)
                    results_labels += crop_results

            results[frames_name] = (results_frames, results_labels)

    return results


def main(annotations_root, out_dir, crop_size):

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    merged_results_frames = {}
    merged_results_labels = {}

    annotations_files = sorted(glob.glob(os.path.join(annotations_root, "*.h5")))
    for annotations_file in annotations_files:
        results = extract_annotations(annotations_file, crop_size)

        for cam_name in results:
            frames, labels = results[cam_name]
            if cam_name not in merged_results_frames:
                merged_results_frames[cam_name] = frames
                merged_results_labels[cam_name] = labels
            else:
                cur_frame_index = int(np.max(merged_results_labels[cam_name], axis=0)[0]) + 1
                # update labels frame index cumulative
                for i in range(len(labels)):
                    labels[i][0] = labels[i][0] + cur_frame_index

                merged_results_frames[cam_name] = merged_results_frames[cam_name] + frames
                merged_results_labels[cam_name] = merged_results_labels[cam_name] + labels

    # save results
    for cam_name in merged_results_frames:

        minimal_name = cam_name[-2:]

        results_labels = merged_results_labels[cam_name]
        results_frames = merged_results_frames[cam_name]

        with open(os.path.join(out_dir, f"labels_{minimal_name}.csv"), "w") as results_csv_f:
            writer = csv.writer(results_csv_f)
            for row in results_labels:
                writer.writerow(row)

        with h5py.File(os.path.join(out_dir, f"frames_{minimal_name}.h5"), "w") as results_frames_f:
            results_frames_f.create_dataset("frames", data=results_frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations_root", type=str)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--out_dir", type=str, default="out")
    args = parser.parse_args()

    main(**vars(args))
