import argparse
import pandas as pd
import h5py
import cv2

def check(video_path, labels_path, dataset_name):
    print(video_path, labels_path)

    with h5py.File(video_path, "r") as f:
        frames = f[dataset_name][:,0,:,:]

    labels_df = pd.read_csv(labels_path, names=["frame_index", "instance_index", "joint_index", "x", "y", "visible"])

    for frame_index, cur_label in labels_df.groupby("frame_index"):
        cur_frame = frames[frame_index]
        for x,y in zip(cur_label["x"], cur_label["y"]):
            cv2.circle(cur_frame, (int(y), int(x)), 2, 255, -1)

        print(frame_index)
        cv2.imshow("", cur_frame)
        cv2.waitKey()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    parser.add_argument("labels_path", type=str)
    parser.add_argument("--dataset_name", type=str, default="frames")
    args = parser.parse_args()

    check(**vars(args))
