import shutil
import cv2
from PIL import Image
import os
import lmdb
import pickle

def read_mp4(mp4_path):
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
        print(mp4_path)
    else:
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frames.append(frame)
            else:
                break
    cap.release()
    cv2.destroyAllWindows()
    return frames


def save_video_to_jpg(dir_path, frames):
    for ind, frame in enumerate(frames):
        im = Image.fromarray(frame)
        im.save(f'{dir_path}/{ind}.jpeg')


def save_lmdb(videos_dir, lmdb_path):

    all_subdirs = os.listdir(videos_dir)
    all_subdirs = sorted(all_subdirs)[:-1]
    vids = []
    total_frame = 0
    frames_ind = {}
    videos_ind = {}
    for vid_ind, sub_dir in enumerate(all_subdirs):
        frames = read_mp4(videos_dir+'/'+sub_dir)
        videos_ind[vid_ind] = total_frame
        for i in range(len(frames)):
            frames_ind[total_frame] = vid_ind
            total_frame += 1

        vids.append(frames)

    map_size = total_frame * 640 * 480 + total_frame * 2*4
    env = lmdb.open(lmdb_path, map_size=map_size)
    with env.begin(write=True) as txn:
        key = "videos_ind"
        txn.put(key.encode('utf-8'), pickle.dumps(videos_ind))
        key = "frames_ind"
        txn.put(key.encode('utf-8'), pickle.dumps(frames_ind))

        fi = 0
        for vid_ind, frames in enumerate(vids):
            for frame in frames:
                img = cv2.imencode('.jpg', frame)[1]
                key = f"{fi}"
                img = pickle.dumps(img)
                txn.put(key.encode('utf-8'), img)
                fi += 1
