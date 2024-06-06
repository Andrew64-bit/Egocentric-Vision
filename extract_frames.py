import subprocess
import os
from utils.args import args

def video_to_frames(
    video_path: str,
    frames_folder: str
):
    """
    Cut a video in frames.
    """

    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)

    command = f"ffmpeg -i {video_path} -vf \"scale=456:256\" -q:v 2 {frames_folder}/frame_%010d.jpg"
    subprocess.run(command, shell = True)


if __name__ == '__main__':
    video_to_frames(args.video_path, args.frames_folder)