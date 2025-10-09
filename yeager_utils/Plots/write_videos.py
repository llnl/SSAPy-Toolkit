def write_video(video_name, frames, fps=30):
    """
    Write frames to an MP4 video file.

    Parameters
    ----------
    video_name : str
        Output path ending with .mp4
    frames : list of str
        Paths to image files (readable by OpenCV).
    fps : int
        Frame rate.
    """
    import cv2

    print(f'Writing video: {video_name}')

    img = cv2.imread(frames[0])
    h, w, _ = img.shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(video_name, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(cv2.imread(frame))

    writer.release()
    print(f'Wrote: {video_name}')