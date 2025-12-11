def write_video(
    video_name,
    frames,
    fps: int = 30,
    *,
    sort_frames: bool = True,
    warn_on_ambiguous: bool = True,
    uniform_size: bool = True,
    target_size: tuple = None,         # (width, height); if None, use first frame
    freeze_last_seconds: float = 0.0,  # extra time to hold last frame
):
    """
    Write frames to an MP4 video file, with optional natural sorting and
    size normalization similar to write_gif.

    Parameters
    ----------
    video_name : str
        Output path ending with .mp4
    frames : iterable of str
        Paths to image files (readable by OpenCV).
    fps : int
        Frame rate.
    sort_frames : bool
        If True, apply natural sort to the input paths.
    warn_on_ambiguous : bool
        Emit warnings when sorting is ambiguous.
    uniform_size : bool
        If True, resize all frames to a common size.
    target_size : (W, H) or None
        If None and uniform_size is True, use the first frame's size.
    freeze_last_seconds : float
        Extra number of seconds to “freeze” the last frame by repeating it.
    """
    import cv2

    paths = list(frames)
    if not paths:
        raise ValueError("frames list is empty")

    if sort_frames:
        paths = _sort_frames(paths, warn_on_ambiguous=warn_on_ambiguous)

    print(f"Writing video: {video_name}")

    # Read first frame for size
    first = cv2.imread(paths[0])
    if first is None:
        raise ValueError(f"Could not read first frame: {paths[0]}")

    if uniform_size:
        if target_size is None:
            h, w = first.shape[:2]
            target_size = (w, h)
        W, H = target_size
        # Make sure first frame matches target_size
        if (first.shape[1], first.shape[0]) != (W, H):
            first = cv2.resize(first, (W, H))
    else:
        h, w = first.shape[:2]
        target_size = (w, h)
        W, H = target_size

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(video_name, fourcc, fps, (W, H))

    last_frame = None

    # Write first frame
    writer.write(first)
    last_frame = first

    # Write the rest
    for p in paths[1:]:
        img = cv2.imread(p)
        if img is None:
            warnings.warn(f"Could not read frame: {p}; skipping.")
            continue

        if uniform_size and (img.shape[1], img.shape[0]) != (W, H):
            img = cv2.resize(img, (W, H))

        writer.write(img)
        last_frame = img

    # Freeze on last frame
    if freeze_last_seconds > 0 and last_frame is not None:
        extra_frames = int(np.round(freeze_last_seconds * float(fps)))
        for _ in range(extra_frames):
            writer.write(last_frame)

    writer.release()
    print(f"Wrote: {video_name}")
