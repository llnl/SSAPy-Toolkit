import os


def figpath(filename):
    """
    Author:
    -------
    Travis Yeager (yeager7@llnl.gov)
    """
    # Remove any extension if present
    filename = os.path.splitext(filename)[0]

    directories = [
        "/p/lustre1/yeager7/cislunar/figures/",
        "/g/g16/yeager7/workdir/Figures/",
        "/home/yeager7/Figures/"
    ]

    for fig_path in directories:
        try:
            os.makedirs(fig_path, exist_ok=True)
            base = os.path.join(fig_path, filename)
            jpg_path = base + ".jpg"

            # Remove files with same base name and different extension
            for f in os.listdir(fig_path):
                f_path = os.path.join(fig_path, f)
                if os.path.isfile(f_path) and f.startswith(filename + ".") and f != filename + ".jpg":
                    os.remove(f_path)

            return jpg_path  # Always return path to write .jpg (overwrite if it exists)
        except (OSError, PermissionError):
            continue

    raise Exception("Could not create or access any of the specified directories")
