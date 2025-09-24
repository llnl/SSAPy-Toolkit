def bbox_min(lons, lats):
    """
    Compute the minimal bounding box for points given by lats/lons.
    Returns:
      (lat_min, lat_max, lon_left, lon_right, lon_span_deg)

    Notes:
      - lon_left/lon_right are representatives in [-180, 180).
      - If lon_left > lon_right, the bbox crosses the antimeridian.
      - lon_span_deg is the minimal circular span covering all longitudes.
    """
    if lons is None or lats is None:
        raise ValueError("lons and lats must be provided")
    lons = list(lons)
    lats = list(lats)
    if len(lons) != len(lats) or len(lons) == 0:
        raise ValueError("lons and lats must be same nonzero length")

    # Latitude range is simple (no wrap)
    lat_min = min(lats)
    lat_max = max(lats)

    # Normalize longitudes to [0, 360)
    a = [((x % 360) + 360) % 360 for x in lons]
    a.sort()

    # Single-point case
    if len(a) == 1:
        n = ((a[0] + 180.0) % 360.0) - 180.0
        return lat_min, lat_max, n, n, 0.0

    # Find largest gap on the circle
    gaps = [ (a[(i+1) % len(a)] - a[i]) % 360.0 for i in range(len(a)) ]
    k = max(range(len(gaps)), key=lambda i: gaps[i])
    largest_gap = gaps[k]

    # Minimal arc is the complement of the largest gap
    span = 360.0 - largest_gap
    left = a[(k + 1) % len(a)]
    right = left + span

    # Map representatives to [-180, 180)
    norm180 = lambda x: ((x + 180.0) % 360.0) - 180.0
    lon_left = norm180(left)
    lon_right = norm180(right)

    return lat_min, lat_max, lon_left, lon_right, span
