import matpl

def globe_plot(r: np.ndarray, t: np.ndarray, limits: Optional[float] = False, title: str = '',
               figsize: Tuple[int, int] = (7, 8), save_path: Optional[str] = False, 
               el: int = 30, az: int = 0, scale: float = 1) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate a 3D globe plot showing the position of points in Earth-centered 
    coordinates. Optionally save the plot to a file.

    Author:
    -------
    Travis Yeager (yeager7@llnl.gov)
    """
    # Scale the coordinates by RGEO
    r = r / RGEO

    # Set limits if not provided
    if limits is False:
        limits = np.nanmax(np.abs([r[:, 0], r[:, 1], r[:, 2]])) * 1.2

    # Load and scale Earth image
    earth_png = PILImage.open(find_file("earth", ext=".png"))
    earth_png = earth_png.resize((5400 // scale, 2700 // scale))
    bm = np.array(earth_png.resize([int(d) for d in earth_png.size])) / 256.

    # Generate mesh for globe surface
    lons = np.linspace(-180, 180, bm.shape[1]) * np.pi / 180
    lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi / 180
    mesh_x = np.outer(np.cos(lons), np.cos(lats)).T * EARTH_RADIUS / RGEO
    mesh_y = np.outer(np.sin(lons), np.cos(lats)).T * EARTH_RADIUS / RGEO
    mesh_z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T * EARTH_RADIUS / RGEO

    # Set color for the scatter plot
    dotcolors = plt.cm.rainbow(np.linspace(0, 1, len(r[:, 0])))

    # Create the figure and 3D axis
    fig = plt.figure(dpi=100, figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('black')
    ax.tick_params(axis='both', colors='white')
    ax.grid(True, color='grey', linestyle='--', linewidth=0.5)
    ax.set_facecolor('black')  # Set plot background color to black

    # Plot the satellite positions and the Earth surface
    ax.scatter(r[:, 0], r[:, 1], r[:, 2], color=dotcolors, s=1)
    ax.plot_surface(mesh_x, mesh_y, mesh_z, rstride=4, cstride=4, facecolors=bm, shade=False)

    # Set the view angle and axis limits
    ax.view_init(elev=el, azim=az)
    x_ticks = np.linspace(-limits, limits, 5)
    y_ticks = np.linspace(-limits, limits, 5)
    z_ticks = np.linspace(-limits, limits, 5)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)

    # Set axis labels with white color
    ax.set_xlabel('x [GEO]', color='white')
    ax.set_ylabel('y [GEO]', color='white')
    ax.set_zlabel('z [GEO]', color='white')

    # Set tick label colors to white
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    ax.set_aspect('equal')

    # Apply black background function (assuming `make_black` function exists)
    fig, ax = make_black(fig, ax)

    # Save the plot if save_path is provided
    if save_path:
        save_plot(fig, save_path)

    return fig, ax
