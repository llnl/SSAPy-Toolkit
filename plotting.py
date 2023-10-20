######################################################################
# COLLECTION OF ALL PLOTTING AND MEDIA
######################################################################
import numpy as np
from ssapy.constants import RGEO, WGS84_EARTH_RADIUS, WGS84_EARTH_MU
from ssapy.utils import Time, moonPos
from cislunar_utilities import calculate_orbital_elements, mag, astTime
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mpl_toolkits.mplot3d import Axes3D
from PyPDF2 import PdfMerger
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image as PILImage
import io

from IPython.display import Image as IPythonImage
import imageio
import cv2

plt.rcParams.update({'font.size': 7, 'figure.facecolor':'w'})
######################################################################
# COLORS
######################################################################
def darken(color, amount=0.5):
    import colorsys
    try:
        c = matplotlib.colors.cnames[color]
    except:
        c = color
    colors = []
    for i in amount:
        c = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(c))
        colors.append(colorsys.hls_to_rgb(c[0], 1 - i * (1 - c[1]), c[2]))
    return colors
def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b

def generate_rainbow_colors(num_iterations):
    cmap = plt.get_cmap('rainbow')
    colors = [matplotlib.colors.rgb2hex(cmap(i/num_iterations)) for i in range(num_iterations)]
    return colors

######################################################################
# Write video mp4
######################################################################
def write_video(video_name, frames, fps=30):
    print(f'Writing video: {video_name}')
    """
    Writes frames to an mp4 video file
    :param video_name: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """
    img = cv2.imread(frames[0])
    h, w, layers = img.shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(video_name, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(cv2.imread(frame))

    writer.release()
    print(f'Wrote: {video_name}')
    return
######################################################################
# write gif
######################################################################
# build gif
def write_gif(gif_name, frames, fps = 30):
    print(f'Writing gif: {gif_name}')
    with imageio.get_writer(gif_name, mode='I', duration = 1/fps) as writer:
        for i, filename in enumerate(frames):
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f'Wrote {gif_name}')
    return
# def write_gif(gif_name, frames, fps = 30):
#     # read images with imageio
#     images = [imageio.imread(file_path) for file_path in frames]
#     # save image as .mp4 file
#     imagio.mimsave(f'{gif_name}.mp4', images, duration=1/fps)
#     # create .gif from .mp4 using FFmpeg
#     os.system(f'ffmpeg -i {gif_name}.mp4 {gif_name}.gif')
#     # remove the created .mp4 file
#     os.system(f'rm {gif_name}.mp4')
#     return

######################################################################
# Save figures appended to a pdf.
######################################################################
save_plot_to_pdf_call_count = 0
def save_plot_to_pdf(figure, pdf_path):
    global save_plot_to_pdf_call_count
    save_plot_to_pdf_call_count += 1
    if '~' == pdf_path[0]:
        pdf_path = os.path.expanduser(pdf_path)
    if '.' in pdf_path:
        temp_pdf_path = re.sub(r"\.[^.]+$", "_temp.pdf", pdf_path)
    else:
        temp_pdf_path = f"{pdf_path}_temp.pdf"
    # Save the figure as a PNG in-memory using BytesIO
    png_buffer = io.BytesIO()
    figure.savefig(png_buffer, format='png', dpi=300, bbox_inches='tight')
    # Rewind the buffer to the beginning
    png_buffer.seek(0)
    # Open the in-memory PNG using PIL
    png_image = PILImage.open(png_buffer)
    with PdfPages(temp_pdf_path) as pdf:
        # Create a new figure and axis to display the image
        img_fig, img_ax = plt.subplots()
        img_ax.imshow(png_image)
        img_ax.axis('off')
        # Save the figure with the image into the PDF
        pdf.savefig(img_fig, dpi=300, bbox_inches='tight')
    if os.path.exists(pdf_path):
        merger = PdfMerger()
        with open(pdf_path, "rb") as main_pdf, open(temp_pdf_path, "rb") as temp_pdf:
            merger.append(main_pdf)
            merger.append(temp_pdf)
            with open(pdf_path, "wb") as merged_pdf:
                merger.write(merged_pdf)
        os.remove(temp_pdf_path)
    else:
        os.rename(temp_pdf_path, pdf_path)
    plt.close(figure); plt.close(img_fig)# Close the figure and new figure created
    print(f"Saved figure {save_plot_to_pdf_call_count} to {pdf_path}")
    return

def saveplot(fig, filename, bbox_inches="tight", pad_inches=0.1, transparent=True, facecolor="w", edgecolor="w", orientation="landscape"):
    fig.savefig(filename, bbox_inches =bbox_inches, pad_inches = pad_inches, transparent = True, facecolor =facecolor, edgecolor =edgecolor, orientation =orientation)
    return
def loadplot(filename):
    return IPythonImage(filename)

######################################################################
# Formatting x axis
######################################################################
def format_xaxis_decimal_year(time_array, ax):
    n = 5  # Number of nearly evenly spaced points to select
    time_span_in_months = (time_array[-1].datetime - time_array[0].datetime).days / 30

    if n >= time_span_in_months:
        # Get evenly spaced points in the time_array
        selected_indices = np.round(np.linspace(0, len(time_array) - 1, n)).astype(int)
        selected_times = time_array[selected_indices]
        selected_month_year_strings = [t.strftime('%d-%b-%Y') for t in selected_times]
    else:
        # Get the first of n nearly evenly spaced months in the time
        step = int(len(time_array) / (n - 1))
        selected_times = time_array[::step]
        selected_month_year_strings = [t.strftime('%b-%Y') for t in selected_times]
    selected_decimal_years = [t.decimalyear for t in selected_times]
    # Set the x-axis tick positions and labels
    ax.set_xticks(selected_decimal_years)
    ax.set_xticklabels(selected_month_year_strings)

    # Optional: Rotate the tick labels for better visibility
    plt.xticks(rotation=0)




######################################################################
# Histograms
######################################################################
def koe_plot(r,v, times=Time("2025-01-01", scale='utc') + np.linspace(0, int(1*365.25), int(365.25*24)), elements=['a','e','i']):
    orbital_elements = calculate_orbital_elements(r,v)
    fig, ax1 = plt.subplots(dpi=200)
    plt.rcParams.update({'font.size': 7, 'figure.facecolor':'w'})
    if 'a' in elements:
        ax1.plot([], [], label='semi-major axis [GEO]', c='C0')
        ax2 = ax1.twinx(); a = [x/RGEO for x in orbital_elements['semi_major_axis']]
        ax2.plot(astTime(times).decimalyear, a, label='semi-major axis [GEO]', c='C0', linestyle='--')
        ax2.yaxis.label.set_color('C0'); ax2.tick_params(axis='y', colors='C0'); ax2.spines['right'].set_color('C0'); ax2.set_ylabel('semi-major axis [GEO]')
        if np.abs(np.max(a) - np.min(a)) < 2:
            ax2.set_ylim((np.min(a)-.5, np.max(a)+.5))
    if 'e' in elements:
        ax1.plot(astTime(times).decimalyear, [x for x in orbital_elements['eccentricity']], label='eccentricity', c='C1')
    if 'i' in elements:
        ax1.plot(astTime(times).decimalyear, [x for x in orbital_elements['inclination']], label='inclination [rad]', c='C2')
    
    ax1.set_xlabel('Year')
    ax1.legend(loc='upper center'); plt.show(block=False)
    return fig, ax1

def koe_2dhist(stable_data, title=f"Initial orbital elements of\n20 year stable cislunar orbits", limits=[1,100], bins=100, logscale=True):
    if logscale or logscale == 'log':
        norm = matplotlib.colors.LogNorm(limits[0], limits[1])
    else:
        norm = matplotlib.colors.Normalize(limits[0], limits[1])
    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(dpi=100, figsize=(10,8), nrows=3, ncols=3)
    st = fig.suptitle(title, fontsize=12); st.set_x(0.46); st.set_y(0.9)
    ax = axes.flat[0]
    ax.hist2d([x/RGEO for x in stable_data.a], [x for x in stable_data.e], bins=bins, norm=norm); 
    ax.set_xlabel(""); ax.set_ylabel("eccentricity"); ax.set_xticks(np.arange(1,20,2)); ax.set_yticks(np.arange(0,1,.2)); ax.set_xlim((1,18))
    axes.flat[1].set_axis_off()
    axes.flat[2].set_axis_off()

    ax = axes.flat[3]
    ax.hist2d([x/RGEO for x in stable_data.a], [np.degrees(x) for x in stable_data.i], bins=bins, norm=norm); 
    ax.set_xlabel(""); ax.set_ylabel("inclination [deg]"); ax.set_xticks(np.arange(1,20,2)); ax.set_yticks(np.arange(0,91,15)); ax.set_xlim((1,18))
    ax = axes.flat[4]
    ax.hist2d([x for x in stable_data.e], [np.degrees(x) for x in stable_data.i], bins=bins, norm=norm); 
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_xticks(np.arange(0,1,.2)); ax.set_yticks(np.arange(0,91,15))
    axes.flat[5].set_axis_off()

    ax = axes.flat[6]
    ax.hist2d([x/RGEO for x in stable_data.a], [np.degrees(x) for x in stable_data.trueAnomaly], bins=bins, norm=norm); 
    ax.set_xlabel("semi-major axis [GEO]"); ax.set_ylabel("True Anomaly [deg]"); ax.set_xticks(np.arange(1,20,2)); ax.set_yticks(np.arange(0,361,60)); ax.set_xlim((1,18))
    ax = axes.flat[7]
    ax.hist2d([x for x in stable_data.e], [np.degrees(x) for x in stable_data.trueAnomaly], bins=bins, norm=norm); 
    ax.set_xlabel("eccentricity"); ax.set_ylabel(""); ax.set_xticks(np.arange(0,1,.2)); ax.set_yticks(np.arange(0,361,60))
    ax = axes.flat[8]
    ax.hist2d([np.degrees(x) for x in stable_data.i], [np.degrees(x) for x in stable_data.trueAnomaly], bins=bins, norm=norm); 
    ax.set_xlabel("inclination [deg]"); ax.set_ylabel(""); ax.set_xticks(np.arange(0,91,15)); ax.set_yticks(np.arange(0,361,60))

    im = fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax, norm=norm)
    return

######################################################################
# SCATTER PLOTS
######################################################################      
#Color gradient scatter plots
def scatter2d(x,y, cs, xlabel='x', ylabel='y', title='', cbar_label='', dotsize=1, colorsMap='jet', colorscale='linear', colormin=False, colormax = False):
    fig = plt.figure()
    if colormax == False:
        colormax = np.max(cs)
    if colormin == False:
        colormin = np.min(cs)
    cm = plt.get_cmap(colorsMap)
    if colorscale == 'linear':
        cNorm = matplotlib.colors.Normalize(vmin=colormin, vmax=colormax)
    elif colorscale == 'log':
        cNorm = matplotlib.colors.LogNorm(vmin=colormin, vmax=colormax)
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
    plt.scatter(x, y, c=scalarMap.to_rgba(cs), s=dotsize); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap, shrink=.5, label=f'{cbar_label}', pad=0.04)
    plt.tight_layout(); plt.show(block=False)
    return
    
def scatter3d(x,y=None,z=None, cs=None, xlabel='x', ylabel='y', zlabel = 'z', cbar_label='', dotsize=1, colorsMap='jet', title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if x.ndim > 1:
        r = x
        x = r[:,0]; y = r[:,1]; z = r[:,2]
        print(x, y, z)
    if cs is None:
        ax.scatter(x, y, z, s=dotsize)
    else:
        cm = plt.get_cmap(colorsMap)
        cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
        ax.scatter(x, y, z, c=scalarMap.to_rgba(cs), s=dotsize)
        scalarMap.set_array(cs)
        fig.colorbar(scalarMap, shrink=.5, label=f'{cbar_label}', pad=0.075)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel); plt.title(title); plt.tight_layout(); plt.show(block=False)
    return


def dotcolors_scaled(num_colors):
    return cm.rainbow(np.linspace(0, 1, num_colors))
lunar_semi_major = 384399000 #m

#Make a plot of multiple cislunar orbit in GCRF frame.
def orbit_divergence_plot(rs, r_moon=[], times = False, limits = False, title=''):
    if limits == False:
        limits = np.nanmax(mag(rs, axis=1)/RGEO)*1.2; print(f'limits: {limits}')
    if np.size(r_moon) < 1:
        r_moon = moonPos(times)
    else:    
        # print('Lunar position(s) provided.')
        if r_moon.ndim != 2:
            raise IndexError(f"input moon data shape: {np.shape(r_moon)}, input should be 2 dimensions.")
            return None
        if np.shape(r_moon)[1] == 3:
            r_moon = r_moon.T
            # print(f"Tranposed input to {np.shape(r_moon)}")
    plt.rcParams.update({'font.size': 7, 'figure.facecolor':'w'})
    fig = plt.figure(dpi=100, figsize=(15,4))
    for i in range(rs.shape[-1]):
        r = rs[:,:,i]
        x = r[:,0]/RGEO; y = r[:,1]/RGEO; z = r[:,2]/RGEO; xm = r_moon[0]/RGEO; ym = r_moon[1]/RGEO; zm = r_moon[2]/RGEO;
        dotcolors = cm.rainbow(np.linspace(0, 1, len(x))); #print(f'length of x {len(x)}, colors: {np.shape(dotcolors)}')
        
        # Creating plot
        plt.subplot(1,3,1)
        plt.scatter(x, y, color = dotcolors, s=1); plt.scatter(0, 0, color = "blue", s=50); plt.scatter(xm,ym, color = "grey", s=5)
        plt.axis('scaled'); plt.xlabel('x [GEO]'); plt.ylabel('y [GEO]'); plt.xlim((-limits,limits)); plt.ylim((-limits,limits)); plt.text(x[0],y[0],'$\leftarrow$ start'); plt.text(x[-1],y[-1],'$\leftarrow$ end')
        
        plt.subplot(1,3,2)
        plt.scatter(x, z, color = dotcolors, s=1); plt.scatter(0, 0, color = "blue", s=50); plt.scatter(xm,zm, color = "grey", s=5)
        plt.axis('scaled'); plt.xlabel('x [GEO]'); plt.ylabel('z [GEO]'); plt.xlim((-limits,limits)); plt.ylim((-limits,limits)); plt.text(x[0],z[0],'$\leftarrow$ start'); plt.text(x[-1],z[-1],'$\leftarrow$ end'); plt.title(f'{title}')

        plt.subplot(1,3,3)
        plt.scatter(y, z, color = dotcolors, s=1); plt.scatter(0, 0, color = "blue", s=50); plt.scatter(ym,zm, color = "grey", s=5)
        plt.axis('scaled'); plt.xlabel('y [GEO]'); plt.ylabel('z [GEO]'); plt.xlim((-limits,limits)); plt.ylim((-limits,limits)); plt.text(y[0],z[0],'$\leftarrow$ start'); plt.text(y[-1],z[-1],'$\leftarrow$ end')
    plt.tight_layout()
    plt.show(block=False)
#Make a plot of a cislunar orbit in GCRF frame.
def gcrf_plot(r, r_moon=[], times = False, limits = False, title=''):
    if limits == False:
        limits = np.nanmax(mag(r)/RGEO)*1.2; print(f'limits: {limits}')
    if r.ndim != 2:
        raise IndexError(f"input data shape: {np.shape(r)}, input should be 3 dimensions.")
        return None
    if np.shape(r)[1] == 3:
        r = r.T
        # print(f"Tranposed input to {np.shape(r)}")
    if np.size(r_moon) < 1:
        r_moon = moonPos(times)
    else:    
        # print('Lunar position(s) provided.')
        if r_moon.ndim != 2:
            raise IndexError(f"input moon data shape: {np.shape(r_moon)}, input should be 2 dimensions.")
            return None
        if np.shape(r_moon)[1] == 3:
            r_moon = r_moon.T
            # print(f"Tranposed input to {np.shape(r_moon)}")
    x = r[0]/RGEO; y = r[1]/RGEO; z = r[2]/RGEO; xm = r_moon[0]/RGEO; ym = r_moon[1]/RGEO; zm = r_moon[2]/RGEO;
    dotcolors = cm.rainbow(np.linspace(0, 1, len(x))); print(f'length of x {len(x)}, colors: {np.shape(dotcolors)}')
    plt.rcParams.update({'font.size': 10, 'figure.facecolor':'w'})
    fig = plt.figure(dpi=100, figsize=(10,10))
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    # Creating plot
    ax.scatter3D(x, y, z, color = dotcolors, s=1); ax.scatter3D(0, 0, 0, color = "blue", s=50); ax.scatter3D(xm,ym,zm, color = "grey", s=5)
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))  # aspect ratio is 1:1:1 in data space
    ax.set_xlabel('x [GEO]'); ax.set_ylabel('y [GEO]'); ax.set_zlabel('z [GEO]')

    # Creating plot
    plt.subplot(2,2,1)
    plt.scatter(x, y, color = dotcolors, s=1); plt.scatter(0, 0, color = "blue", s=50); plt.scatter(xm,ym, color = "grey", s=5)
    plt.axis('scaled'); plt.xlabel('x [GEO]'); plt.ylabel('y [GEO]'); plt.xlim((-limits,limits)); plt.ylim((-limits,limits)); plt.text(x[0],y[0],'$\leftarrow$ start'); plt.text(x[-1],y[-1],'$\leftarrow$ end'); plt.title(f'{title}')
    
    plt.subplot(2,2,2)
    plt.scatter(x, z, color = dotcolors, s=1); plt.scatter(0, 0, color = "blue", s=50); plt.scatter(xm,zm, color = "grey", s=5)
    plt.axis('scaled'); plt.xlabel('x [GEO]'); plt.ylabel('z [GEO]'); plt.xlim((-limits,limits)); plt.ylim((-limits,limits)); plt.text(x[0],z[0],'$\leftarrow$ start'); plt.text(x[-1],z[-1],'$\leftarrow$ end')

    plt.subplot(2,2,3)
    plt.scatter(y, z, color = dotcolors, s=1); plt.scatter(0, 0, color = "blue", s=50); plt.scatter(ym,zm, color = "grey", s=5)
    plt.axis('scaled'); plt.xlabel('y [GEO]'); plt.ylabel('z [GEO]'); plt.xlim((-limits,limits)); plt.ylim((-limits,limits)); plt.text(y[0],z[0],'$\leftarrow$ start'); plt.text(y[-1],z[-1],'$\leftarrow$ end')
    plt.tight_layout()
    plt.show(block=False)
    return fig, ax


print(f'Imported cislunar_plots.py')