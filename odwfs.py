# odwfs.py
# Utilities for analyzing ODWFS pupil images.
# Last edited: 2026-08-05

# Import packages
import os
import glob
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus
from scipy import ndimage
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_closing
from scipy.optimize import least_squares
from skimage.measure import find_contours

def inspect_dataset(directory):
    """
    Inspect a dataset directory.

    Checks that an initialization image exists and reports which
    scan series are available.

    Returns
    -------
    init_file : pathlib.Path
        Path to the initialization image.

    available : dict
        Dictionary of available scan series.
        Example:
            {
                "B": {"x": True, "y": True},
                "T": {"x": True, "y": False},
            }
    """

    directory = Path(directory)

    # -------------------------------
    # Initialization image
    # -------------------------------

    init_files = sorted(directory.glob("init*.tif"))

    if len(init_files) == 0:
        raise FileNotFoundError(
            f"No initialization image ('init*.tif') found in {directory}"
        )

    if len(init_files) > 1:
        raise ValueError(
            f"Multiple initialization images found in {directory}"
        )

    init_file = init_files[0]

    # -------------------------------
    # Determine available scan series
    # -------------------------------

    available = {}

    for series in ("B", "T"):

        available[series] = {
            "x": any(directory.glob(f"{series}_series_x*.tif")),
            "y": any(directory.glob(f"{series}_series_y*.tif")),
        }

    print(f"Initialization image: {init_file.name}\n")

    print("Available scan series:")

    for series in ("B", "T"):

        scans = []

        if available[series]["x"]:
            scans.append("x")

        if available[series]["y"]:
            scans.append("y")

        if scans:
            print(f"  {series}: {', '.join(scans)}")
        else:
            print(f"  {series}: none")

    return init_file, available

def decode_voltage(text):
    """
    Convert filename voltage into float.

    Examples
    --------
    Xn1d070 -> -1.070
    Yp0d200 ->  0.200
    """

    sign = -1 if text[1] == "n" else 1

    value = float(
        text[2:].replace("d", ".")
    )

    return sign * value

def fit_circle(x, y):
    """
    Fit a circle to contour coordinates using least squares.

    Returns
    -------
    xc : float
        Circle center x-coordinate.

    yc : float
        Circle center y-coordinate.

    radius : float
        Fitted circle radius.
    """

    def residuals(params):
        xc, yc, radius = params

        distance = np.sqrt(
            (x - xc) ** 2
            + (y - yc) ** 2
        )

        return distance - radius

    x0 = x.mean()
    y0 = y.mean()

    radius0 = np.mean(
        np.sqrt(
            (x - x0) ** 2
            + (y - y0) ** 2
        )
    )

    result = least_squares(
        residuals,
        [x0, y0, radius0],
    )

    return tuple(result.x)


def find_init_pupils(
    init_file,
    threshold=0.08,
    smoothing_sigma=1,
    min_region_pixels=10000,
    flip_vertical=True,
    annulus_inner_offset=1,
    annulus_outer_offset=3,
    plot=True,
):
    """
    Find and fit all pupils in an initialization image.

    Parameters
    ----------
    init_file : str or pathlib.Path
        Path to the initialization TIFF image.

    threshold : float
        Threshold applied after normalizing the image to 0-1.

    smoothing_sigma : float
        Gaussian smoothing width in pixels.

    min_region_pixels : int
        Ignore connected regions smaller than this value.

    flip_vertical : bool
        Flip the image vertically before processing.

    plot : bool
        Display the original image, mask, and fitted circles.

    Returns
    -------
    result : dict
        Dictionary containing:

        image
            Processed image, including the vertical flip.

        mask
            Cleaned binary pupil mask.

        pupils
            List of pupil dictionaries. Each dictionary contains
            center, radius, x, y, and area.
    """

    init_file = Path(init_file)
    filename = init_file.stem

    parts = filename.split("_")
    
    x_voltage = decode_voltage(parts[1])
    y_voltage = decode_voltage(parts[2])
    
    print(f"Initialization image: {init_file.name}")
    print(f"Voltages: X = {x_voltage:.3f} V, Y = {y_voltage:.3f} V")

    if not init_file.exists():
        raise FileNotFoundError(
            f"Initialization image does not exist: {init_file}"
        )

    image = mpimg.imread(init_file)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D image, but received shape {image.shape}"
        )

    if flip_vertical:
        image = np.flipud(image)

    # Normalize to the range 0-1.
    img = image.astype(float)

    img -= img.min()

    image_range = img.max()

    if image_range == 0:
        raise ValueError(
            f"The image contains no intensity variation: {init_file}"
        )

    img /= image_range

    # Smooth before thresholding.
    img_smooth = gaussian_filter(
        img,
        sigma=smoothing_sigma,
    )

    # Create and clean the binary mask.
    mask = img_smooth > threshold

    mask = binary_fill_holes(mask)

    mask = binary_closing(
        mask,
        iterations=2,
    )

    # Find connected regions.
    labels, nlabels = ndimage.label(mask)

    pupils = []

    for label_number in range(1, nlabels + 1):

        region = labels == label_number
        area = int(region.sum())

        if area < min_region_pixels:
            continue

        contours = find_contours(
            region.astype(float),
            level=0.5,
        )

        if not contours:
            continue

        # Use the longest contour, which should be the outer pupil edge.
        contour = max(
            contours,
            key=len,
        )

        y = contour[:, 0]
        x = contour[:, 1]

        xc, yc, radius = fit_circle(x, y)

        pupils.append(
            {
                "center": (xc, yc),
                "x": xc,
                "y": yc,
                "radius": radius,
                "area": area,
                "annulus_inner_offset": annulus_inner_offset,
                "annulus_outer_offset": annulus_outer_offset,
            }
        )

    if not pupils:
        raise RuntimeError(
            "No pupil-sized regions were found. "
            "Try changing threshold or min_region_pixels."
        )

    # Top-to-bottom, then left-to-right.
    pupils.sort(
        key=lambda pupil: (
            -pupil["y"],
            pupil["x"],
        )
    )

    if plot:
        fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )
    
        axes[0].imshow(
            image,
            origin="lower",
            cmap="gray",
        )
        axes[0].set_title("Initialization image")
        axes[0].axis("off")

        axes[1].imshow(
            mask,
            origin="lower",
            cmap="gray",
        )
        axes[1].set_title("Cleaned binary mask")
        axes[1].axis("off")

        axes[2].imshow(
            image,
            origin="lower",
            cmap="gray",
        )

        theta = np.linspace(
            0,
            2*np.pi,
            500,
        )

        for index, pupil in enumerate(pupils, start=1):

            xc, yc = pupil["center"]
            radius = pupil["radius"]

            # Pupil aperture
            aperture = CircularAperture(
                (xc, yc),
                r=radius,
            )

            aperture.plot(
                ax=axes[2],
                color="red",
                lw=1,
            )

            # Annulus (used for drift monitoring)
            annulus = CircularAnnulus(
                (xc, yc),
                r_in=radius + annulus_inner_offset,
                r_out=radius + annulus_outer_offset,
            )

            annulus.plot(
                ax=axes[2],
                color="cyan",
                lw=1,
            )

            axes[2].plot(
                xc,
                yc,
                "+",
                color="yellow",
                markersize=12,
                markeredgewidth=2,
            )

            axes[2].text(
                xc,
                yc + radius + 15,
                str(index),
                color="white",
                ha="center",
                fontsize=10,
            )

        axes[2].set_title(
            f"Pupil apertures (red)\n"
            f"Annulus: r+{annulus_inner_offset} to r+{annulus_outer_offset} px (cyan)"
        )
        axes[2].text(
            0.02,
            0.98,
            f"Annulus:\n"
            f"Inner = r + {annulus_inner_offset}px\n"
            f"Outer = r + {annulus_outer_offset}px",
            transform=axes[2].transAxes,
            va="top",
            color="white",
            bbox=dict(facecolor="black", alpha=0.6),
        )
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    print(f"Found {len(pupils)} pupils:")

    for index, pupil in enumerate(pupils, start=1):
        xc, yc = pupil["center"]

        print(
            f"  Pupil {index}: "
            f"center=({xc:.1f}, {yc:.1f}), "
            f"radius={pupil['radius']:.1f}"
        )

    return {
        "image": image,
        "mask": mask,
        "pupils": pupils,
    }

def pupil_flux(image, pupil):
    """
    Measure the integrated intensity inside a pupil.
    """

    aperture = CircularAperture(
        pupil["center"],
        r=pupil["radius"],
    )

    photometry = aperture_photometry(
        image,
        aperture,
    )

    return float(
        photometry["aperture_sum"][0]
    )

def annulus_flux(image, pupil):
    """
    Measure the integrated intensity in the annulus around a pupil.
    """

    annulus = CircularAnnulus(
        pupil["center"],
        r_in=pupil["radius"] + pupil["annulus_inner_offset"],
        r_out=pupil["radius"] + pupil["annulus_outer_offset"],
    )

    photometry = aperture_photometry(
        image,
        annulus,
    )

    return float(
        photometry["aperture_sum"][0]
    )

def measure_recorded_series(
    directory,
    init,
    flux_function=pupil_flux,
    flip_vertical=True,
):
    """
    Measure pupil flux using scan points recorded in metadata.csv.

    Returns
    -------
    voltage : ndarray
        Scanned-axis voltages, sorted from low to high.

    flux : ndarray
        Integrated intensity with shape
        (number of scan points, number of pupils).

    fixed_voltage : float
        Voltage held constant during the scan.

    series : str
        Either "B" or "T".

    fixed_axis : str
        Either "x" or "y".
    """
    directory = Path(directory)
    metadata_path = directory / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file does not exist: {metadata_path}"
        )

    with metadata_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        records = [
            row
            for row in csv.DictReader(file)
            if row["capture_type"] == "scan_point"
        ]

    if not records:
        raise RuntimeError(
            f"No scan points were recorded in {metadata_path}"
        )

    series_values = {
        row["series"] for row in records
    }
    fixed_axes = {
        row["fixed_axis"] for row in records
    }

    if len(series_values) != 1:
        raise ValueError(
            "Metadata contains more than one series."
        )

    if len(fixed_axes) != 1:
        raise ValueError(
            "Metadata contains more than one fixed axis."
        )

    series = next(iter(series_values))
    fixed_axis = next(iter(fixed_axes))

    pupils = init["pupils"]
    voltage = np.zeros(len(records), dtype=float)
    flux = np.zeros(
        (len(records), len(pupils)),
        dtype=float,
    )

    fixed_values = []

    for image_index, record in enumerate(records):
        x_voltage = float(record["x_voltage_v"])
        y_voltage = float(record["y_voltage_v"])

        if fixed_axis == "x":
            fixed_values.append(x_voltage)
            voltage[image_index] = y_voltage
        else:
            fixed_values.append(y_voltage)
            voltage[image_index] = x_voltage

        image_path = directory / record["filename"]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Recorded image is missing: {image_path}"
            )

        image = plt.imread(image_path)

        if flip_vertical:
            image = np.flipud(image)

        for pupil_index, pupil in enumerate(pupils):
            flux[image_index, pupil_index] = (
                flux_function(image, pupil)
            )

    fixed_values = np.asarray(
        fixed_values,
        dtype=float,
    )

    if not np.allclose(
        fixed_values,
        fixed_values[0],
        atol=1e-9,
    ):
        raise ValueError(
            "The supposedly fixed voltage changed within the scan."
        )

    order = np.argsort(voltage)

    return (
        voltage[order],
        flux[order],
        float(fixed_values[0]),
        series,
        fixed_axis,
    )

def measure_series(
    directory,
    measurement,
    pupils,
    flux_function=pupil_flux,
    flip_vertical=True,
):
    """
    Measure a flux value for every pupil across one image series.

    Parameters
    ----------
    directory : str
        Directory containing the TIFF images.

    measurement : str
        "x" means X is fixed and Y is scanned.
        "y" means Y is fixed and X is scanned.

    pupils : list
        Output from find_init_pupils()["pupils"].

    flux_function : callable
        Function used to measure each pupil. For example:
            pupil_flux
            annulus_flux

    flip_vertical : bool
        Vertically flip each image before measuring.

    Returns
    -------
    voltage : ndarray
        Scanned voltage, sorted from low to high.

    flux : ndarray
        Shape (number of images, number of pupils).

    fixed_voltage : float
        Voltage held constant during the scan.
    """

    pattern = os.path.join(
        directory,
        f"{measurement}_*.tif",
    )

    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matching '{measurement}_*.tif' found in {directory}"
        )

    n_images = len(files)
    n_pupils = len(pupils)

    voltage = np.zeros(n_images, dtype=float)
    flux = np.zeros((n_images, n_pupils), dtype=float)

    fixed_voltage = None

    for i, file in enumerate(files):

        image = plt.imread(file)

        if flip_vertical:
            image = np.flipud(image)

        filename = os.path.splitext(
            os.path.basename(file)
        )[0]

        parts = filename.split("_")

        x_voltage = decode_voltage(parts[1])
        y_voltage = decode_voltage(parts[2])

        if measurement == "x":
            fixed_voltage = x_voltage
            voltage[i] = y_voltage

        elif measurement == "y":
            fixed_voltage = y_voltage
            voltage[i] = x_voltage

        else:
            raise ValueError(
                "measurement must be either 'x' or 'y'"
            )

        for j, pupil in enumerate(pupils):
        
            flux[i, j] = flux_function(
                image,
                pupil,
            )

    order = np.argsort(voltage)

    return (
        voltage[order],
        flux[order],
        fixed_voltage,
    )

def plot_intensity(
    measurement,
    voltage,
    flux,
    fixed_voltage,
    ylabel="Integrated intensity",
    title_prefix="Pupil",
    output_path=None,
    show=True,
):
    """
    Plot the intensity of every pupil as a function of scan voltage.

    Parameters
    ----------
    measurement : str
        "x" or "y".

    voltage : ndarray
        Scanned voltage.

    flux : ndarray
        Shape (n_images, n_pupils).

    fixed_voltage : float
        Voltage held constant during the scan.

    ylabel : str
        Label for the y-axis.

    title_prefix : str
        Prefix for the plot title.
    """

    if measurement == "x":
        xlabel = "Y Voltage (V)"
        title = (
            f"{title_prefix} Intensity\n"
            f"X = {fixed_voltage:.3f} V"
        )

    elif measurement == "y":
        xlabel = "X Voltage (V)"
        title = (
            f"{title_prefix} Intensity\n"
            f"Y = {fixed_voltage:.3f} V"
        )

    else:
        raise ValueError(
            "measurement must be 'x' or 'y'"
        )

    fig = plt.figure(figsize=(7, 5))

    markers = ["o", "s", "^", "d", "v", "*"]

    for i in range(flux.shape[1]):

        plt.plot(
            voltage,
            flux[:, i],
            marker=markers[i % len(markers)],
            linewidth=2,
            label=f"Pupil {i+1}",
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

def directional_annulus_flux(
    image,
    pupil,
):
    """
    Measure intensity in four sectors of the annulus around one pupil.

    The annulus geometry is taken from the pupil dictionary created by
    find_init_pupils().

    Returns
    -------
    result : dict
        Flux in the top, bottom, left, and right annulus sectors,
        plus horizontal and vertical drift signals.
    """

    xc, yc = pupil["center"]

    radius = pupil["radius"]

    r_in = radius + pupil["annulus_inner_offset"]
    r_out = radius + pupil["annulus_outer_offset"]

    yy, xx = np.indices(image.shape)

    dx = xx - xc
    dy = yy - yc

    distance = np.sqrt(
        dx**2 + dy**2
    )

    annulus_mask = (
        (distance >= r_in)
        & (distance < r_out)
    )

    # Divide the annulus into four directional sectors.
    right_mask = annulus_mask & (dx >= np.abs(dy))
    left_mask = annulus_mask & (-dx >= np.abs(dy))
    top_mask = annulus_mask & (dy >= np.abs(dx))
    bottom_mask = annulus_mask & (-dy >= np.abs(dx))

    right_flux = image[right_mask].sum()
    left_flux = image[left_mask].sum()
    top_flux = image[top_mask].sum()
    bottom_flux = image[bottom_mask].sum()

    # Normalize so overall image brightness changes have less effect.
    horizontal_total = right_flux + left_flux
    vertical_total = top_flux + bottom_flux

    horizontal_signal = (
        (right_flux - left_flux) / horizontal_total
        if horizontal_total > 0
        else np.nan
    )

    vertical_signal = (
        (top_flux - bottom_flux) / vertical_total
        if vertical_total > 0
        else np.nan
    )

    return {
        "top": float(top_flux),
        "bottom": float(bottom_flux),
        "left": float(left_flux),
        "right": float(right_flux),
        "horizontal_signal": float(horizontal_signal),
        "vertical_signal": float(vertical_signal),
    }

def directional_annulus_signal(image, pupil):
    """
    Measure left-right and top-bottom intensity asymmetry in the annulus.

    Returns
    -------
    horizontal : float
        Positive means more annulus light on the right.
        Negative means more annulus light on the left.

    vertical : float
        Positive means more annulus light on the top.
        Negative means more annulus light on the bottom.
    """

    xc, yc = pupil["center"]
    radius = pupil["radius"]

    r_in = radius + pupil["annulus_inner_offset"]
    r_out = radius + pupil["annulus_outer_offset"]

    yy, xx = np.indices(image.shape)

    dx = xx - xc
    dy = yy - yc
    rr = np.sqrt(dx**2 + dy**2)

    annulus = (rr >= r_in) & (rr <= r_out)

    # Four non-overlapping 90-degree sectors
    right = annulus & (dx >= np.abs(dy))
    left = annulus & (-dx >= np.abs(dy))
    top = annulus & (dy >= np.abs(dx))
    bottom = annulus & (-dy >= np.abs(dx))

    right_flux = image[right].sum()
    left_flux = image[left].sum()
    top_flux = image[top].sum()
    bottom_flux = image[bottom].sum()

    horizontal_total = right_flux + left_flux
    vertical_total = top_flux + bottom_flux

    horizontal = (
        (right_flux - left_flux) / horizontal_total
        if horizontal_total > 0
        else np.nan
    )

    vertical = (
        (top_flux - bottom_flux) / vertical_total
        if vertical_total > 0
        else np.nan
    )

    return horizontal, vertical

def measure_annulus_drift(
    directory,
    measurement,
    init,
    flip_vertical=True,
):
    """
    Measure directional annulus asymmetry across a scan.

    The init-image asymmetry is subtracted so that the initial pupil
    position corresponds to approximately zero.

    Returns
    -------
    voltage : ndarray
        Scanned voltage.

    horizontal : ndarray
        Shape (n_images, n_pupils).
        Positive means motion toward the right side of the annulus.

    vertical : ndarray
        Shape (n_images, n_pupils).
        Positive means motion toward the top side of the annulus.

    fixed_voltage : float
        Voltage held constant.
    """

    if measurement not in ("x", "y"):
        raise ValueError("measurement must be 'x' or 'y'")

    pattern = os.path.join(
        directory,
        f"{measurement}_*.tif",
    )

    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matching '{measurement}_*.tif' found in {directory}"
        )

    pupils = init["pupils"]
    init_image = init["image"]

    n_images = len(files)
    n_pupils = len(pupils)

    voltage = np.zeros(n_images, dtype=float)
    horizontal = np.zeros((n_images, n_pupils), dtype=float)
    vertical = np.zeros((n_images, n_pupils), dtype=float)

    # Signal in the init image becomes the zero point.
    init_horizontal = np.zeros(n_pupils, dtype=float)
    init_vertical = np.zeros(n_pupils, dtype=float)

    for j, pupil in enumerate(pupils):
        init_horizontal[j], init_vertical[j] = (
            directional_annulus_signal(
                init_image,
                pupil,
            )
        )

    fixed_voltage = None

    for i, file in enumerate(files):

        image = plt.imread(file)

        if flip_vertical:
            image = np.flipud(image)

        parts = Path(file).stem.split("_")

        x_voltage = decode_voltage(parts[1])
        y_voltage = decode_voltage(parts[2])

        if measurement == "x":
            fixed_voltage = x_voltage
            voltage[i] = y_voltage

        else:
            fixed_voltage = y_voltage
            voltage[i] = x_voltage

        for j, pupil in enumerate(pupils):

            h_signal, v_signal = directional_annulus_signal(
                image,
                pupil,
            )

            horizontal[i, j] = h_signal - init_horizontal[j]
            vertical[i, j] = v_signal - init_vertical[j]

    order = np.argsort(voltage)

    return (
        voltage[order],
        horizontal[order],
        vertical[order],
        fixed_voltage,
    )

def plot_annulus_drift(
    measurement,
    voltage,
    horizontal,
    vertical,
    fixed_voltage,
):
    """
    Plot directional annulus asymmetry for every pupil.
    """

    if measurement == "x":
        xlabel = "Y Voltage (V)"
        title = f"Annulus Drift Indicator\nX = {fixed_voltage:.3f} V"

    elif measurement == "y":
        xlabel = "X Voltage (V)"
        title = f"Annulus Drift Indicator\nY = {fixed_voltage:.3f} V"

    else:
        raise ValueError("measurement must be 'x' or 'y'")

    markers = ["o", "s", "^", "d", "v", "*"]

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharex=True,
    )

    for i in range(horizontal.shape[1]):

        ax[0].plot(
            voltage,
            horizontal[:, i],
            marker=markers[i % len(markers)],
            label=f"Pupil {i + 1}",
        )

        ax[1].plot(
            voltage,
            vertical[:, i],
            marker=markers[i % len(markers)],
            label=f"Pupil {i + 1}",
        )

    ax[0].axhline(0, color="black", linewidth=1)
    ax[1].axhline(0, color="black", linewidth=1)

    ax[0].set_title("Left ↔ Right")
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel("Normalized right − left signal")

    ax[1].set_title("Bottom ↔ Top")
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel("Normalized top − bottom signal")

    for axis in ax:
        axis.grid(alpha=0.3)
        axis.legend()

    fig.suptitle(title)

    plt.tight_layout()
    plt.show()