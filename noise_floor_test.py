# noise_floor_test.py
# Last edited: 08/11/26

# Import packages
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import tifffile
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from PIL import Image, ImageTk

import odwfs as od

SDK_DLL_DIRECTORY = Path(
    r"C:\Users\testadmin\Downloads"
    r"\scientific_camera_interfaces_windows-2.1"
    r"\Scientific Camera Interfaces"
    r"\SDK\Native Toolkit\dlls\Native_64_lib"
)

# Make the Thorlabs native libraries visible to this process.
os.environ["PATH"] = (
    str(SDK_DLL_DIRECTORY)
    + os.pathsep
    + os.environ["PATH"]
)

dll_directory_handle = os.add_dll_directory(
    str(SDK_DLL_DIRECTORY)
)

METADATA_COLUMNS = (
    "filename",
    "capture_type",
    "series",
    "fixed_axis",
    "x_voltage_v",
    "y_voltage_v",
    "exposure_us",
    "timestamp",
    "camera_serial",
    "image_height",
    "image_width",
    "minimum_value",
    "maximum_value",
)


def append_metadata(metadata_path, record):
    """
    Append one capture record, creating the CSV header if needed.
    """
    write_header = not metadata_path.exists()

    with metadata_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=METADATA_COLUMNS,
        )

        if write_header:
            writer.writeheader()

        writer.writerow(record)

def encode_voltage(voltage):
    """
    Convert a voltage into the filename format.

    Examples:
        -1.221 -> n1d221
         0.153 -> p0d153
    """
    sign = "p" if voltage >= 0 else "n"
    magnitude = f"{abs(voltage):.3f}".replace(".", "d")

    return sign + magnitude


def parse_signed_voltage(sign_text, magnitude_text, field_name):
    """Combine a Positive/Negative selection with a voltage magnitude."""
    if sign_text not in ("Positive", "Negative"):
        raise ValueError(f"Select Positive or Negative for {field_name}.")

    try:
        magnitude = float(magnitude_text)
    except ValueError as error:
        raise ValueError(f"Enter a numeric magnitude for {field_name}.") from error

    if magnitude < 0:
        raise ValueError(
            f"Enter a non-negative magnitude for {field_name}; "
            "use the sign dropdown for its sign."
        )

    return -magnitude if sign_text == "Negative" else magnitude


def create_dataset_directory():
    """
    Create the next available data/YYYY-MM-DD_expNN directory.
    """
    date_text = datetime.now().strftime("%Y-%m-%d")
    base_directory = Path("data")

    experiment_number = 1

    while True:
        directory = (
            base_directory
            / f"{date_text}_exp{experiment_number:02d}"
        )

        if not directory.exists():
            directory.mkdir(parents=True)
            return directory

        experiment_number += 1

def save_intensity_results(
    output_path,
    voltage,
    flux,
    series,
    fixed_axis,
    fixed_voltage,
):
    """
    Save the analyzed voltage and pupil intensities.
    """
    scanned_axis = (
        "y" if fixed_axis == "x" else "x"
    )

    fieldnames = [
        "series",
        "fixed_axis",
        "fixed_voltage_v",
        f"{scanned_axis}_voltage_v",
    ]

    fieldnames.extend(
        f"pupil_{index}_intensity"
        for index in range(1, flux.shape[1] + 1)
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row_index, scanned_voltage in enumerate(voltage):
            row = {
                "series": series,
                "fixed_axis": fixed_axis,
                "fixed_voltage_v": fixed_voltage,
                f"{scanned_axis}_voltage_v": scanned_voltage,
            }

            for pupil_index in range(flux.shape[1]):
                row[
                    f"pupil_{pupil_index + 1}_intensity"
                ] = flux[row_index, pupil_index]

            writer.writerow(row)

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

class AcquisitionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("ODWFS Acquisition")
        self.geometry("800x800")
        self.state("zoomed")

        self.sdk = None
        self.camera = None
        self.camera_serial = None
        self.preview_photo = None
        self.preview_job = None
        self.latest_image = None
        self.scan_config = None
        self.dataset_directory = None
        self.pending_capture = None
        self.initialization_path = None

        self.scan_point_count = 0
        self.scan_finished = False

        self.noise_frames = []
        self.noise_frame_numbers = []
        self.noise_capture_active = False
        self.noise_difference = None
        self.noise_output_directory = None

        self.series_var = tk.StringVar(value="B")
        self.fixed_axis_var = tk.StringVar(value="x")
        self.initial_x_sign_var = tk.StringVar()
        self.initial_x_var = tk.StringVar()
        self.initial_y_sign_var = tk.StringVar()
        self.initial_y_var = tk.StringVar()

        self.protocol("WM_DELETE_WINDOW", self.close_app)

        main_frame = ttk.Frame(self)
        main_frame.pack(
            fill="both",
            expand=True,
        )

        controls_frame = ttk.Frame(
            main_frame,
            padding=10,
        )
        controls_frame.pack(
            side="left",
            fill="y",
        )

        preview_frame = ttk.Frame(
            main_frame,
            padding=10,
        )
        preview_frame.pack(
            side="right",
            fill="both",
            expand=True,
        )

        title = ttk.Label(
            controls_frame,
            text="ODWFS Camera Acquisition",
            font=("TkDefaultFont", 16),
        )
        title.pack(pady=10)

        self.status = ttk.Label(
            self,
            text="Connecting to camera...",
        )
        self.status.pack(pady=5)

        exposure_frame = ttk.Frame(controls_frame)
        exposure_frame.pack(pady=5)

        ttk.Label(
            exposure_frame,
            text="Exposure (seconds):",
        ).pack(side="left", padx=5)

        self.exposure_var = tk.StringVar(value="0.0001")

        exposure_entry = ttk.Entry(
            exposure_frame,
            textvariable=self.exposure_var,
            width=12,
        )
        exposure_entry.pack(side="left", padx=5)

        apply_exposure_button = ttk.Button(
            exposure_frame,
            text="Apply",
            command=self.apply_exposure,
        )
        apply_exposure_button.pack(side="left", padx=5)

        noise_frame = ttk.LabelFrame(
            controls_frame,
            text="Noise-floor test",
            padding=10,
        )
        noise_frame.pack(
            fill="x",
            padx=20,
            pady=10,
        )

        self.noise_status = ttk.Label(
            noise_frame,
            text="Ready to capture two consecutive frames.",
        )
        self.noise_status.pack(
            side="left",
            padx=10,
        )

        self.noise_capture_button = ttk.Button(
            noise_frame,
            text="Capture Noise Pair",
            command=self.begin_noise_test,
        )
        self.noise_capture_button.pack(
            side="right",
            padx=10,
        )

        scan_frame = ttk.LabelFrame(
            controls_frame,
            text="Scan setup",
            padding=10,
        )
        scan_frame.pack(
            fill="x",
            padx=20,
            pady=10,
        )

        ttk.Label(
            scan_frame,
            text="Series:",
        ).grid(
            row=0,
            column=0,
            padx=5,
            pady=5,
            sticky="e",
        )

        series_box = ttk.Combobox(
            scan_frame,
            textvariable=self.series_var,
            values=("B", "T"),
            state="readonly",
            width=8,
        )
        series_box.grid(
            row=0,
            column=1,
            padx=5,
            pady=5,
            sticky="w",
        )

        ttk.Label(
            scan_frame,
            text="Fixed axis:",
        ).grid(
            row=0,
            column=2,
            padx=5,
            pady=5,
            sticky="e",
        )

        self.fixed_axis_var = tk.StringVar(value="x")

        fixed_axis_box = ttk.Combobox(
            scan_frame,
            textvariable=self.fixed_axis_var,
            values=("x", "y"),
            state="readonly",
            width=8,
        )
        fixed_axis_box.grid(
            row=0,
            column=3,
            padx=5,
            pady=5,
            sticky="w",
        )

        ttk.Label(
            scan_frame,
            text="Initial X voltage:",
        ).grid(
            row=1,
            column=0,
            padx=5,
            pady=5,
            sticky="e",
        )

        initial_x_frame = ttk.Frame(scan_frame)
        initial_x_frame.grid(
            row=1,
            column=1,
            padx=5,
            pady=5,
            sticky="w",
        )

        ttk.Combobox(
            initial_x_frame,
            textvariable=self.initial_x_sign_var,
            values=("Positive", "Negative"),
            state="readonly",
            width=9,
        ).pack(side="left", padx=(0, 5))

        ttk.Entry(
            initial_x_frame,
            textvariable=self.initial_x_var,
            width=9,
        ).pack(side="left")

        ttk.Label(
            scan_frame,
            text="Initial Y voltage:",
        ).grid(
            row=1,
            column=2,
            padx=5,
            pady=5,
            sticky="e",
        )

        initial_y_frame = ttk.Frame(scan_frame)
        initial_y_frame.grid(
            row=1,
            column=3,
            padx=5,
            pady=5,
            sticky="w",
        )

        ttk.Combobox(
            initial_y_frame,
            textvariable=self.initial_y_sign_var,
            values=("Positive", "Negative"),
            state="readonly",
            width=9,
        ).pack(side="left", padx=(0, 5))

        ttk.Entry(
            initial_y_frame,
            textvariable=self.initial_y_var,
            width=9,
        ).pack(side="left")

        self.confirm_button = ttk.Button(
            scan_frame,
            text="Confirm Setup",
            command=self.confirm_scan_setup,
        )
        self.confirm_button.grid(
            row=2,
            column=0,
            columnspan=4,
            pady=10,
        )

        self.capture_init_button = ttk.Button(
            scan_frame,
            text="Capture Initialization",
            command=self.request_initialization_capture,
            state="disabled",
        )
        self.capture_init_button.grid(
            row=3,
            column=0,
            columnspan=4,
            pady=5,
        )

        scan_voltage_frame = ttk.Frame(scan_frame)
        scan_voltage_frame.grid(
            row=4,
            column=0,
            columnspan=4,
            pady=10,
        )

        self.finish_button = ttk.Button(
            scan_frame,
            text="Finish Scan",
            command=self.finish_scan,
            state="disabled",
        )
        self.finish_button.grid(
            row=5,
            column=0,
            columnspan=4,
            pady=10,
        )

        self.scan_voltage_label = ttk.Label(
            scan_voltage_frame,
            text="Measured scan voltage:",
        )
        self.scan_voltage_label.pack(
            side="left",
            padx=5,
        )

        self.scan_voltage_var = tk.StringVar()

        self.scan_sign_var = tk.StringVar()

        self.scan_sign_box = ttk.Combobox(
            scan_voltage_frame,
            textvariable=self.scan_sign_var,
            values=("Positive", "Negative"),
            state="disabled",
            width=9,
        )
        self.scan_sign_box.pack(
            side="left",
            padx=5,
        )

        self.scan_voltage_entry = ttk.Entry(
            scan_voltage_frame,
            textvariable=self.scan_voltage_var,
            width=12,
            state="disabled",
        )
        self.scan_voltage_entry.pack(
            side="left",
            padx=5,
        )

        self.capture_point_button = ttk.Button(
            scan_voltage_frame,
            text="Capture Point",
            command=self.request_scan_point_capture,
            state="disabled",
        )
        self.capture_point_button.pack(
            side="left",
            padx=5,
        )

        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(
            expand=True,
            padx=10,
            pady=10,
        )

        capture_button = ttk.Button(
            controls_frame,
            text="Save Test Frame",
            command=self.save_test_frame,
        )
        capture_button.pack(pady=5)

        close_button = ttk.Button(
            controls_frame,
            text="Close",
            command=self.close_app,
        )
        close_button.pack(pady=10)

        self.after(100, self.connect_camera)

    def begin_noise_test(self):
        if self.camera is None:
            self.noise_status.config(
                text="Camera is not connected."
            )
            return

        if self.noise_capture_active:
            self.noise_status.config(
                text="A noise capture is already running."
            )
            return

        if self.pending_capture is not None:
            self.noise_status.config(
                text="Wait for the pending scan capture."
            )
            return

        self.noise_frames = []
        self.noise_frame_numbers = []
        self.noise_difference = None

        self.noise_capture_button.config(
            state="disabled"
        )

        self.noise_status.config(
            text="Preparing two-frame capture..."
        )
        self.update_idletasks()

        try:
            # Stop unlimited live acquisition.
            self.camera.disarm()

            # Request exactly two frames from one trigger.
            self.camera.frames_per_trigger_zero_for_unlimited = 2
            self.camera.image_poll_timeout_ms = 0

            self.camera.arm(2)

            self.noise_capture_active = True

            self.camera.issue_software_trigger()


        except Exception as error:

            self.noise_capture_active = False

            self.noise_capture_button.config(
                state="normal"
            )

            self.noise_status.config(
                text=f"Could not start noise capture: {error}"
            )

            print(f"Could not start noise capture: {error}")
            return

        self.noise_status.config(
            text="Waiting for frame 1 of 2..."
        )

        print("Noise-pair capture started.")

    def finish_noise_pair_capture(self):
        self.noise_capture_active = False

        if len(self.noise_frames) != 2:
            self.noise_status.config(
                text="Noise-pair capture was incomplete."
            )
            self.noise_capture_button.config(
                state="normal"
            )
            return

        first_image = self.noise_frames[0]
        second_image = self.noise_frames[1]

        first_number = self.noise_frame_numbers[0]
        second_number = self.noise_frame_numbers[1]

        frame_gap = second_number - first_number

        # The two-frame acquisition is finished.
        # Restore the camera's normal continuous-preview mode.
        try:
            self.camera.disarm()

            self.camera.frames_per_trigger_zero_for_unlimited = 0
            self.camera.image_poll_timeout_ms = 0

            self.camera.arm(2)
            self.camera.issue_software_trigger()

        except Exception as error:
            self.noise_status.config(
                text=f"Could not restart live view: {error}"
            )
            self.noise_capture_button.config(
                state="normal"
            )

            print(f"Could not restart live view: {error}")
            return

        if first_image.shape != second_image.shape:
            self.noise_status.config(
                text="The two frames have different dimensions."
            )
            self.noise_capture_button.config(
                state="normal"
            )

            print(
                "Cannot subtract frames with different dimensions."
            )
            return

        # Convert before subtraction so negative differences
        # remain negative rather than wrapping around.
        first_signed = first_image.astype(np.int16)
        second_signed = second_image.astype(np.int16)

        self.noise_difference = (
                second_signed - first_signed
        )

        difference_mean = float(
            self.noise_difference.mean()
        )

        difference_std = float(
            self.noise_difference.std(ddof=1)
        )

        single_frame_noise = (
                difference_std / np.sqrt(2)
        )

        capture_time = datetime.now()

        timestamp = capture_time.strftime(
            "%Y-%m-%dT%H-%M-%S.%f"
        )[:-3]

        exposure_us = self.camera.exposure_time_us

        self.noise_output_directory = (
                Path("data")
                / "noise_tests"
                / (
                    f"{timestamp}_"
                    f"{exposure_us}us_"
                    f"frames_{first_number}-{second_number}"
                )
        )

        try:
            self.noise_output_directory.mkdir(
                parents=True,
                exist_ok=False,
            )

            first_path = (
                    self.noise_output_directory
                    / "frame_1.tif"
            )

            second_path = (
                    self.noise_output_directory
                    / "frame_2.tif"
            )

            difference_path = (
                    self.noise_output_directory
                    / "difference_frame_2_minus_frame_1.tif"
            )

            tifffile.imwrite(
                first_path,
                first_image,
            )

            tifffile.imwrite(
                second_path,
                second_image,
            )

            tifffile.imwrite(
                difference_path,
                self.noise_difference,
            )

            # Verify that the signed difference survives saving.
            reloaded_difference = tifffile.imread(
                difference_path
            )

            if reloaded_difference.dtype != np.int16:
                raise RuntimeError(
                    "Reloaded difference image is not int16."
                )

            if not np.array_equal(
                    reloaded_difference,
                    self.noise_difference,
            ):
                raise RuntimeError(
                    "Reloaded difference does not match memory."
                )

        except Exception as error:
            self.noise_capture_button.config(
                state="normal"
            )

            self.noise_status.config(
                text=f"Could not save noise images: {error}"
            )

            print(f"Could not save noise images: {error}")
            return

        plot_path = None

        try:
            plot_path = self.create_noise_plot(
                self.noise_difference,
                difference_mean,
                difference_std,
                single_frame_noise,
                self.noise_output_directory,
            )

        except Exception as error:
            print(
                f"Noise images were saved, "
                f"but plotting failed: {error}"
            )

        self.noise_capture_button.config(
            state="normal"
        )

        self.noise_status.config(
            text=(
                f"Captured 2 consecutive frames; "
                f"noise={single_frame_noise:.3f} ADU."
            )
        )

        print(
            f"Difference: "
            f"range={self.noise_difference.min()}–"
            f"{self.noise_difference.max()}, "
            f"mean={difference_mean:.4f} ADU, "
            f"std={difference_std:.4f} ADU"
        )

        print(
            f"Estimated single-frame temporal noise: "
            f"{single_frame_noise:.4f} ADU"
        )

        print("Noise-pair capture complete.")

        print(
            f"Frame 1: number={first_number}, "
            f"shape={first_image.shape}, "
            f"range={first_image.min()}–{first_image.max()}, "
            f"mean={first_image.mean():.3f}"
        )

        print(
            f"Noise-test directory: "
            f"{self.noise_output_directory.resolve()}"
        )

        print(f"Saved raw frame 1: {first_path.name}")
        print(f"Saved raw frame 2: {second_path.name}")

        print(
            f"Saved signed difference: "
            f"{difference_path.name}"
        )

        print(
            f"Verified difference TIFF: "
            f"dtype={reloaded_difference.dtype}, "
            f"range={reloaded_difference.min()}–"
            f"{reloaded_difference.max()}"
        )

        print(
            f"Frame 2: number={second_number}, "
            f"shape={second_image.shape}, "
            f"range={second_image.min()}–{second_image.max()}, "
            f"mean={second_image.mean():.3f}"
        )

        print(f"Frame-number gap: {frame_gap}")

        if frame_gap == 1:
            print("The two frames were consecutive.")
        else:
            print(
                "Warning: the frames were not consecutive."
            )

        if plot_path is not None:
            print(
                f"Saved noise plot: "
                f"{plot_path.resolve()}"
            )

            self.show_saved_plot(plot_path)

    def connect_camera(self):
        try:
            self.sdk = TLCameraSDK()
            serial_numbers = self.sdk.discover_available_cameras()

            if not serial_numbers:
                self.status.config(
                    text="No available camera detected."
                )
                return

            serial_number = serial_numbers[0]
            self.camera_serial = serial_number
            self.camera = self.sdk.open_camera(serial_number)

            # Start with a 100 µs exposure.
            self.camera.exposure_time_us = 100

            # Zero means continuous acquisition.
            self.camera.frames_per_trigger_zero_for_unlimited = 0

            # Frame polling should not block the GUI.
            self.camera.image_poll_timeout_ms = 0

            self.camera.arm(2)
            self.camera.issue_software_trigger()

            self.status.config(
                text=f"Live camera: {serial_number} — exposure: 100 µs"
            )
            print(f"Opened camera: {serial_number}")

            self.update_live_view()

        except Exception as error:
            self.status.config(
                text=f"Camera connection failed: {error}"
            )
            print(f"Camera connection failed: {error}")
            self.release_camera()

    def apply_exposure(self):
        if self.camera is None:
            self.status.config(
                text="Cannot set exposure: camera is not connected."
            )
            return

        try:
            exposure_seconds = float(self.exposure_var.get())

            if exposure_seconds <= 0:
                raise ValueError

        except ValueError:
            self.status.config(
                text="Exposure must be a positive number."
            )
            return

        exposure_us = round(exposure_seconds * 1_000_000)

        minimum_us, maximum_us = (
            self.camera.exposure_time_range_us
        )

        if not minimum_us <= exposure_us <= maximum_us:
            self.status.config(
                text=(
                    f"Exposure must be between "
                    f"{minimum_us / 1_000_000:g} and "
                    f"{maximum_us / 1_000_000:g} seconds."
                )
            )
            return

        try:
            # Pause acquisition before changing the setting.
            self.camera.disarm()

            self.camera.exposure_time_us = exposure_us

            self.camera.arm(2)
            self.camera.issue_software_trigger()

            actual_us = self.camera.exposure_time_us

            self.status.config(
                text=(
                    f"Live camera — exposure: "
                    f"{actual_us / 1_000_000:g} s "
                    f"({actual_us} µs)"
                )
            )

            print(
                f"Exposure changed to "
                f"{actual_us / 1_000_000:g} s "
                f"({actual_us} µs)"
            )

        except Exception as error:
            self.status.config(
                text=f"Could not change exposure: {error}"
            )
            print(f"Could not change exposure: {error}")

    def update_live_view(self):
        if self.camera is None:
            return

        try:
            frame = self.camera.get_pending_frame_or_null()

            if frame is not None:
                # Copy the unmodified camera data before making the preview.
                self.latest_image = np.copy(frame.image_buffer)

                if self.noise_capture_active:
                    self.noise_frames.append(
                        np.copy(self.latest_image)
                    )
                    self.noise_frame_numbers.append(
                        frame.frame_count
                    )

                    captured_count = len(self.noise_frames)

                    self.noise_status.config(
                        text=f"Captured frame {captured_count} of 2."
                    )

                    if captured_count == 2:
                        self.finish_noise_pair_capture()

                # After frame 1, immediately poll for frame 2 without
                # spending time resizing and displaying the preview.
                if self.noise_capture_active:
                    self.preview_job = self.after(
                        1,
                        self.update_live_view,
                    )
                    return

                if self.pending_capture is not None:
                    request = self.pending_capture
                    self.pending_capture = None

                    if request["kind"] == "initialization":
                        self.save_initialization(
                            self.latest_image
                        )

                    elif request["kind"] == "scan_point":
                        self.save_scan_point(
                            self.latest_image,
                            request,
                        )

                shift = self.camera.bit_depth - 8

                display_array = (
                    frame.image_buffer >> shift
                ).astype("uint8")

                image = Image.fromarray(display_array)
                image = image.resize(
                    (600, 450),
                    Image.Resampling.LANCZOS,
                )

                self.preview_photo = ImageTk.PhotoImage(image)

                self.preview_label.config(
                    image=self.preview_photo
                )

        except Exception as error:
            self.status.config(
                text=f"Live-view error: {error}"
            )
            print(f"Live-view error: {error}")
            return

        # Check for another frame in approximately 10 ms.
        self.preview_job = self.after(
            10,
            self.update_live_view,
        )

    def confirm_scan_setup(self):
        try:
            initial_x = parse_signed_voltage(
                self.initial_x_sign_var.get(),
                self.initial_x_var.get(),
                "initial X voltage",
            )
            initial_y = parse_signed_voltage(
                self.initial_y_sign_var.get(),
                self.initial_y_var.get(),
                "initial Y voltage",
            )

        except ValueError as error:
            self.status.config(
                text=str(error)
            )
            return

        series = self.series_var.get()
        fixed_axis = self.fixed_axis_var.get()

        self.scan_config = {
            "series": series,
            "fixed_axis": fixed_axis,
            "initial_x": initial_x,
            "initial_y": initial_y,
        }

        if fixed_axis == "x":
            fixed_voltage = initial_x
            scanned_axis = "Y"
        else:
            fixed_voltage = initial_y
            scanned_axis = "X"

        message = (
            f"{series} series confirmed: "
            f"{fixed_axis.upper()} fixed at "
            f"{fixed_voltage:.3f} V; "
            f"{scanned_axis} will be scanned."
        )

        self.scan_voltage_label.config(
            text=f"Measured {scanned_axis} voltage:"
        )

        self.status.config(text=message)
        print(message)
        print(
            f"Initialization point: "
            f"X={initial_x:.3f} V, "
            f"Y={initial_y:.3f} V"
        )
        self.capture_init_button.config(state="normal")

    def request_initialization_capture(self):
        if self.scan_config is None:
            self.status.config(
                text="Confirm the scan setup first."
            )
            return

        if self.camera is None:
            self.status.config(
                text="Camera is not connected."
            )
            return

        if self.pending_capture is not None:
            self.status.config(
                text="A capture is already pending."
            )
            return

        if self.dataset_directory is None:
            self.dataset_directory = (
                create_dataset_directory()
            )

            print(
                f"Created dataset directory: "
                f"{self.dataset_directory.resolve()}"
            )

        self.pending_capture = {
            "kind": "initialization",
        }

        self.status.config(
            text="Waiting for the next frame..."
        )

        self.capture_init_button.config(
            state="disabled"
        )

    def save_initialization(self, image):
        initial_x = self.scan_config["initial_x"]
        initial_y = self.scan_config["initial_y"]

        capture_time = datetime.now()

        timestamp = capture_time.strftime(
            "%Y-%m-%dT%H-%M-%S.%f"
        )[:-3]

        filename = (
            f"init_"
            f"X{encode_voltage(initial_x)}_"
            f"Y{encode_voltage(initial_y)}_"
            f"{timestamp}.tif"
        )

        output_path = (
                self.dataset_directory / filename
        )

        try:
            tifffile.imwrite(
                output_path,
                np.copy(image),
            )

            append_metadata(
                self.dataset_directory / "metadata.csv",
                {
                    "filename": filename,
                    "capture_type": "initialization",
                    "series": self.scan_config["series"],
                    "fixed_axis": self.scan_config["fixed_axis"],
                    "x_voltage_v": initial_x,
                    "y_voltage_v": initial_y,
                    "exposure_us": self.camera.exposure_time_us,
                    "timestamp": capture_time.isoformat(
                        timespec="milliseconds"
                    ),
                    "camera_serial": self.camera_serial,
                    "image_height": image.shape[0],
                    "image_width": image.shape[1],
                    "minimum_value": int(image.min()),
                    "maximum_value": int(image.max()),
                },
            )

        except Exception as error:
            self.status.config(
                text=f"Could not save initialization: {error}"
            )
            self.capture_init_button.config(
                state="normal"
            )
            print(f"Initialization save failed: {error}")
            return

        self.initialization_path = output_path
        self.confirm_button.config(state="disabled")

        self.status.config(
            text=f"Saved initialization: {filename}"
        )

        print(
            f"Capture 0 (initialization): "
            f"X={initial_x:.3f} V, "
            f"Y={initial_y:.3f} V"
        )
        print(
            f"Saved initialization: "
            f"{output_path.resolve()}"
        )

        self.scan_voltage_entry.config(
            state="normal"
        )
        self.scan_sign_box.config(
            state="readonly"
        )
        self.capture_point_button.config(
            state="normal"
        )
        self.scan_voltage_entry.focus_set()
        self.finish_button.config(
            state="normal"
        )

    def request_scan_point_capture(self):
        if self.initialization_path is None:
            self.status.config(
                text="Capture the initialization image first."
            )
            return

        if self.pending_capture is not None:
            self.status.config(
                text="A capture is already pending."
            )
            return

        try:
            scanned_voltage = parse_signed_voltage(
                self.scan_sign_var.get(),
                self.scan_voltage_var.get(),
                "scan voltage",
            )

        except ValueError as error:
            self.status.config(
                text=str(error)
            )
            return

        fixed_axis = self.scan_config["fixed_axis"]

        if fixed_axis == "x":
            x_voltage = self.scan_config["initial_x"]
            y_voltage = scanned_voltage
        else:
            x_voltage = scanned_voltage
            y_voltage = self.scan_config["initial_y"]

        self.pending_capture = {
            "kind": "scan_point",
            "x_voltage": x_voltage,
            "y_voltage": y_voltage,
        }

        self.capture_point_button.config(
            state="disabled"
        )

        self.status.config(
            text="Waiting for the next scan frame..."
        )

    def save_scan_point(self, image, request):
        series = self.scan_config["series"]
        fixed_axis = self.scan_config["fixed_axis"]

        x_voltage = request["x_voltage"]
        y_voltage = request["y_voltage"]

        capture_time = datetime.now()

        timestamp = capture_time.strftime(
            "%Y-%m-%dT%H-%M-%S.%f"
        )[:-3]

        filename = (
            f"{series}_series_{fixed_axis}_"
            f"X{encode_voltage(x_voltage)}_"
            f"Y{encode_voltage(y_voltage)}_"
            f"{timestamp}.tif"
        )

        output_path = (
                self.dataset_directory / filename
        )

        try:
            tifffile.imwrite(
                output_path,
                np.copy(image),
            )

            append_metadata(
                self.dataset_directory / "metadata.csv",
                {
                    "filename": filename,
                    "capture_type": "scan_point",
                    "series": series,
                    "fixed_axis": fixed_axis,
                    "x_voltage_v": x_voltage,
                    "y_voltage_v": y_voltage,
                    "exposure_us": self.camera.exposure_time_us,
                    "timestamp": capture_time.isoformat(
                        timespec="milliseconds"
                    ),
                    "camera_serial": self.camera_serial,
                    "image_height": image.shape[0],
                    "image_width": image.shape[1],
                    "minimum_value": int(image.min()),
                    "maximum_value": int(image.max()),
                },
            )

        except Exception as error:
            self.status.config(
                text=f"Could not save scan point: {error}"
            )
            print(f"Scan-point save failed: {error}")

        else:
            self.scan_point_count += 1

            self.status.config(
                text=(
                    f"Saved point {self.scan_point_count}: "
                    f"X={x_voltage:.3f} V, "
                    f"Y={y_voltage:.3f} V"
                )
            )

            print(
                f"Capture {self.scan_point_count}: "
                f"X={x_voltage:.3f} V, "
                f"Y={y_voltage:.3f} V"
            )
            print(
                f"Saved scan point: "
                f"{output_path.resolve()}"
            )

            self.scan_voltage_var.set("")
            self.scan_sign_var.set("")
            self.scan_voltage_entry.focus_set()

        finally:
            self.capture_point_button.config(
                state="normal"
            )

    def save_test_frame(self):
        if self.latest_image is None:
            self.status.config(
                text="No camera frame is available yet."
            )
            return

        timestamp = datetime.now().strftime(
            "%Y-%m-%dT%H-%M-%S.%f"
        )[:-3]

        output_directory = (
                Path("data") / "test_captures"
        )
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
                output_directory
                / f"test_{timestamp}.tif"
        )

        # Copy again so the displayed stream cannot change this image.
        image_to_save = np.copy(self.latest_image)

        tifffile.imwrite(
            output_path,
            image_to_save,
        )

        self.status.config(
            text=f"Saved: {output_path.name}"
        )

        print(f"Saved: {output_path.resolve()}")
        print(
            f"Shape: {image_to_save.shape}, "
            f"range: {image_to_save.min()}–"
            f"{image_to_save.max()}"
        )

    def create_noise_plot(
            self,
            difference,
            difference_mean,
            difference_std,
            single_frame_noise,
            output_directory,
    ):
        """
        Save a signed difference image and pixel histogram.
        """
        absolute_difference = np.abs(
            difference.astype(float)
        )

        # Use the central 99.5% of values for display contrast.
        # Extreme pixels remain preserved in the TIFF.
        color_limit = float(
            np.percentile(
                absolute_difference,
                99.5,
            )
        )

        if color_limit == 0:
            color_limit = 1.0

        plot_path = (
                output_directory
                / "difference_analysis.png"
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 5),
        )

        image_plot = axes[0].imshow(
            difference,
            cmap="coolwarm",
            vmin=-color_limit,
            vmax=color_limit,
        )

        axes[0].set_title(
            "Frame 2 − Frame 1"
        )
        axes[0].set_xlabel("X pixel")
        axes[0].set_ylabel("Y pixel")

        fig.colorbar(
            image_plot,
            ax=axes[0],
            label="Difference (ADU)",
        )

        axes[1].hist(
            difference.ravel(),
            bins=101,
            range=(-color_limit, color_limit),
            color="steelblue",
            edgecolor="black",
            linewidth=0.3,
        )

        axes[1].axvline(
            difference_mean,
            color="red",
            linestyle="--",
            label=f"Mean = {difference_mean:.3f}",
        )

        axes[1].set_title(
            "Difference-Value Distribution"
        )
        axes[1].set_xlabel("Frame 2 − Frame 1 (ADU)")
        axes[1].set_ylabel("Pixel count")
        axes[1].legend()

        fig.suptitle(
            f"Difference σ = {difference_std:.3f} ADU; "
            f"single-frame noise = "
            f"{single_frame_noise:.3f} ADU"
        )

        fig.tight_layout()

        fig.savefig(
            plot_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        return plot_path

    def show_saved_plot(self, plot_path):
        plot_window = tk.Toplevel(self)
        plot_window.title("Integrated Pupil Intensity")

        image = Image.open(plot_path)

        # Fit the plot comfortably on screen while preserving its shape.
        image.thumbnail(
            (900, 650),
            Image.Resampling.LANCZOS,
        )

        plot_photo = ImageTk.PhotoImage(image)

        plot_label = ttk.Label(
            plot_window,
            image=plot_photo,
        )
        plot_label.pack(
            padx=10,
            pady=10,
        )

        # Tkinter must retain a reference to the image.
        plot_window.plot_photo = plot_photo

        close_button = ttk.Button(
            plot_window,
            text="Close Plot",
            command=plot_window.destroy,
        )
        close_button.pack(pady=10)

    def finish_scan(self):
        if self.initialization_path is None:
            self.status.config(
                text="No initialization image has been captured."
            )
            return

        if self.pending_capture is not None:
            self.status.config(
                text="Wait for the pending capture to finish."
            )
            return

        if self.scan_point_count < 2:
            self.status.config(
                text="Capture at least two scan points before finishing."
            )
            return

        self.status.config(
            text="Closing camera and analyzing scan..."
        )
        self.update_idletasks()

        # Release the camera before doing CPU-intensive analysis.
        self.release_camera()

        self.capture_point_button.config(
            state="disabled"
        )
        self.scan_voltage_entry.config(
            state="disabled"
        )
        self.scan_sign_box.config(
            state="disabled"
        )
        self.finish_button.config(
            state="disabled"
        )

        try:
            init = od.find_init_pupils(
                self.initialization_path,
                plot=False,
            )

            (
                voltage,
                flux,
                fixed_voltage,
                series,
                fixed_axis,
            ) = od.measure_recorded_series(
                self.dataset_directory,
                init,
            )

            results_path = (
                    self.dataset_directory
                    / (
                        f"{series}_series_{fixed_axis}_"
                        f"integrated_intensity.csv"
                    )
            )

            save_intensity_results(
                results_path,
                voltage,
                flux,
                series,
                fixed_axis,
                fixed_voltage,
            )

            plot_path = (
                    self.dataset_directory
                    / (
                        f"{series}_series_{fixed_axis}_"
                        f"integrated_intensity.png"
                    )
            )

            od.plot_intensity(
                fixed_axis,
                voltage,
                flux,
                fixed_voltage,
                ylabel="Integrated pupil intensity",
                title_prefix=f"{series} Series Pupil",
                output_path=plot_path,
                show=False,
            )

        except Exception as error:
            self.status.config(
                text=(
                    "Images were saved, but analysis failed: "
                    f"{error}"
                )
            )

            print(f"Analysis failed: {error}")
            print(
                f"Images remain saved in: "
                f"{self.dataset_directory.resolve()}"
            )
            return

        self.scan_finished = True

        message = (
            f"Finished {series} series with "
            f"{fixed_axis.upper()} fixed: "
            f"{self.scan_point_count} points analyzed."
        )

        self.status.config(text=message)

        print(message)
        print(
            f"Dataset: "
            f"{self.dataset_directory.resolve()}"
        )
        print(f"Plot saved: {plot_path.resolve()}")
        self.show_saved_plot(plot_path)

        print(
            f"Numerical results saved: "
            f"{results_path.resolve()}"
        )

    def release_camera(self):
        if self.preview_job is not None:
            try:
                self.after_cancel(self.preview_job)
            except Exception:
                pass

            self.preview_job = None

        if self.camera is not None:
            try:
                self.camera.disarm()
            except Exception:
                pass

            self.camera.dispose()
            self.camera = None

        if self.sdk is not None:
            self.sdk.dispose()
            self.sdk = None

    def close_app(self):
        self.status.config(text="Closing camera...")
        self.update_idletasks()

        self.release_camera()
        self.destroy()

        print("Camera released.")

def main():
    app = AcquisitionApp()
    app.mainloop()

    print("Acquisition window closed cleanly.")
    print("Goodbye!")

if __name__ == "__main__":
    main()



