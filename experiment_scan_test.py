# experiment_scan_test.py
# Translation and rotation acquisition test GUI.

# Import packages
import os
import csv
import numpy as np
from datetime import datetime

import tifffile
import tkinter as tk
import matplotlib.pyplot as plt
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
    "experiment_type",
    "series",
    "fixed_axis",
    "x_voltage_v",
    "y_voltage_v",
    "rotation_degrees",
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


def create_dataset_directory(experiment_type):
    """
    Create the next available data/YYYY-MM-DD/expNN directory.
    """
    date_text = datetime.now().strftime("%Y-%m-%d")
    base_directory = Path("data") / date_text
    base_directory.mkdir(parents=True, exist_ok=True)

    experiment_number = 1

    while True:
        directory = (
            base_directory
            / f"exp{experiment_number:02d}"
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
    fieldnames.append("summed_intensity")

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

            row["summed_intensity"] = float(flux[row_index].sum())
            writer.writerow(row)


def measure_rotation_series(directory, init, flip_vertical=True):
    """Measure pupil flux for rotation points recorded in metadata.csv."""
    directory = Path(directory)
    metadata_path = directory / "metadata.csv"

    with metadata_path.open("r", newline="", encoding="utf-8") as file:
        records = [
            row for row in csv.DictReader(file)
            if row["capture_type"] == "scan_point"
            and row["experiment_type"] == "rotation"
        ]

    if not records:
        raise RuntimeError("No rotation points were recorded.")

    pupils = init["pupils"]
    angles = np.zeros(len(records), dtype=float)
    flux = np.zeros((len(records), len(pupils)), dtype=float)
    x_values = []
    y_values = []

    for image_index, record in enumerate(records):
        angles[image_index] = float(record["rotation_degrees"])
        x_values.append(float(record["x_voltage_v"]))
        y_values.append(float(record["y_voltage_v"]))

        image_path = directory / record["filename"]
        image = tifffile.imread(image_path)

        if flip_vertical:
            image = np.flipud(image)

        for pupil_index, pupil in enumerate(pupils):
            flux[image_index, pupil_index] = od.pupil_flux(image, pupil)

    if not np.allclose(x_values, x_values[0], atol=1e-9):
        raise ValueError("X voltage changed during the rotation scan.")
    if not np.allclose(y_values, y_values[0], atol=1e-9):
        raise ValueError("Y voltage changed during the rotation scan.")

    order = np.argsort(angles)
    return angles[order], flux[order], float(x_values[0]), float(y_values[0])


def save_rotation_results(output_path, angles, flux, x_voltage, y_voltage):
    fieldnames = [
        "rotation_degrees",
        "x_voltage_v",
        "y_voltage_v",
    ] + [
        f"pupil_{index}_intensity"
        for index in range(1, flux.shape[1] + 1)
    ]
    fieldnames.append("summed_intensity")

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row_index, angle in enumerate(angles):
            row = {
                "rotation_degrees": angle,
                "x_voltage_v": x_voltage,
                "y_voltage_v": y_voltage,
            }
            for pupil_index in range(flux.shape[1]):
                row[f"pupil_{pupil_index + 1}_intensity"] = (
                    flux[row_index, pupil_index]
                )
            row["summed_intensity"] = float(flux[row_index].sum())
            writer.writerow(row)


def plot_rotation_intensity(
    angles,
    flux,
    x_voltage,
    y_voltage,
    output_path,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = ["o", "s", "^", "d", "v", "*"]

    for pupil_index in range(flux.shape[1]):
        ax.plot(
            angles,
            flux[:, pupil_index],
            marker=markers[pupil_index % len(markers)],
            linewidth=2,
            label=f"Pupil {pupil_index + 1}",
        )

    ax.plot(
        angles,
        flux.sum(axis=1),
        marker="X",
        linestyle="--",
        linewidth=2.5,
        color="black",
        label="Sum of all pupils",
    )

    ax.set_xlabel("Rotation angle (degrees)")
    ax.set_ylabel("Integrated pupil intensity")
    ax.set_title(
        f"Rotation Series Pupil Intensity\n"
        f"X = {x_voltage:.3f} V, Y = {y_voltage:.3f} V"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

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

        self.experiment_type_var = tk.StringVar(value="Translation")
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
            controls_frame,
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
            text="Experiment type:",
        ).grid(
            row=0,
            column=0,
            padx=5,
            pady=5,
            sticky="e",
        )

        self.experiment_type_box = ttk.Combobox(
            scan_frame,
            textvariable=self.experiment_type_var,
            values=("Translation", "Rotation"),
            state="readonly",
            width=12,
        )
        self.experiment_type_box.grid(
            row=0,
            column=1,
            padx=5,
            pady=5,
            sticky="w",
        )
        self.experiment_type_box.bind(
            "<<ComboboxSelected>>",
            self.on_experiment_type_changed,
        )

        self.series_label = ttk.Label(
            scan_frame,
            text="Series:",
        )
        self.series_label.grid(
            row=1,
            column=0,
            padx=5,
            pady=5,
            sticky="e",
        )

        self.series_box = ttk.Combobox(
            scan_frame,
            textvariable=self.series_var,
            values=("B", "T"),
            state="readonly",
            width=8,
        )
        self.series_box.grid(
            row=1,
            column=1,
            padx=5,
            pady=5,
            sticky="w",
        )

        self.fixed_axis_label = ttk.Label(
            scan_frame,
            text="Fixed axis:",
        )
        self.fixed_axis_label.grid(
            row=1,
            column=2,
            padx=5,
            pady=5,
            sticky="e",
        )

        self.fixed_axis_var = tk.StringVar(value="x")

        self.fixed_axis_box = ttk.Combobox(
            scan_frame,
            textvariable=self.fixed_axis_var,
            values=("x", "y"),
            state="readonly",
            width=8,
        )
        self.fixed_axis_box.grid(
            row=1,
            column=3,
            padx=5,
            pady=5,
            sticky="w",
        )

        ttk.Label(
            scan_frame,
            text="Initial X voltage:",
        ).grid(
            row=2,
            column=0,
            padx=5,
            pady=5,
            sticky="e",
        )

        initial_x_frame = ttk.Frame(scan_frame)
        initial_x_frame.grid(
            row=2,
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
            row=2,
            column=2,
            padx=5,
            pady=5,
            sticky="e",
        )

        initial_y_frame = ttk.Frame(scan_frame)
        initial_y_frame.grid(
            row=2,
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
            row=3,
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
            row=4,
            column=0,
            columnspan=4,
            pady=5,
        )

        scan_voltage_frame = ttk.Frame(scan_frame)
        scan_voltage_frame.grid(
            row=5,
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
            row=6,
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

        notes_frame = ttk.LabelFrame(
            controls_frame,
            text="Experiment notes",
            padding=8,
        )
        notes_frame.pack(
            fill="x",
            padx=20,
            pady=5,
        )

        self.notes_text = tk.Text(
            notes_frame,
            height=4,
            wrap="word",
        )
        self.notes_text.pack(
            fill="x",
            expand=True,
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

    def save_experiment_notes(self):
        """Write the current GUI notes and experiment context to notes.txt."""
        if self.dataset_directory is None or self.scan_config is None:
            return None

        user_notes = self.notes_text.get("1.0", "end-1c").strip()
        experiment_type = self.scan_config["experiment_type"]

        lines = [
            f"Experiment type: {experiment_type}",
            f"X voltage: {self.scan_config['initial_x']:.6f} V",
            f"Y voltage: {self.scan_config['initial_y']:.6f} V",
        ]

        if experiment_type == "translation":
            lines.extend(
                [
                    f"Series: {self.scan_config['series']}",
                    f"Fixed axis: {self.scan_config['fixed_axis']}",
                ]
            )

        lines.extend(
            [
                f"Last saved: {datetime.now().isoformat(timespec='seconds')}",
                "",
                "Notes:",
                user_notes,
                "",
            ]
        )

        notes_path = self.dataset_directory / "notes.txt"

        try:
            notes_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception as error:
            print(f"Could not save experiment notes: {error}")
            return None

        return notes_path

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

    def on_experiment_type_changed(self, _event=None):
        """Enable controls that apply to the selected experiment type."""
        if self.experiment_type_var.get() == "Rotation":
            self.series_box.config(state="disabled")
            self.fixed_axis_box.config(state="disabled")
            self.scan_voltage_label.config(text="Rotation angle (degrees):")
        else:
            self.series_box.config(state="readonly")
            self.fixed_axis_box.config(state="readonly")
            self.scan_voltage_label.config(text="Measured scan voltage:")

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

        experiment_type = self.experiment_type_var.get().lower()
        series = self.series_var.get() if experiment_type == "translation" else ""
        fixed_axis = (
            self.fixed_axis_var.get() if experiment_type == "translation" else ""
        )

        self.scan_config = {
            "experiment_type": experiment_type,
            "series": series,
            "fixed_axis": fixed_axis,
            "initial_x": initial_x,
            "initial_y": initial_y,
        }

        if experiment_type == "rotation":
            message = (
                f"Rotation scan confirmed: X={initial_x:.3f} V, "
                f"Y={initial_y:.3f} V."
            )
            self.scan_voltage_label.config(text="Rotation angle (degrees):")
        else:
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
                create_dataset_directory(
                    self.scan_config["experiment_type"]
                )
            )

            print(
                f"Created dataset directory: "
                f"{self.dataset_directory.resolve()}"
            )
            self.save_experiment_notes()

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
                    "experiment_type": self.scan_config["experiment_type"],
                    "series": self.scan_config["series"],
                    "fixed_axis": self.scan_config["fixed_axis"],
                    "x_voltage_v": initial_x,
                    "y_voltage_v": initial_y,
                    "rotation_degrees": "",
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
        self.experiment_type_box.config(state="disabled")
        self.series_box.config(state="disabled")
        self.fixed_axis_box.config(state="disabled")

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
        self.save_experiment_notes()

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

        experiment_type = self.scan_config["experiment_type"]
        field_name = (
            "rotation angle" if experiment_type == "rotation" else "scan voltage"
        )

        try:
            scanned_value = parse_signed_voltage(
                self.scan_sign_var.get(),
                self.scan_voltage_var.get(),
                field_name,
            )

        except ValueError as error:
            self.status.config(
                text=str(error)
            )
            return

        if experiment_type == "rotation":
            x_voltage = self.scan_config["initial_x"]
            y_voltage = self.scan_config["initial_y"]
            rotation_degrees = scanned_value
        else:
            fixed_axis = self.scan_config["fixed_axis"]
            rotation_degrees = None

            if fixed_axis == "x":
                x_voltage = self.scan_config["initial_x"]
                y_voltage = scanned_value
            else:
                x_voltage = scanned_value
                y_voltage = self.scan_config["initial_y"]

        self.pending_capture = {
            "kind": "scan_point",
            "x_voltage": x_voltage,
            "y_voltage": y_voltage,
            "rotation_degrees": rotation_degrees,
        }

        self.capture_point_button.config(
            state="disabled"
        )

        self.status.config(
            text="Waiting for the next scan frame..."
        )

    def save_scan_point(self, image, request):
        experiment_type = self.scan_config["experiment_type"]
        series = self.scan_config["series"]
        fixed_axis = self.scan_config["fixed_axis"]

        x_voltage = request["x_voltage"]
        y_voltage = request["y_voltage"]
        rotation_degrees = request["rotation_degrees"]

        capture_time = datetime.now()

        timestamp = capture_time.strftime(
            "%Y-%m-%dT%H-%M-%S.%f"
        )[:-3]

        if experiment_type == "rotation":
            filename = (
                f"rotation_D{encode_voltage(rotation_degrees)}_"
                f"X{encode_voltage(x_voltage)}_"
                f"Y{encode_voltage(y_voltage)}_"
                f"{timestamp}.tif"
            )
        else:
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
                    "experiment_type": experiment_type,
                    "series": series,
                    "fixed_axis": fixed_axis,
                    "x_voltage_v": x_voltage,
                    "y_voltage_v": y_voltage,
                    "rotation_degrees": (
                        rotation_degrees if rotation_degrees is not None else ""
                    ),
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

            if experiment_type == "rotation":
                point_message = (
                    f"Saved point {self.scan_point_count}: "
                    f"angle={rotation_degrees:.3f}°, "
                    f"X={x_voltage:.3f} V, Y={y_voltage:.3f} V"
                )
            else:
                point_message = (
                    f"Saved point {self.scan_point_count}: "
                    f"X={x_voltage:.3f} V, Y={y_voltage:.3f} V"
                )

            self.status.config(text=point_message)
            print(f"Capture {self.scan_point_count}: {point_message.split(': ', 1)[1]}")
            print(
                f"Saved scan point: "
                f"{output_path.resolve()}"
            )
            self.save_experiment_notes()

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

        if self.scan_point_count < 1:
            self.status.config(
                text="Capture at least one scan point before finishing."
            )
            return

        self.status.config(
            text="Closing camera and analyzing scan..."
        )
        self.update_idletasks()
        notes_path = self.save_experiment_notes()

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

            experiment_type = self.scan_config["experiment_type"]

            if experiment_type == "rotation":
                angles, flux, x_voltage, y_voltage = measure_rotation_series(
                    self.dataset_directory,
                    init,
                )
                results_path = (
                    self.dataset_directory / "rotation_integrated_intensity.csv"
                )
                plot_path = (
                    self.dataset_directory / "rotation_integrated_intensity.png"
                )
                save_rotation_results(
                    results_path,
                    angles,
                    flux,
                    x_voltage,
                    y_voltage,
                )
                plot_rotation_intensity(
                    angles,
                    flux,
                    x_voltage,
                    y_voltage,
                    plot_path,
                )
            else:
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

        if experiment_type == "rotation":
            message = (
                f"Finished rotation scan: "
                f"{self.scan_point_count} points analyzed."
            )
        else:
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

        if notes_path is not None:
            print(f"Notes saved: {notes_path.resolve()}")

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

        self.save_experiment_notes()
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



