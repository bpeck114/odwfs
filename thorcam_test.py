# thorcam_test.py
# Last edited: 08/10/26

import os
import numpy as np
from pathlib import Path
import tifffile
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="Capture one image from a Thorlabs camera."
)

parser.add_argument(
    "-e",
    "--exposure",
    type=float,
    default=0.0001,
    metavar="SECONDS",
    help="Exposure time in seconds; default: 0.0001 (100 µs)",
)

args = parser.parse_args()

if args.exposure <= 0:
    parser.error("--exposure must be greater than zero")

exposure_time_us = round(args.exposure * 1_000_000)

SDK_DLL_DIRECTORY = (
    r"C:\Users\testadmin\Downloads"
    r"\scientific_camera_interfaces_windows-2.1"
    r"\Scientific Camera Interfaces"
    r"\SDK\Native Toolkit\dlls\Native_64_lib"
)

os.environ["PATH"] = (
    SDK_DLL_DIRECTORY
    + os.pathsep
    + os.environ["PATH"]
)
dll_directory_handle = os.add_dll_directory(SDK_DLL_DIRECTORY)

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK


with TLCameraSDK() as sdk:
    serial_numbers = sdk.discover_available_cameras()

    if not serial_numbers:
        raise RuntimeError(
            "No available Thorlabs cameras were detected. "
            "Make sure ThorCam is closed."
        )

    serial_number = serial_numbers[0]
    print(f"Opening camera {serial_number}")

    with sdk.open_camera(serial_number) as camera:
        camera.exposure_time_us = exposure_time_us
        print(f"Exposure: {args.exposure:g} s ({exposure_time_us} µs)")

        camera.frames_per_trigger_zero_for_unlimited = 1
        camera.image_poll_timeout_ms = 2_000

        camera.arm(2)
        camera.issue_software_trigger()

        frame = camera.get_pending_frame_or_null()

        if frame is None:
            raise RuntimeError(
                "The camera did not return a frame within 2 seconds."
            )

        image = np.copy(frame.image_buffer)

        camera.disarm()

print(f"Shape: {image.shape}")
print(f"Data type: {image.dtype}")
print(f"Minimum value: {image.min()}")
print(f"Maximum value: {image.max()}")

output_path = Path("data") / "thorcam_test.tif"
output_path.parent.mkdir(parents=True, exist_ok=True)

tifffile.imwrite(output_path, image)

print(f"Saved image to: {output_path.resolve()}")

plt.imshow(image, cmap="gray", vmin=0, vmax=1023)
plt.colorbar(label="Camera value")
plt.title("CS165MU — single frame")
plt.axis("off")
plt.show()