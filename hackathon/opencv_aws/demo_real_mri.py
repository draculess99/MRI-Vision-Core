from pathlib import Path
import sys

import pydicom
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hackathon.opencv_aws.opencv_mri import (
    process_mri_slice,
    save_result,
)


def find_first_dicom(root: Path) -> Path:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".dcm", ""}:
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=True)
                if hasattr(ds, "SOPInstanceUID"):
                    return path
            except Exception:
                pass

    raise FileNotFoundError(f"No readable DICOM file found under {root}")


def main():
    # Use your existing RSNA MRI data folder.
    data_root = Path(r"D:\DEVPOST\MRI-Vision-Core\data\rsna-knee")

    print("Searching for one real MRI DICOM...")
    dicom_path = find_first_dicom(data_root)

    print("Using DICOM:")
    print(dicom_path)

    ds = pydicom.dcmread(str(dicom_path))
    image = ds.pixel_array.astype(np.float32)

    print("Image shape:", image.shape)
    print("Image min/max:", float(image.min()), float(image.max()))

    result = process_mri_slice(image)

    output_dir = Path(
        r"D:\DEVPOST\MRI-Vision-Core\outputs\opencv_aws_demo"
    )

    paths = save_result(
        result,
        output_dir,
        prefix="real_knee_mri",
    )

    print("\nCreated demo artifacts:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()