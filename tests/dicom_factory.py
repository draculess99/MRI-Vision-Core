"""Synthetic DICOM series for the series-loader and RSNA adapter tests."""

from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

AXIAL = (1, 0, 0, 0, 1, 0)
CORONAL = (1, 0, 0, 0, 0, -1)
SAGITTAL = (0, 1, 0, 0, 0, -1)
STUDY_UID = "1.2.826.1"
SERIES_UID = "1.2.826.2"


def make_slice(number, position, *, orientation=AXIAL, series_uid=SERIES_UID, study_uid=STUDY_UID, rows=8, columns=8):
    """One MR slice whose pixels all equal 10 * InstanceNumber, positioned `position` mm along its slice normal."""
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset("slice.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "MR"
    ds.SeriesDescription = "SYNTH SERIES"
    ds.InstanceNumber = number
    ds.ImageOrientationPatient = [float(v) for v in orientation]
    normal = np.cross(orientation[:3], orientation[3:])
    ds.ImagePositionPatient = [float(v) for v in normal * position]
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 3.0
    ds.SpacingBetweenSlices = 4.0
    ds.Rows, ds.Columns = rows, columns
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.full((rows, columns), number * 10, dtype=np.uint16).tobytes()
    return ds


def write_series(directory, count=4, *, orientation=AXIAL, spacing=5.0, instance_numbers=None, filenames=None,
                 mutate=None, series_uid=SERIES_UID, study_uid=STUDY_UID, rows=8, columns=8):
    """Write a series and return its paths in file-write order.

    Physical position follows InstanceNumber ((number - 1) * spacing). By default filenames descend as
    InstanceNumber ascends, so sorting by filename gives the wrong order. `mutate(index, ds)` may edit each
    dataset (index is the write position) before it is saved.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    numbers = list(instance_numbers) if instance_numbers is not None else list(range(1, count + 1))
    paths = []
    for index, number in enumerate(numbers):
        ds = make_slice(number, (number - 1) * spacing, orientation=orientation, series_uid=series_uid,
                        study_uid=study_uid, rows=rows, columns=columns)
        if mutate is not None:
            mutate(index, ds)
        name = filenames[index] if filenames else f"{len(numbers) - index:03d}.dcm"
        path = directory / name
        ds.save_as(path)
        paths.append(path)
    return paths


def at(index, edit):
    """Build a mutate hook that applies edit(ds) to one slice only."""
    return lambda i, ds: edit(ds) if i == index else None
