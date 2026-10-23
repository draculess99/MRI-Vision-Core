# mri_core
# Lazy imports to avoid loading heavy dependencies (pydicom) unless actually used
# This allows mri_core.execution and mri_core.config to be imported without pydicom

import importlib

__all__ = [
    "load_image",
    "load_dicom",
    "load_nifti",
    "load_numpy",
    "load_mri",
    "detect_file_format",
    "MRIVolume",
    "preprocess_image",
    "preprocess_mri_slice",
    "robust_percentile_normalize",
    "validate_finite",
    "process_mri_image",
    "load_dicom_series",
    "DicomSeriesError",
    "DicomSeriesWarning",
    "discover_mrnet_root",
    "list_mrnet_exams",
    "load_mrnet_exam",
]


def __getattr__(name):
    """Lazy load module attributes on demand."""
    _lazy_imports = {
        "load_image": ("loader", "load_image"),
        "load_dicom": ("loader", "load_dicom"),
        "load_nifti": ("loader", "load_nifti"),
        "load_numpy": ("loader", "load_numpy"),
        "load_mri": ("loader", "load_mri"),
        "detect_file_format": ("loader", "detect_file_format"),
        "MRIVolume": ("mri_volume", "MRIVolume"),
        "preprocess_image": ("preprocessing", "preprocess_image"),
        "preprocess_mri_slice": ("preprocessing", "preprocess_mri_slice"),
        "robust_percentile_normalize": ("preprocessing", "robust_percentile_normalize"),
        "validate_finite": ("preprocessing", "validate_finite"),
        "process_mri_image": ("pipeline", "process_mri_image"),
        "load_dicom_series": ("dicom_series", "load_dicom_series"),
        "DicomSeriesError": ("dicom_series", "DicomSeriesError"),
        "DicomSeriesWarning": ("dicom_series", "DicomSeriesWarning"),
        "discover_mrnet_root": ("dataset_discovery", "discover_mrnet_root"),
        "list_mrnet_exams": ("dataset_discovery", "list_mrnet_exams"),
        "load_mrnet_exam": ("dataset_discovery", "load_mrnet_exam"),
    }

    if name in _lazy_imports:
        module_name, attr_name = _lazy_imports[name]
        module = importlib.import_module(f".{module_name}", __name__)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

