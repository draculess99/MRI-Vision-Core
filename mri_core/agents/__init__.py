"""Shared specialist-critic-consensus evidence-review layer for MRI-Vision-Core.

This package is deliberately generic -- it consumes mri_core.rsna_inference_engine's
StudyInferenceResult and nothing else, so it can be reused by the OpenCV, Nebius, and AMD
hackathon branches without any of them duplicating this logic. See
mri_core/agents/README.md for the full architecture and integration guide.
"""
