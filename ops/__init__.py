"""Operators and metrics for PEADMM-RIM."""

__all__ = [
    "SinglePixelHadamardOperator",
    "fwht",
    "compute_reconstruction_metrics",
    "peak_signal_to_noise_ratio",
]


def __getattr__(name: str):
    if name == "SinglePixelHadamardOperator":
        from ops.forward_models import SinglePixelHadamardOperator

        return SinglePixelHadamardOperator
    if name == "fwht":
        from ops.hadamard import fwht

        return fwht
    if name == "compute_reconstruction_metrics":
        from ops.metrics import compute_reconstruction_metrics

        return compute_reconstruction_metrics
    if name == "peak_signal_to_noise_ratio":
        from ops.metrics import peak_signal_to_noise_ratio

        return peak_signal_to_noise_ratio
    raise AttributeError(f"module 'ops' has no attribute {name!r}")
