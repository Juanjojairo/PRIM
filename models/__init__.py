"""Model modules for PEADMM-RIM."""

__all__ = [
    "MNISTGenerator",
    "MNISTDiscriminator",
    "ObservationEncoder",
    "ImageSpaceRIM",
    "run_eadmm",
    "run_peadmm",
]


def __getattr__(name: str):
    if name in {"MNISTGenerator", "MNISTDiscriminator"}:
        from models.generator import MNISTDiscriminator, MNISTGenerator

        mapping = {
            "MNISTGenerator": MNISTGenerator,
            "MNISTDiscriminator": MNISTDiscriminator,
        }
        return mapping[name]
    if name == "ObservationEncoder":
        from models.encoder import ObservationEncoder

        return ObservationEncoder
    if name == "ImageSpaceRIM":
        from models.rim import ImageSpaceRIM

        return ImageSpaceRIM
    if name in {"run_eadmm", "run_peadmm"}:
        from models.baselines import run_eadmm, run_peadmm

        mapping = {
            "run_eadmm": run_eadmm,
            "run_peadmm": run_peadmm,
        }
        return mapping[name]
    raise AttributeError(f"module 'models' has no attribute {name!r}")
