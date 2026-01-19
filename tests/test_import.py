"""Smoke tests for package imports."""


def test_import_flux2_mlx():
    """Test that the main package can be imported."""
    import flux2_mlx
    assert hasattr(flux2_mlx, "__name__")


def test_import_pipeline():
    """Test that Flux2Pipeline can be imported."""
    from flux2_mlx import Flux2Pipeline
    assert Flux2Pipeline is not None


def test_import_defaults():
    """Test that defaults are accessible."""
    from flux2_mlx.defaults import (
        DEFAULT_WIDTH,
        DEFAULT_HEIGHT,
        DEFAULT_STEPS,
        DEFAULT_GUIDANCE,
    )
    assert DEFAULT_WIDTH == 512
    assert DEFAULT_HEIGHT == 512
    assert DEFAULT_STEPS == 4
    assert DEFAULT_GUIDANCE == 1.0
