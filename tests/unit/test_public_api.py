"""Tests for the top-level mlfit public API surface."""
import importlib
import unittest


class TestTopLevelFunctions(unittest.TestCase):
    def test_recommend_function_exists(self):
        import mlfit
        assert callable(mlfit.recommend)

    def test_profile_function_exists(self):
        import mlfit
        assert callable(mlfit.profile)

    def test_advisor_class_exists(self):
        import mlfit
        assert callable(mlfit.Advisor)

    def test_detect_hardware_exists(self):
        import mlfit
        assert callable(mlfit.detect_hardware)


class TestAllSymbolsImportable(unittest.TestCase):
    EXPECTED = [
        "Advisor",
        "recommend",
        "profile",
        "HardwareProfile",
        "ModelProfile",
        "AlternativeModel",
        "RecommendResult",
        "ProfilingResult",
        "BenchmarkPoint",
        "BackendConfig",
        "FeasibilityScore",
        "detect_hardware",
        "__version__",
    ]

    def test_all_contains_expected_symbols(self):
        import mlfit
        for name in self.EXPECTED:
            assert name in mlfit.__all__, f"{name!r} missing from mlfit.__all__"

    def test_every_all_symbol_importable(self):
        mlfit = importlib.import_module("mlfit")
        for name in mlfit.__all__:
            assert hasattr(mlfit, name), f"mlfit.{name} not accessible"


class TestVersion(unittest.TestCase):
    def test_version_is_string(self):
        import mlfit
        assert isinstance(mlfit.__version__, str)
        assert len(mlfit.__version__) > 0

    def test_version_is_0_2_0(self):
        import mlfit
        assert mlfit.__version__ == "0.2.0"

    def test_version_matches_version_file(self):
        import mlfit
        from mlfit._version import __version__
        assert mlfit.__version__ == __version__
