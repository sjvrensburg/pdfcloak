from pdfcloak import sidecar


def test_user_binary_path_under_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert sidecar.user_binary_path() == tmp_path / "pdfcloak" / "bin" / "clipcloak-server"


def test_bundled_sources_found_in_repo():
    sources = sidecar.bundled_sources_dir()
    assert sources is not None
    assert (sources / "Cargo.toml").exists()
    assert (sources / "vendor-gliner2-rs" / "Cargo.toml").exists()


def test_finder_prefers_explicit(tmp_path):
    fake = tmp_path / "clipcloak-server"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    assert sidecar.find_sidecar_binary(fake) == fake


def test_build_sidecar_no_cargo(monkeypatch, tmp_path):
    import pdfcloak
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda name: None)
    try:
        sidecar.build_sidecar()
        raise AssertionError("expected PdfCloakError")
    except pdfcloak.PdfCloakError as exc:
        assert "cargo" in str(exc)
