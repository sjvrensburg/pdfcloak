"""OCR path, policy profiles, batch mode, web UI, and --lang tests."""
import json
import os
import stat
import threading
import urllib.request

import pymupdf as fitz
import pytest

from pdfcloak import PdfCloakError
from pdfcloak.cli import main
from pdfcloak.ocr import run_ocr
from pdfcloak.profiles import get_profile


# --- OCR (fake ocrmypdf on PATH; no real engine needed) --------------------

@pytest.fixture()
def fake_ocrmypdf(tmp_path, monkeypatch):
    """Fake ocrmypdf: records argv to $FAKE_OCR_ARGV, copies src->dest."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "ocrmypdf"
    exe.write_text(
        "#!/usr/bin/env python3\n"
        "import os, shutil, sys\n"
        "open(os.environ['FAKE_OCR_ARGV'], 'w').write(repr(sys.argv))\n"
        "shutil.copyfile(sys.argv[-2], sys.argv[-1])\n", encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_OCR_ARGV", str(tmp_path / "argv.txt"))
    return exe


@pytest.mark.skipif(os.name == "nt", reason="fake engine is a POSIX script")
def test_ocr_argv_order_and_lang(sample_pdf, tmp_path, fake_ocrmypdf):
    out = tmp_path / "ocr.pdf"
    run_ocr(sample_pdf, out, lang="afr")
    argv = eval((tmp_path / "argv.txt").read_text())
    assert argv[1:] == ["--language", "afr", str(sample_pdf), str(out)]


def test_ocr_engine_failure_is_clean_error(sample_pdf, tmp_path, monkeypatch):
    bindir = tmp_path / "bin2"
    bindir.mkdir()
    exe = bindir / "ocrmypdf"
    exe.write_text("#!/usr/bin/env python3\nimport sys; sys.exit(7)\n",
                   encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(PdfCloakError, match="ocrmypdf failed"):
        run_ocr(sample_pdf, tmp_path / "o.pdf")


def test_ocr_fake_engine_copies(sample_pdf, tmp_path, fake_ocrmypdf):
    out = tmp_path / "ocr.pdf"
    assert run_ocr(sample_pdf, out) == out and out.exists()


def test_ocr_missing_engine_is_clean_error(sample_pdf, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # nothing on PATH
    with pytest.raises(PdfCloakError, match="ocrmypdf not found"):
        run_ocr(sample_pdf, tmp_path / "o.pdf")


def test_ocr_cli(sample_pdf, tmp_path, fake_ocrmypdf):
    out = tmp_path / "ocr.pdf"
    assert main(["ocr", str(sample_pdf), "-o", str(out)]) == 0
    assert out.exists()


# --- profiles ---------------------------------------------------------------

def test_profiles_exist_and_merge():
    assert get_profile("gdpr").rules and get_profile("hipaa").threshold == 0.5
    assert get_profile("minimal").labels == []
    with pytest.raises(PdfCloakError):
        get_profile("nope")


def test_profile_sa_finds_id_without_rules(sample_pdf, capsys):
    # Built-ins alone find only the 2 emails; the sa profile adds the ID rule.
    assert main(["scan", str(sample_pdf), "--profile", "sa"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["counts_by_label"].get("sa_id") == 1
    assert report["counts_by_label"].get("email") == 2


def test_profile_in_help(capsys):
    with pytest.raises(SystemExit):
        main(["scan", "--help"])
    assert "--profile" in capsys.readouterr().out


# --- batch ------------------------------------------------------------------

def _make_clean_pdf(path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 750), "Nothing sensitive here.")
    doc.save(path)
    doc.close()


def test_batch_scan_aggregates(sample_pdf, tmp_path, capsys):
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "a.pdf").write_bytes(sample_pdf.read_bytes())
    _make_clean_pdf(folder / "b.pdf")
    (folder / "ignore.txt").write_text("not a pdf")
    rc = main(["batch", "scan", str(folder)])
    report = json.loads(capsys.readouterr().out)
    assert rc == 3  # a.pdf has matches
    assert report["total_matches"] == 2
    assert [f["rc"] for f in report["files"]] == [3, 0]


def test_batch_redact_writes_outputs(sample_pdf, tmp_path, capsys):
    folder, outdir = tmp_path / "in", tmp_path / "out"
    folder.mkdir()
    (folder / "a.pdf").write_bytes(sample_pdf.read_bytes())
    rc = main(["batch", "redact", str(folder), "-o", str(outdir),
               "--profile", "sa"])
    report = json.loads(capsys.readouterr().out)
    assert rc == 0
    out = outdir / "a.pdf"
    assert out.exists()
    with fitz.open(out) as doc:
        text = "\n".join(p.get_text() for p in doc)
    assert "jane.doe@example.com" not in text
    assert report["files"][0]["output"] == str(out)


def test_batch_bad_directory(capsys):
    assert main(["batch", "scan", "/nonexistent-dir-xyz"]) == 2


# --- i18n through the CLI -----------------------------------------------------

def test_lang_af_verdict(sample_pdf, capsys):
    assert main(["scan", str(sample_pdf), "--lang", "af"]) == 0
    assert "treffer" in capsys.readouterr().err.lower()


def test_lang_de_verify_clean(tmp_path, capsys):
    path = tmp_path / "clean.pdf"
    _make_clean_pdf(path)
    assert main(["verify", str(path), "--lang", "de"]) == 0
    assert "sauber" in capsys.readouterr().err.lower()


# --- web UI -------------------------------------------------------------------

@pytest.fixture()
def ui_server():
    from pdfcloak.web import serve
    server = serve("127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_web_ui_scan_masks_pii(ui_server, sample_pdf):
    boundary = "----testboundary"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"a.pdf\"\r\nContent-Type: application/pdf\r\n\r\n".encode()
        + sample_pdf.read_bytes()
        + f"\r\n--{boundary}\r\nContent-Disposition: form-data; "
          "name=\"action\"\r\n\r\nscan".encode()
        + f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        ui_server + "/scan", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        page = resp.read().decode()
    assert resp.status == 200
    assert "j***@e***.com" in page
    assert "jane.doe@example.com" not in page  # never raw PII in the UI


def test_web_ui_home_and_redact(ui_server, sample_pdf):
    with urllib.request.urlopen(ui_server + "/", timeout=10) as resp:
        home = resp.read().decode()
    assert 'accept="application/pdf"' in home and "<label" in home
    from pdfcloak.web import redact_bytes
    out = redact_bytes(sample_pdf.read_bytes())
    assert out[:5] == b"%PDF-"
    with fitz.open(stream=out) as doc:
        text = "\n".join(p.get_text() for p in doc)
    assert "jane.doe@example.com" not in text


def test_web_ui_refuses_remote_bind(monkeypatch):
    from pdfcloak.web import serve
    monkeypatch.delenv("PDFCLOAK_ALLOW_REMOTE_SERVE", raising=False)
    with pytest.raises(PdfCloakError):
        serve("0.0.0.0", 0)
