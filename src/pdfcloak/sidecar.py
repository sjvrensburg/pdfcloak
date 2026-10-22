"""Option 2 distribution: spawn the vendored ClipCloak daemon as a sidecar.

The CLI generates a token, picks a free localhost port, waits for the child
to announce it is listening, verify its authenticated /identity,
and terminates the child on exit. Model weights (~412 MB int8) are *not* bundled:
they are fetched from Hugging Face on first run unless PII_MODELS_DIR is set.
"""
from __future__ import annotations

import atexit
import collections
import hashlib
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import PdfCloakError
from .detectors import ClipCloakDetector, DEFAULT_LABELS

ENV_BIN = "PDFCLOAK_SIDECAR"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _binary_name() -> str:
    return "clipcloak-server.exe" if os.name == "nt" else "clipcloak-server"


def user_data_dir() -> Path:
    """Per-user location for the built sidecar, used by pipx/uvx installs."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "pdfcloak"


def user_binary_path() -> Path:
    return user_data_dir() / "bin" / _binary_name()


def bundled_sources_dir() -> Path | None:
    """Rust sidecar sources: bundled in the wheel, or the repo's vendor/ tree."""
    installed = Path(__file__).resolve().parent / "_sidecar"
    if (installed / "Cargo.toml").exists():
        return installed
    repo = _repo_root() / "vendor" / "clipcloak-server"
    if (repo / "Cargo.toml").exists():
        return repo
    return None


def build_sidecar(*, force: bool = False) -> Path:
    """Compile the vendored sidecar and install it under the user data dir.

    This is what makes Option 2 work for pipx/uvx installs: the wheel carries
    the Rust sources (~300 KB), the user runs this once with cargo installed.
    """
    import shutil
    dest = user_binary_path()
    if dest.is_file() and not force:
        return dest
    cargo = shutil.which("cargo")
    if cargo is None:
        raise PdfCloakError(
            "cargo not found; install Rust from https://rustup.rs, "
            "then re-run `pdfcloak sidecar build`")
    sources = bundled_sources_dir()
    if sources is None:
        raise PdfCloakError("sidecar sources not bundled in this installation")
    build_dir = user_data_dir() / "sidecar-src"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    shutil.copytree(sources, build_dir,
                    ignore=shutil.ignore_patterns("target", "__pycache__"))
    print("building sidecar (cargo, may take a few minutes) ...", file=sys.stderr)
    # Own process group so a timeout kills cargo *and* its rustc children,
    # and partial output survives for diagnosis.
    proc = subprocess.Popen([cargo, "build", "--release", "--locked"], cwd=build_dir,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=1800)
    except subprocess.TimeoutExpired:
        import signal
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        stdout, stderr = proc.communicate()
        raise PdfCloakError(
            "cargo build timed out after 1800s; last output:\n"
            f"{(stdout or '')[-1000:]}\n{(stderr or '')[-1000:]}") from None
    if proc.returncode != 0:
        raise PdfCloakError(
            f"cargo build failed:\n{stdout[-2000:]}\n{stderr[-2000:]}")
    built = build_dir / "target" / "release" / _binary_name()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, dest)
    dest.chmod(0o755)
    return dest


def find_sidecar_binary(explicit: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get(ENV_BIN):
        candidates.append(Path(os.environ[ENV_BIN]))
    root = _repo_root()
    name = _binary_name()
    candidates += [
        user_binary_path(),
        root / "vendor" / "clipcloak-server" / "target" / "release" / name,
        root / "bin" / name,
        Path(sys.prefix) / "bin" / name,
    ]
    for c in candidates:
        if c.is_file() and (os.name == "nt" or os.access(c, os.X_OK)):
            return c
    import shutil
    found = shutil.which(name)
    return Path(found) if found else None


def find_onnxruntime_lib(explicit: str | None = None) -> str | None:
    """ClipCloak's `ort` is load-dynamic: it needs libonnxruntime.so at runtime."""
    if explicit or os.environ.get("ORT_DYLIB_PATH"):
        return explicit or os.environ.get("ORT_DYLIB_PATH")
    patterns = (["libonnxruntime*.dll"] if os.name == "nt"
                else ["libonnxruntime*.dylib"] if sys.platform == "darwin"
                else ["libonnxruntime.so*"])
    try:
        import onnxruntime
        capi = Path(onnxruntime.__file__).parent / "capi"
        for pat in patterns:
            hits = sorted(capi.glob(pat))
            if hits:
                return str(hits[0])
    except ImportError:
        pass
    cache = Path.home() / ".cache"
    if cache.exists():
        for pat in patterns:
            hits = sorted(cache.rglob(pat))[:1]
            if hits:
                return str(hits[0])
    return None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Sidecar:
    """Context manager owning a clipcloak-server child process."""

    def __init__(self, *, binary: str | Path | None = None,
                 models_dir: str | Path | None = None,
                 labels: list[str] | None = None, threshold: float = 0.55,
                 startup_timeout: float = 600.0, log_file: str | Path | None = None):
        self.binary = find_sidecar_binary(binary)
        self.models_dir = models_dir
        self.labels = labels or DEFAULT_LABELS
        self.threshold = threshold
        self.startup_timeout = startup_timeout
        self.log_file = log_file
        self.token = secrets.token_urlsafe(24)
        self.port = _free_port()
        self.proc: subprocess.Popen | None = None
        self.detector = ClipCloakDetector(f"http://127.0.0.1:{self.port}", self.token)

    def __enter__(self) -> ClipCloakDetector:
        expected_hash = os.environ.get("PDFCLOAK_SIDECAR_SHA256")
        if expected_hash and self.binary is not None:
            actual = hashlib.sha256(Path(self.binary).read_bytes()).hexdigest()
            if actual.lower() != expected_hash.lower():
                raise PdfCloakError(
                    f"sidecar binary hash mismatch: got {actual}, "
                    f"expected {expected_hash} (PDFCLOAK_SIDECAR_SHA256)")
        if self.binary is None:
            raise PdfCloakError(
                "ClipCloak sidecar binary not found. Build it once with "
                "`pdfcloak sidecar build` (needs cargo), or pass --sidecar-bin / "
                f"set {ENV_BIN}, or connect to a running daemon with --sidecar-url.")
        env = dict(os.environ)
        env.update({
            "PII_PORT": str(self.port),
            "PII_TOKEN": self.token,
            "PII_LABELS": ",".join(self.labels),
            "PII_THRESHOLD": str(self.threshold),
            "PII_ALLOWED_ORIGINS": "",
        })
        if self.models_dir:
            env["PII_MODELS_DIR"] = str(self.models_dir)
        ort_lib = find_onnxruntime_lib()
        if ort_lib:
            env["ORT_DYLIB_PATH"] = ort_lib
        self._log_handle = open(self.log_file, "ab") if self.log_file else None
        self._output: collections.deque[str] = collections.deque(maxlen=200)
        self._listening = threading.Event()
        try:
            self.proc = subprocess.Popen(
                [str(self.binary)], env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self._reader = threading.Thread(target=self._pump_output, daemon=True)
            self._reader.start()
            deadline = time.time() + self.startup_timeout
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    raise PdfCloakError(
                        f"sidecar exited early (code {self.proc.returncode}): "
                        + " | ".join(list(self._output)[-3:]))
                # Only trust the port once *our child* says it is listening,
                # and only after its authenticated identity check passes —
                # a foreign process that grabbed the port cannot satisfy both.
                if self._listening.is_set() and self.detector.identity(timeout=2.0) \
                        and self.proc.poll() is None:
                    atexit.register(self.stop)
                    return self.detector
                time.sleep(0.25)
            raise PdfCloakError(
                f"sidecar did not become healthy within {self.startup_timeout:.0f}s")
        except BaseException:
            self.stop()
            raise

    def _pump_output(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for raw in self.proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip()
            self._output.append(line)
            if "Listening on http" in line:
                self._listening.set()
            if self._log_handle:
                try:
                    self._log_handle.write(raw)
                    self._log_handle.flush()
                except OSError:
                    pass
            elif line:
                # Surface startup/download progress: a silent 600s wait for
                # a 412 MB model download otherwise looks like a hang.
                print(f"sidecar: {line}", file=sys.stderr)

    def stop(self) -> None:
        atexit.unregister(self.stop)
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        reader = getattr(self, "_reader", None)
        if reader is not None and reader.is_alive():
            reader.join(timeout=2)
        handle = getattr(self, "_log_handle", None)
        if handle is not None:
            handle.close()
            self._log_handle = None

    def __exit__(self, *exc) -> None:
        self.stop()


def model_cache_present() -> bool:
    """Best-effort check that GLiNER2 weights are already downloaded."""
    import os as _os
    models_dir = _os.environ.get("PII_MODELS_DIR")
    if models_dir:
        return Path(models_dir).is_dir()
    hub = Path(_os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"
    try:
        return hub.is_dir() and any("gliner" in entry.name.lower() for entry in hub.iterdir())
    except OSError:
        return False


def doctor_checks() -> list[tuple[str, bool, str]]:
    """(check, ok, exact fix) rows for `pdfcloak sidecar doctor`."""
    import shutil
    rows: list[tuple[str, bool, str]] = []
    binary = find_sidecar_binary()
    rows.append(("sidecar binary", binary is not None,
                 "run: pdfcloak sidecar build"))
    rows.append(("cargo (only needed to build)", shutil.which("cargo") is not None,
                 "install Rust from https://rustup.rs"))
    rows.append(("onnxruntime library", find_onnxruntime_lib() is not None,
                 "uv tool install --with onnxruntime pdfcloak, or set ORT_DYLIB_PATH"))
    rows.append(("model weights cached", model_cache_present(),
                 "first AI run downloads ~412 MB from Hugging Face (needs network)"))
    return rows
