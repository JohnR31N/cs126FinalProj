"""Build and verify the model-only GitHub Release asset without data or papers."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = sorted((ROOT / "results").rglob("*.pt"))
    if not paths:
        raise SystemExit("No trained checkpoints found; download the release or train first.")
    output = ROOT / "artifacts/checkpoints-v1.zip"
    output.parent.mkdir(exist_ok=True)
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in paths:
            archive.write(p, p.relative_to(ROOT).as_posix())
        archive.writestr("checkpoint_manifest.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for name, expected in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(json.dumps(dict(file=output.name, checkpoints=len(paths), bytes=output.stat().st_size,
                         sha256=digest, verified=True), indent=2))


if __name__ == "__main__":
    main()
