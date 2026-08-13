from pathlib import Path
from types import SimpleNamespace

import pytest

import ssapy_toolkit.run_all_demos as gallery_cli
from ssapy_toolkit.demo_gallery import discover_demo_files


def test_find_default_demos_dir_finds_packaged_or_repo_demos():
    demos_dir = gallery_cli.find_default_demos_dir()

    assert demos_dir is not None
    assert demos_dir.name == "demos"
    assert any(demos_dir.rglob("demo_gifify.py"))


def test_gallery_cli_runs_from_outside_repo(tmp_path, monkeypatch):
    captured = {}

    def fake_run_all_demos(*, demos_dir, output_root, clean):
        captured["demos_dir"] = Path(demos_dir)
        captured["output_root"] = Path(output_root)
        captured["clean"] = clean
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "index.html").write_text("<html></html>", encoding="utf-8")
        return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gallery_cli, "run_all_demos", fake_run_all_demos)

    status = gallery_cli.main(["--output", str(tmp_path / "gallery")])

    assert status == 0
    assert captured["demos_dir"].name == "demos"
    assert any(captured["demos_dir"].rglob("demo_gifify.py"))
    assert captured["output_root"] == tmp_path / "gallery"
    assert captured["clean"] is True


def test_gallery_cli_distribution_fallback_errors_and_open(tmp_path, monkeypatch, capsys):
    calls = iter([False, False, True])
    monkeypatch.setattr(gallery_cli, "_looks_like_demos_dir", lambda path: next(calls))

    class FakeDist:
        files = [Path("demos/__init__.py")]

        def locate_file(self, file):
            return tmp_path / "site" / file

    monkeypatch.setattr(gallery_cli, "distribution", lambda name: FakeDist())
    assert gallery_cli.find_default_demos_dir() == (tmp_path / "site" / "demos").resolve()

    monkeypatch.setattr(gallery_cli, "_looks_like_demos_dir", lambda path: False)
    monkeypatch.setattr(gallery_cli, "distribution", lambda name: (_ for _ in ()).throw(gallery_cli.PackageNotFoundError(name)))
    assert gallery_cli.find_default_demos_dir() is None
    with pytest.raises(SystemExit, match="Demo directory not found"):
        gallery_cli.main(["--output", str(tmp_path / "out")])

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    with pytest.raises(SystemExit, match="Demo directory not found"):
        gallery_cli.main(["--demos-dir", str(bad_dir), "--output", str(tmp_path / "out")])

    demo_dir = tmp_path / "demos"
    demo_dir.mkdir()
    (demo_dir / "demo_one.py").write_text("", encoding="utf-8")
    output = tmp_path / "gallery"

    def fake_run_all_demos(*, demos_dir, output_root, clean):
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "index.html").write_text("<html></html>", encoding="utf-8")
        return [SimpleNamespace(status="success"), SimpleNamespace(status="failed")]

    monkeypatch.setattr(gallery_cli, "_looks_like_demos_dir", lambda path: path == demo_dir)
    monkeypatch.setattr(gallery_cli, "run_all_demos", fake_run_all_demos)
    monkeypatch.setattr(gallery_cli.webbrowser, "open", lambda uri: (_ for _ in ()).throw(RuntimeError("no browser")))
    assert gallery_cli.main(["--demos-dir", str(demo_dir), "--output", str(output), "--open", "--no-clean"]) == 1
    out = capsys.readouterr().out
    assert "success: 1" in out
    assert "failed : 1" in out


def test_gallery_discovery_skips_test_only_demos():
    root = Path("demos")
    names = {path.relative_to(root).as_posix() for path in discover_demo_files(root)}

    assert "getting_started/demo_parsing_3le.py" not in names
    assert "demo_transfer_vburn.py" not in names
    assert "demo_transfer_rendezvous.py" not in names
    assert "demo_transfer_ssapy.py" not in names
    assert "orbital_mechanics/demo_orbital_maneuvers.py" in names
    assert "getting_started/demo_first_user_workflow.py" in names
    assert "getting_started/demo_data_package_access.py" in names


def test_pyproject_installs_demo_package_and_cli_script():
    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'include = ["ssapy_toolkit*", "demos*"]' in pyproject_text
    assert 'ssapy-demo-gallery = "ssapy_toolkit.run_all_demos:main"' in pyproject_text
