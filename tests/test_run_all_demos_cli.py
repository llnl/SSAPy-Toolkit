from pathlib import Path
import tomllib

import ssapy_toolkit.run_all_demos as gallery_cli
from ssapy_toolkit.demo_gallery import discover_demo_files


def test_find_default_demos_dir_finds_packaged_or_repo_demos():
    demos_dir = gallery_cli.find_default_demos_dir()

    assert demos_dir is not None
    assert demos_dir.name == "demos"
    assert (demos_dir / "demo_gifify.py").exists()


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
    assert (captured["demos_dir"] / "demo_gifify.py").exists()
    assert captured["output_root"] == tmp_path / "gallery"
    assert captured["clean"] is True


def test_gallery_discovery_skips_test_only_demos():
    names = {path.name for path in discover_demo_files(Path("demos"))}

    assert "demo_parsing_3le.py" not in names
    assert "demo_transfer_vburn.py" not in names
    assert "demo_first_user_workflow.py" in names
    assert "demo_data_package_access.py" in names


def test_pyproject_installs_demo_package_and_cli_script():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    includes = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "demos*" in includes
    assert pyproject["project"]["scripts"]["ssapy-demo-gallery"] == "ssapy_toolkit.run_all_demos:main"
