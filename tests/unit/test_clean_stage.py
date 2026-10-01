"""Unit tests for generated-output cleanup safeguards."""

from pathlib import Path

import pytest

from fashion_classifier.clean_stage import (
    clean_generated_outputs,
    remove_python_caches,
)


@pytest.mark.unit
def test_cleanup_refuses_project_root_and_protected_fixture(tmp_path: Path) -> None:
    """Prevent configured output paths from deleting source or fixtures."""
    project_root = tmp_path / "project"
    fixture_dir = project_root / "tests" / "fixtures"
    project_root.mkdir(parents=True)
    fixture_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="project root"):
        clean_generated_outputs(project_root, tmp_path / "artifacts", project_root, ())
    with pytest.raises(ValueError, match="protected path"):
        clean_generated_outputs(
            fixture_dir,
            tmp_path / "artifacts",
            project_root,
            (fixture_dir,),
        )


@pytest.mark.unit
def test_cleanup_removes_only_configured_output_directories(tmp_path: Path) -> None:
    """Remove processed/artifact outputs and leave unrelated files intact."""
    project_root = tmp_path / "project"
    processed_dir = project_root / "data" / "processed"
    artifacts_dir = project_root / "artifacts"
    raw_file = project_root / "styles.csv"
    processed_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    raw_file.write_text("raw\n", encoding="utf-8")

    removed = clean_generated_outputs(
        processed_dir,
        artifacts_dir,
        project_root,
        (raw_file, project_root / "tests" / "fixtures"),
    )

    assert set(removed) == {processed_dir, artifacts_dir}
    assert not processed_dir.exists()
    assert not artifacts_dir.exists()
    assert raw_file.is_file()


@pytest.mark.unit
def test_cleanup_removes_python_caches_but_skips_git(tmp_path: Path) -> None:
    """Remove bytecode and pytest caches without walking into Git metadata."""
    project_root = tmp_path / "project"
    bytecode = project_root / "package" / "__pycache__"
    pytest_cache = project_root / ".pytest_cache"
    git_cache = project_root / ".git" / "__pycache__"
    bytecode.mkdir(parents=True)
    pytest_cache.mkdir(parents=True)
    git_cache.mkdir(parents=True)

    removed = remove_python_caches(project_root)

    assert set(removed) == {bytecode, pytest_cache}
    assert not bytecode.exists()
    assert not pytest_cache.exists()
    assert git_cache.is_dir()
