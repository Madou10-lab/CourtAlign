import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_method_packages_and_entry_points_are_explicit():
    required = [
        ROOT / "src/courtalign_2s",
        ROOT / "src/courtalign_e2e",
        ROOT / "src/courtalign_common",
        ROOT / "scripts/courtalign_2s/train.py",
        ROOT / "scripts/courtalign_2s/evaluate_segmentation.py",
        ROOT / "scripts/courtalign_e2e/train.py",
        ROOT / "scripts/courtalign_e2e/evaluate.py",
        ROOT / "assets/courtalign_2s/tennis/templates/tennis_court_reference.png",
    ]
    assert all(path.exists() for path in required)

    package_names = {
        path.name
        for path in (ROOT / "src").iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert package_names == {"courtalign", "courtalign_2s", "courtalign_e2e", "courtalign_common"}

    script_groups = {
        path.name
        for path in (ROOT / "scripts").iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert script_groups == {"courtalign_2s", "courtalign_e2e"}


def test_split_manifests_use_portable_paths():
    expected_columns = [
        "dataset_id",
        "sport",
        "task",
        "split",
        "image_id",
        "image_path",
        "mask_path",
        "image_sha256",
        "mask_sha256",
        "width",
        "height",
        "n_classes",
        "label_schema",
    ]
    for manifest in ("tennis_fullcourt.csv", "badminton_zones.csv"):
        with (ROOT / "data/splits" / manifest).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == expected_columns
            for row in reader:
                assert not Path(row["image_path"]).is_absolute()
                assert not Path(row["mask_path"]).is_absolute()


def test_public_tree_has_no_development_identifiers():
    forbidden_path_tokens = {
        "method0",
        "method_0",
        "method1",
        "method_1",
        "e2e-v2",
        "e2e_v2",
        "faithful-v2",
        "faithful_v2",
        "codex",
        "claude",
    }
    forbidden_text = {
        "/home/ahmed",
        "codex_chapter_folder",
        "claude_chapter_folder",
        "chapter",
        "codex",
        "claude",
        "courtalign-e2e-v2",
        "courtalign_e2e_v2",
        "from courtreg",
        "faithful-v2",
        "faithful_v2",
        "import courtreg",
        "method0",
        "method_0",
        "method1",
        "method_1",
        "official_benchmark_gt_candidate",
        "raw_courtalign_2s_candidate_status",
        "segminton_on_tennis",
    }
    text_suffixes = {
        ".cff",
        ".csv",
        ".ini",
        ".json",
        ".md",
        ".py",
        ".tex",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }

    offenders = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        lowered_path = relative.as_posix().lower()
        if any(token in lowered_path for token in forbidden_path_tokens):
            offenders.append(lowered_path)
        if path.is_file() and path.suffix.lower() in text_suffixes:
            lowered_text = path.read_text(encoding="utf-8").lower()
            for token in forbidden_text:
                if token in lowered_text:
                    offenders.append(f"{lowered_path}: {token}")

    assert not (ROOT / "data/benchmark_gt/official/verification").exists()
    assert not offenders, "Development identifiers remain in the public tree:\n" + "\n".join(offenders)
