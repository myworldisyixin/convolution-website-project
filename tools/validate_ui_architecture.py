from __future__ import annotations

from pathlib import Path
import re


REQUIRED_FILES = [
    "templates/partials/script_parts/ui_registry.html",
    "templates/partials/script_parts/ui_feature_manifest.html",
    "templates/partials/script_parts/features/matrix_manifest.html",
    "templates/partials/script_parts/features/dicom_viewer_manifest.html",
    "templates/partials/script_parts/features/acr_modules_manifest.html",
    "templates/partials/script_parts/features/histogram_manifest.html",
    "templates/partials/script_groups/core_scripts.html",
    "templates/partials/script_groups/dicom_viewer_scripts.html",
    "templates/partials/script_groups/acr_module_scripts.html",
    "templates/partials/script_groups/matrix_scripts.html",
    "templates/partials/script_parts/ui_result_renderers.html",
    "templates/partials/script_parts/dicom_panels.html",
    "templates/partials/dicom/dicom_tabs.html",
    "templates/partials/tool_cards/matrix_card.html",
    "templates/partials/tool_cards/dicom_roi_card.html",
    "templates/partials/tool_cards/module3_uniformity_card.html",
    "templates/partials/tool_cards/module1_ct_number_card.html",
    "templates/partials/tool_cards/histogram_card.html",
    "templates/partials/acr_modules/module3_uniformity_panel.html",
    "templates/partials/acr_modules/module1_ct_number_panel.html",
    "templates/partials/dicom/panels/info_panel.html",
    "templates/partials/dicom/panels/window_panel.html",
    "templates/partials/dicom/panels/roi_panel.html",
    "templates/partials/dicom/panels/results_panel.html",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def scan_html_files(root: Path) -> list[Path]:
    templates = root / "templates"
    if not templates.exists():
        return []
    return sorted(path for path in templates.rglob("*.html") if "_safe_backups" not in path.parts)


def count_pattern(files: list[Path], pattern: str, flags: int = re.IGNORECASE) -> int:
    total = 0
    rx = re.compile(pattern, flags)
    for path in files:
        total += len(rx.findall(read(path)))
    return total


def collect_ids(files: list[Path]) -> dict[str, list[str]]:
    ids: dict[str, list[str]] = {}
    rx = re.compile(r"\bid\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
    for path in files:
        for match in rx.finditer(read(path)):
            ids.setdefault(match.group(1), []).append(str(path))
    return ids


def collect_includes(files: list[Path]) -> dict[str, list[str]]:
    includes: dict[str, list[str]] = {}
    rx = re.compile(r"{%\s*include\s+['\"]([^'\"]+)['\"]\s*%}")
    for path in files:
        for match in rx.finditer(read(path)):
            includes.setdefault(match.group(1), []).append(str(path))
    return includes


def check_js_syntax(root: Path, files: list[Path]) -> list[tuple[str, str]]:
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    if not node:
        return [("WARN", "Node was not found; JS syntax checks skipped.")]

    results: list[tuple[str, str]] = []

    for path in files:
        text = read(path)
        text = re.sub(
            r"{%\s*include\s+['\"][^'\"]+['\"]\s*%}",
            "",
            text,
        )

        if "<script" in text.lower():
            scripts = re.findall(r"<script\b[^>]*>([\s\S]*?)</script>", text, flags=re.IGNORECASE)
            if scripts:
                text = "\n\n".join(scripts)

        if not text.strip():
            continue

        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)

        try:
            proc = subprocess.run([node, "--check", str(tmp_path)], cwd=str(root), text=True, capture_output=True)
            if proc.returncode != 0:
                results.append(("FAIL", f"JS syntax failed: {path.relative_to(root)}\n{proc.stderr.strip()}"))
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass

    return results


def main() -> int:
    root = Path.cwd()

    if not (root / "app.py").exists():
        print("ERROR: run from the project root that contains app.py")
        return 2

    html_files = scan_html_files(root)
    script_dir = root / "templates" / "partials" / "script_parts"
    script_files = sorted(script_dir.rglob("*.html")) if script_dir.exists() else []

    messages: list[tuple[str, str]] = []

    for rel in REQUIRED_FILES:
        if not (root / rel).exists():
            messages.append(("FAIL", f"Missing required architecture file: {rel}"))

    includes = collect_includes(html_files)
    for include_path, sources in sorted(includes.items()):
        target = root / "templates" / include_path
        if not target.exists():
            messages.append(("FAIL", f"Broken Jinja include: {include_path} used in {len(sources)} file(s)"))

    ids = collect_ids(html_files)
    tab_keys = {key.replace("tab-", "") for key in ids if key.startswith("tab-")}
    panel_keys = {key.replace("panel-", "") for key in ids if key.startswith("panel-")}

    tabs_without_panels = sorted(tab_keys - panel_keys)
    panels_without_tabs = sorted(key for key in panel_keys - tab_keys if key not in {"matrix"})

    if tabs_without_panels:
        messages.append(("FAIL", "Tabs without panels: " + ", ".join(tabs_without_panels)))
    if panels_without_tabs:
        messages.append(("WARN", "Panels without tabs: " + ", ".join(panels_without_tabs)))

    duplicate_ids = {
        key: locations
        for key, locations in ids.items()
        if len(locations) > 1 and not key.startswith("preset-")
    }
    if duplicate_ids:
        preview = ", ".join(sorted(duplicate_ids.keys())[:12])
        messages.append(("WARN", f"Duplicate static ids found: {preview}"))

    tool_card_onclick = count_pattern(html_files, r'class\s*=\s*["\'][^"\']*\btool-card\b[^"\']*["\'][^>]*\bonclick\s*=')
    tab_onclick = count_pattern(html_files, r'id\s*=\s*["\']tab-[^"\']+["\'][^>]*\bonclick\s*=')
    raw_fetch = count_pattern(script_files, r"fetch\(\s*['\"]/(?:process|dicom-[^'\"]+)['\"]")
    route_url = count_pattern(script_files, r"UIRegistry\.routeUrl\(")
    data_actions = count_pattern(html_files, r"\bdata-action\s*=")
    data_open_tool = count_pattern(html_files, r"\bdata-open-tool\s*=")
    data_open_panel = count_pattern(html_files, r"\bdata-open-panel\s*=")
    mutation_observer = count_pattern(script_files, r"\bMutationObserver\b")
    stop_propagation = count_pattern(script_files, r"\bstopPropagation\s*\(")
    programmatic_click = count_pattern(script_files, r"\.click\s*\(")

    if tool_card_onclick:
        messages.append(("WARN", f"Tool-card inline onclick count is {tool_card_onclick}."))
    if tab_onclick:
        messages.append(("WARN", f"DICOM tab inline onclick count is {tab_onclick}."))
    if raw_fetch:
        messages.append(("WARN", f"Raw hard-coded fetch route count is {raw_fetch}."))
    if mutation_observer:
        messages.append(("WARN", f"MutationObserver count is {mutation_observer}."))
    if programmatic_click:
        messages.append(("WARN", f"Programmatic .click() count is {programmatic_click}."))

    messages.extend(check_js_syntax(root, script_files))

    print()
    print("UI ARCHITECTURE HEALTH CHECK")
    print("=" * 32)
    print()
    print("Files scanned:", len(html_files))
    print("Script files scanned:", len(script_files))
    print()
    print("Architecture counts:")
    print("  data-action:", data_actions)
    print("  data-open-tool:", data_open_tool)
    print("  data-open-panel:", data_open_panel)
    print("  UIRegistry.routeUrl calls:", route_url)
    print("  raw fetch routes:", raw_fetch)
    print("  tool-card inline onclick:", tool_card_onclick)
    print("  tab inline onclick:", tab_onclick)
    print("  MutationObserver:", mutation_observer)
    print("  stopPropagation:", stop_propagation)
    print("  programmatic .click():", programmatic_click)
    print()
    print("DICOM tabs:", sorted(tab_keys))
    print("DICOM panels:", sorted(panel_keys))
    print()

    fail_count = sum(1 for level, _ in messages if level == "FAIL")
    warn_count = sum(1 for level, _ in messages if level == "WARN")

    if messages:
        print("Messages:")
        for level, message in messages:
            print(f"  [{level}] {message}")
    else:
        print("Messages:")
        print("  [PASS] No architecture warnings found.")

    print()
    print("Summary:")
    print("  FAIL:", fail_count)
    print("  WARN:", warn_count)

    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
