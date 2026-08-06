# UI Architecture Notes

This project used to grow by direct UI patches inside large HTML/script files. The current cleanup moves it toward a feature-based structure.

## Main UI shell

- `templates/partials/tool_selection.html`
  - Top-level tool cards.
  - Cards should use `data-open-tool` and `data-open-panel`.
  - Avoid long inline `onclick` strings.

- `templates/partials/dicom_roi_workspace.html`
  - Main DICOM workspace shell.
  - Should contain shared tabs/panel slots.
  - Large feature panels should be included as partials, not pasted directly into this file.

- `templates/partials/script_parts/ui_registry.html`
  - Central registry for tools, panels, actions, routes, result hosts, and features.

- `templates/partials/script_parts/ui_result_renderers.html`
  - Shared result/status rendering helpers.

- `templates/partials/script_parts/dicom_panels.html`
  - Central DICOM panel controller.
  - `switchDicomPanel(panelKey)` should be the normal way to switch DICOM panels.

## ACR module panels

Panel HTML should live in:

- `templates/partials/acr_modules/module3_uniformity_panel.html`
- `templates/partials/acr_modules/module1_ct_number_panel.html`

The workspace includes them with Jinja includes.

## Adding a future ACR module

Use this pattern:

1. Create a panel partial.
2. Add one tab in `dicom_roi_workspace.html`.
3. Include the panel partial in `dicom_roi_workspace.html`.
4. Register the panel/action/route/result host in `ui_registry.html`.
5. Add backend route in `routes/module_classifier_routes.py`.
6. Add service logic under `services/`.
7. Add a top-level card only if it deserves one.

## Rules going forward

- Do not add fake-click bridges.
- Do not use `MutationObserver` for normal navigation.
- Do not patch UI by broad text replacement at runtime.
- Do not duplicate Split Modules buttons manually across scripts.
- Use shared `data-action` and `UIRegistry.registerAction()`.
- Use route keys through `UIRegistry.routeUrl()`.
- Use result hosts through `UIRegistry.registerResultHost()`.
## Tool cards

Top-level tool cards now live in:

- `templates/partials/tool_cards/matrix_card.html`
- `templates/partials/tool_cards/dicom_roi_card.html`
- `templates/partials/tool_cards/module3_uniformity_card.html`
- `templates/partials/tool_cards/module1_ct_number_card.html`
- `templates/partials/tool_cards/histogram_card.html`
- `templates/partials/tool_cards/upload_new_button.html`

`tool_selection.html` should stay a thin shell that includes these cards.

When adding a future top card, create a new card partial and include it in `tool_selection.html`. Keep routing in `data-open-tool`, `data-open-panel`, or `data-action`; do not add long inline JavaScript.
## DICOM tabs

The DICOM tab row now lives in:

- `templates/partials/dicom/dicom_tabs.html`

`dicom_roi_workspace.html` should include this file instead of containing the tab buttons directly.

When adding a future DICOM panel, update:

1. `templates/partials/dicom/dicom_tabs.html`
2. the matching panel partial/include
3. `templates/partials/script_parts/ui_registry.html`

The tab id must match the panel id:

- `tab-module2`
- `panel-module2`

The key passed to `data-open-panel` should be the shared key:

- `data-open-panel="module2"`
## DICOM core panels

The core DICOM panels now live in:

- `templates/partials/dicom/panels/info_panel.html`
- `templates/partials/dicom/panels/window_panel.html`
- `templates/partials/dicom/panels/roi_panel.html`
- `templates/partials/dicom/panels/results_panel.html`

The ACR module panels live separately under:

- `templates/partials/acr_modules/`

`dicom_roi_workspace.html` should mainly be a shell of includes. Future panel work should edit the relevant panel partial instead of pasting new sections into the workspace.
## Health check

Architecture validation lives in:

- `tools/validate_ui_architecture.py`

Run:

```powershell
py ".\tools\validate_ui_architecture.py"
```

The health check looks for:

- missing architecture files
- broken Jinja includes
- tabs without panels
- panels without tabs
- duplicate ids
- leftover inline `onclick`
- hard-coded fetch routes
- MutationObserver usage
- programmatic `.click()` usage
- JavaScript syntax errors when Node is available

Use this before and after adding new modules.

## UI design system / reusable classes

The active visual design system lives in:

- `templates/partials/design_system.html`

It is included inside the single `<style>` wrapper owned by
`templates/partials/styles.html`. Do not add another page-level style tag.

Reusable launcher classes:

- `launcher-shell` — full launcher page width and spacing.
- `launcher-hero` — unframed compact application identity/title block.
- `launcher-title` — the dominant application title.
- `load-study-zone` — primary file-loading action.
- `module-grid` — expandable grid for top-level tools/modules.
- `module-card` — reusable launcher card for a QA module.
- `module-card-active` — selected module state.

The launcher receives `study-loaded` after file selection and
`tool-workspace-active` while a tool is open. Either state hides the
start-only hero so module selection and active workspaces get the available
screen area. Reset removes both states.

Reusable action-button classes:

- `action-button` — shared height, typography, corners, and interaction.
- `action-primary` — main analysis action.
- `action-secondary` — strong secondary action such as whole-stack scanning.
- `action-utility` — back, split, and supporting controls.
- `action-danger` — clear/destructive display-only actions.
- `action-small` — compact toolbar control.

Reusable workspace and results classes:

- `workspace-surface` — incoming top-level tool surface.
- `result-dashboard` — outer result output region.
- `result-section` — summary, image, or measurement section.
- `metric-card` — compact result metric.
- `image-stage` — constrained medical-image/overlay stage.

Reusable processing and viewer classes:

- `loading-hud loading-hud-full` — result-area indeterminate analysis state.
- `loading-hud loading-hud-compact` — narrow side-panel status state.
- `loading-hud loading-hud-viewport` — compact dark DICOM-viewer loading state.
- `result-toolbar-context` / `context-action` — result actions whose visibility follows the active result context.
- `loading-ring` — CSS-only CT/radar indicator shared by all loader variants.
- `slice-info-card` — compact selected-slice metadata.

Use the shared helper rather than writing loading markup:

```js
showAnalysisLoading(host, title, subtitle, { variant: "compact" });
showAnalysisLoading(resultHost, title, subtitle, { variant: "full" });
showAnalysisLoading(viewerPlaceholder, title, subtitle, { variant: "viewport" });
```

The bottom result toolbar is updated by `setCurrentResultContext()` and
`updateResultToolbarContext()`. Renderers set the context to `roi`, `module1`,
`module3`, or `split`; panel switching then shows only actions supported by the
visible result. Module 3's processed-image toggle is available only after its
processed-image host contains rendered output.

While the DICOM stack is loading, `showAnalysisLoading(..., {
variant: "viewport" })` replaces the placeholder's normal
`viewer-empty-state` class with `viewer-loading-state`. This is intentional:
the viewport loader is centered directly over the dark viewer and must not
inherit the empty-state border, dashed target frame, or card background.
Viewport mode uses dedicated `viewport-loading` and
`viewport-loading-ring` markup rather than the generic `loading-hud` markup.
The host uses full-width/full-height flex centering; do not replace it with an
inline `display: block`.

The viewport variant also owns the reusable `viewport-scan-ring` and
`viewport-scanline` elements. They fill the dark image stage with a restrained
CT/radar preparation cue while `viewport-loading-core` keeps the spinner and
copy centered. The idle `viewer-empty-state` uses the same full-stage visual
language without animation.

The center viewer contains both the placeholder and the persistent canvas
wrapper. Therefore `viewer-empty-state` and `viewer-loading-state` are
absolute overlays (`inset: 0`) anchored to the positioned
`dicom-center-viewer`; they must not participate as width-100% siblings in
the viewer's flex row, or their visual center will be shifted into the left
half of the viewport.

All loader variants use only a spinner, title, and subtitle. Compact mode is
an unboxed inline status for narrow inspectors; full mode is a small centered
result-area block; viewport mode is unboxed on the dark viewer. Do not add
phase labels, decorative grids, engine-state banners, or fake percentages.

Future Module 4 UI should reuse these classes instead of creating another
one-off launcher or result-card design. New tool cards should retain
`data-open-tool` / `data-open-panel`, add `module-card`, and remain wired
through UIRegistry.

## ACR CT Module 4 UI shell

Module 4 currently provides candidate-slice location, selected-slice internal
block detection, an overlay, and a structured result display. The main Split
Modules classifier is the single source of truth: for
plausible Module 4 slices it combines the existing high-contrast score with
raw-pixel outside four-BB marker evidence and stores the result in
`scores["MODULE_4_HIGH_CONTRAST"]` plus `module4Evidence`. The route reuses that
cached classification when available, or runs the same shared classifier when
needed. No line-pair detection, resolution measurement, or pass/fail logic is
implemented.

Whole-stack Module 4 analysis selects the highest-scoring record already
predicted as Module 4 by Split Modules, then runs internal square/line-pair
block detection on only that raw slice. Selected-slice mode analyzes only the
current slice. The service draws labeled block bounding boxes but does not
calculate lp/cm, modulation, limiting resolution, visibility, or pass/fail.

The raw-pixel helper targets four compact perimeter dots in cardinal
top/right/bottom/left positions. For performance, the shared classifier limits
this analysis to at most 12 predicted, neighboring, or top base-score
candidates and scales the longest image dimension to at most 384 pixels.
Classifier metadata records evaluated/skipped counts, candidate indices, and
outer-BB runtime.

- Launcher card: `templates/partials/tool_cards/module4_high_contrast_card.html`
- DICOM panel: `templates/partials/acr_modules/module4_high_contrast_panel.html`
- Frontend controller: `templates/partials/script_parts/dicom_module4.html`
- Panel/tab key: `module4` (`tab-module4` → `panel-module4`)
- Actions: `module4_selected`, `module4_stack`, `clear_module4_results`
- Shared action: `split_modules`
- Status host: `acrModule4Status`
- Result host: `acrModule4ResultArea`
- Route key: `acr_module4_high_contrast`
- Endpoint: `POST /dicom-module4-high-contrast-analysis`

The endpoint returns `implemented: "partial"`, `measurement_status:
"not_implemented"`, the selected classifier candidate, candidate rows, and
classifier-owned `module4Evidence`. It adapts shared classifier records and
must not load pixels or run a second Module 4 selector. Stack mode ranks the
shared classifier's Module 4 predictions. Selected mode reports the shared
classifier evidence for the current slice and marks the result for review. It
must not return fabricated lp/cm, limiting
resolution, modulation, visibility, or pass/fail values. The former Histogram
files remain unused and are not included in the launcher or feature manifest.

The older "SHARP MEDICAL QA POLISH" and "LIGHT MEDICAL CONTROL CONSOLE"
override layers were removed from `styles.html`. Generic base rules remain
for legacy components, while the active launcher and HUD presentation is
owned by the reusable class-based design system.
## Registry split

The registry is now split into:

- `templates/partials/script_parts/ui_registry.html`
  - Core registry API only.
  - Owns registration functions, route helpers, result-host helpers, delegated click handling, and validation.

- `templates/partials/script_parts/ui_feature_manifest.html`
  - Feature declarations.
  - Registers Matrix, DICOM Viewer, ACR Modules, Histogram, routes, actions, panels, result hosts, and template paths.

Keep the core registry small. Add future tools/modules to the feature manifest or a future feature-specific manifest file.
## Script groups

`templates/partials/scripts.html` is now a thin shell.

Script loading is grouped into:

- `templates/partials/script_groups/core_scripts.html`
- `templates/partials/script_groups/dicom_viewer_scripts.html`
- `templates/partials/script_groups/acr_module_scripts.html`
- `templates/partials/script_groups/matrix_scripts.html`

Keep this order:

1. Core scripts
2. DICOM viewer scripts
3. ACR module scripts
4. Matrix scripts

When adding a future feature, create a feature script group instead of adding many includes directly into `scripts.html`.
## Phase 19 health-fix pass

Phase 19 reduced remaining hard-coded frontend route calls by routing them through:

```js
UIRegistry.routeUrl(...)
```

The goal is not visual change. The goal is to keep endpoint names centralized so future backend route changes do not require searching through feature scripts.

## Phase 20B duplicate button repair

Phase 20B fixed the failed Phase 20 duplicate-button cleanup. It removes repeated static ACR action buttons and adds a runtime guard so the Split Modules helper does not clone a second Split Modules button when a static data-action button already exists.
