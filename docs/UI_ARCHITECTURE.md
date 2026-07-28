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
