# Add a New UI Feature

Use this checklist when adding a new feature so the UI does not become patch-based again.

## 1. Add the backend

Create a service under:

```text
services/
```

Create or extend a route under:

```text
routes/
```

Use a clear endpoint name. Example:

```text
/dicom-module2-low-contrast-analysis
```

## 2. Register the route

Add the route to:

```text
templates/partials/script_parts/ui_registry.html
```

Use a route key:

```js
registry.registerRoute({
    key: "acr_module2_low_contrast",
    method: "POST",
    url: "/dicom-module2-low-contrast-analysis"
});
```

Frontend scripts should use:

```js
UIRegistry.routeUrl("acr_module2_low_contrast", "/dicom-module2-low-contrast-analysis")
```

Do not hard-code `fetch("/...")` directly.

## 3. Add the panel partial

Create a panel partial:

```text
templates/partials/acr_modules/module2_low_contrast_panel.html
```

Use matching tab/panel names:

```html
<button class="dicom-tab" data-open-panel="module2" id="tab-module2">Module 2</button>

<div class="dicom-panel-section" id="panel-module2">
    ...
</div>
```

## 4. Register the panel and actions

In `ui_registry.html`, register:

```js
registry.registerPanel({
    key: "module2",
    tool: "dicom_roi",
    tabLabel: "Module 2",
    template: "partials/acr_modules/module2_low_contrast_panel.html"
});

registry.registerAction({
    key: "module2_analysis",
    label: "Analyze Module 2",
    functionName: "runAcrModule2Analysis"
});
```

Buttons should use:

```html
<button data-action="module2_analysis">Analyze Module 2</button>
```

Do not use inline `onclick`.

## 5. Add result hosts

Every output area should have a registered result host:

```js
registry.registerResultHost({
    key: "acr_module2_results",
    elementId: "acrModule2Results"
});
```

Then use the shared renderer layer:

```js
UIRegistry.renderResult("status", {
    host: "acr_module2_results",
    message: "Running Module 2 analysis..."
});
```

## 6. Run the health check

Run:

```powershell
py ".\tools\validate_ui_architecture.py"
```

Fix warnings before adding visual polish.
