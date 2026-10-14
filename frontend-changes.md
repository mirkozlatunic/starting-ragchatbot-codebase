# Frontend changes: dark/light theme toggle

## `frontend/index.html`
- Inline `<head>` script sets `data-theme` on `<html>` before first paint (saved choice from `localStorage`, else OS `prefers-color-scheme`), avoiding a flash.
- Added a fixed top-right `#themeToggle` button with moon and sun SVG icons (`aria-label` updates with state; native `<button>`, so Tab/Enter/Space work).
- Bumped CSS/JS cache-busting versions to `v=11`.

## `frontend/style.css`
- New `[data-theme="light"]` block overriding the CSS variables (light backgrounds, dark slate text, darker secondary text and links for contrast, lighter borders).
- New variables for previously hardcoded colors: `--link-accent`, `--code-bg`, `--error-text`, `--success-text`; existing rules now use them. Fixed `var(--primary)` typo to `var(--primary-color)` in blockquote.
- Toggle button styles: circular, matches surface/border design, hover scale, `:focus-visible` ring; icons cross-fade and rotate. Respects `prefers-reduced-motion`.
- `.theme-transition` class smooths background/color/border changes for 0.3s during a toggle.

## `frontend/script.js`
- Click handler toggles `data-theme` between `dark` and `light`, adds `.theme-transition` temporarily, persists to `localStorage`, and updates the button's `aria-label`.
