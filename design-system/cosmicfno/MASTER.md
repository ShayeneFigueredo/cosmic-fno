# CosmicFNO — Design System Master File

> **LOGIC:** When building or refining a specific page or view, first check `design-system/cosmicfno/pages/[page-name].md`.
> If that file exists, its rules **override** this Master file.
> If not, strictly follow the rules and tokens defined below.

---

**Project:** CosmicFNO — Spectral Gravitational-Wave Observatory (PT / EN / ES)
**Author:** Shayene Figueredo (BSI / UFU)
**Category:** Data-Dense Scientific Observatory & SciML Analytics Dashboard
**Governing Skills:** `ui-ux-pro-max`, `frontend-design` (Anthropic), `antigravity-design-expert`

---

## 1. Global Design Tokens

### Color Palette (Cosmic Purple & Weightless Light Observatory)

| Role | Hex | CSS Variable | Usage & Contrast Notes |
|------|-----|--------------|------------------------|
| Sidebar Background | `#11091f` | `--color-sidebar-bg` | Deep cosmic obsidian-violet rail (`248px` fixed) |
| Sidebar Border | `#211339` | `--color-sidebar-border` | Subtle structural divider (`1px solid`) |
| Primary Brand / Active Nav | `#6d28d9` | `--color-primary` | Active navigation pill, primary accents, spectral bars |
| Primary Glow / Accent | `#7c3aed` | `--color-accent` | Forecast waveform trace (`dash`), active glow, kick bar |
| Secondary Accent | `#c084fc` | `--color-secondary` | Polar-CosmicFNO trace (`dot`), secondary spectral highlights |
| App Canvas Background | `#f7f6fb` | `--color-background` | Soft cosmic-tinted alabaster canvas (reduces eye strain) |
| Primary Ink (Foreground) | `#160e26` | `--color-foreground` | Headings, KPI values, reference SXS waveform (`> 12:1` AAA) |
| Secondary Ink (Muted Text) | `#554a6b` | `--color-muted-foreground` | Editorial subtitles, axis ticks, card descriptions (`> 5.8:1` AA) |
| Sidebar Muted Text | `#b5a6cc` | `--color-sidebar-fg` | Inactive sidebar navigation items |
| Sidebar Section Label | `#7c6b99` | `--color-sidebar-label` | Sentence-case group headers (`Módulos`, `Exploração`) |
| Card Surface | `#ffffff` | `--color-card` | Floating weightless KPI cards, white cards, Plotly containers |
| Plot Canvas Surface | `#faf9fc` | `--color-plot-bg` | Subtle inner plot area background |
| Control & Input Border | `#e2dcf0` | `--color-border` | Selectboxes, dividers, table rules (`#e6e1f0`) |
| Tag / Pill Surface | `#ede9fe` | `--color-muted` | Metadata tags and fidelity table pills (`#5b21b6` text) |
| Deep Solid Badge | `#2e1065` | `--color-badge-solid` | High-salience fidelity badge (`#ffffff` text) |

### Typography & Numeric Discipline

- **Primary Sans Font:** `Montserrat` (`400`, `500`, `600`, `700`, `800`)
- **Technical / Monospace Font:** `Fira Code` (`500`, `600`)
- **Icon Font:** `Material Symbols Rounded` (strictly preserved on `[data-testid="stIconMaterial"]`)
- **Numeric Alignment:** `font-variant-numeric: tabular-nums` enforced on all KPI values (`.kpi-value`), scientific tables (`.modern-table`), and parameter summaries (`.card-sub`) to prevent horizontal jitter during real-time parameter sweeps.
- **Editorial Hierarchy (Anthropic `frontend-design`):**
  - Sentence-case section labels and titles (no tracked-out ALL-CAPS eyebrow clichés).
  - Concise, high-signal scientific copy in Portuguese (`PT`), English (`EN`), and Spanish (`ES`).

**CSS Import:**
```css
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&family=Fira+Code:wght@500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
```

### Elevation & Antigravity Weightless Shadows

| Token | Value | Usage |
|-------|-------|-------|
| `--shadow-card` | `0 20px 40px -15px rgba(22, 14, 38, 0.06), 0 4px 12px -2px rgba(109, 40, 217, 0.03)` | KPI cards, white cards, Plotly containers, table card |
| `--shadow-control` | `0 4px 12px rgba(22, 14, 38, 0.03)` | Light-locked `st.selectbox` and interactive form controls |
| `--shadow-nav-active` | `0 6px 16px -4px rgba(109, 40, 217, 0.45)` | Active sidebar navigation button (`#6d28d9`) |

---

## 2. Layout & Component Architecture

### Fixed Cosmic Sidebar (`248px`)
1. **No Collapse / No Resize / Zero Top Dead-Space:**
   - `[data-testid="stSidebarCollapseButton"]`, `[data-testid="collapsedControl"]`, `[data-testid="stExpandSidebarButton"]`, and `div[data-testid="stSidebarResizeHandle"]` are permanently hidden (`display: none !important`).
   - `div[data-testid="stSidebarHeader"]` has `display: none !important; height: 0 !important; min-height: 0 !important; padding: 0 !important; margin: 0 !important;` to eliminate blank top space.
   - Sidebar width is locked at `248px` (`width: 248px !important; min-width: 248px !important; max-width: 248px !important;`).
2. **Compact Top-Left Language Cluster (`PT` | `EN` | `ES`):**
   - Positioned tightly in the top-left corner using `st.columns([1, 1, 1, 2.8], gap="small")` and scoped CSS (`.st-key-btn_pt`, `.st-key-btn_en`, `.st-key-btn_es`).
   - Button dimensions: `height: 24px !important; min-height: 24px !important; width: 34px !important; padding: 2px 8px !important; font-size: 0.68rem !important; border-radius: 6px !important; justify-content: center !important; text-align: center !important;`.
3. **Full-Width Left-Aligned Navigation Buttons:**
   - All 5 module buttons pass `use_container_width=True`.
   - CSS enforces `width: 100% !important; justify-content: flex-start !important; text-align: left !important; padding: 10px 14px !important; border-radius: 10px !important;`.

### Dark-Mode-Proof Light Controls (`st.selectbox`)
To prevent browser dark-mode preferences from rendering black dropdown inputs against the light `#f7f6fb` observatory canvas:
```css
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #160e26 !important;
    border: 1px solid #e2dcf0 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(22, 14, 38, 0.03) !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] div,
div[data-baseweb="select"] svg {
    color: #160e26 !important;
    fill: #160e26 !important;
}
```

### Weightless KPI Grid & Scientific Plots
- **KPI Grid:** 4-column responsive CSS grid (`.kpi-grid`) with `backdrop-filter: blur(12px)`, diffused elevation, and staggered entrance animation (`floatIn`) that respects `@media (prefers-reduced-motion: reduce)`.
- **Plotly Surfaces:** Explicitly locked to `template="plotly_white"`, `paper_bgcolor="#ffffff"`, `plot_bgcolor="#faf9fc"`, and `theme=None` on `st.plotly_chart` calls.

---

## 3. Anti-Patterns & Pre-Delivery Checklist

- [x] Sidebar is fixed at `248px` with zero top header dead-space and no collapse/resize handles.
- [x] Language switcher (`PT` | `EN` | `ES`) is compact (`34px × 24px`, `4px` gap) and anchored at the top-left of the sidebar.
- [x] All 5 sidebar navigation buttons span `100%` of the sidebar width (`use_container_width=True`) with left-aligned icon + label.
- [x] `st.selectbox` and popover menus are explicitly styled in light tokens (`#ffffff` surface, `#160e26` text, `#e2dcf0` border) regardless of OS/browser dark theme.
- [x] Material Symbols icons (`:material/...:`) render as crisp ligatures without font-family override regression.
- [x] All quantitative metrics use `tabular-nums` and maintain WCAG AA/AAA contrast (`>= 4.5:1`).
- [x] `prefers-reduced-motion: reduce` disables entrance animations cleanly.
