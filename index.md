# Grammar of Figures: The Art & Science of Visualizing Data for Publications

**Author:** Kai Guo  •  **Contact:** [guokai8@gmail.com](mailto:guokai8@gmail.com)

> **index.md** — master table of contents for the files you provided (no appendices/assets referenced).

---

## About this book

A practical, code-first field guide to designing beautiful, rigorous figures for scientific publications using **Python** and **R**. It combines perceptual science, design principles, and journal requirements with reproducible workflows.

* **Audience:** researchers, data scientists, and visualization engineers targeting journals (e.g., Nature, Cell, Science) and preprints.
* **Prereqs:** basic Python/R, familiarity with matplotlib/ggplot2; comfort with command line for reproducible exports.
* **Goal:** help you decide *what* to show, *how* to encode it, and *why*—then ship print-ready, accessible figures.

---
---

## Conventions

* **Code style:** minimal, reproducible. Python uses `matplotlib`/`seaborn`/`plotnine` (when helpful); R uses `ggplot2` + tidyverse. Saving: `plt.savefig()` / `ggsave()` with explicit DPI and dimensions.
* **Typography:** SI units, sentence case for axis titles, Title Case for figure titles (configurable).
* **Accessibility:** color-vision-deficiency-safe palettes; minimum contrast ratios; legends never convey meaning by color alone; redundant encodings (color + shape/line).
* **Terminology:** *mark* (point/line/area), *channel* (position, length, angle, area, hue, saturation, luminance, shape, texture, motion).

---
## How to cite

> Kai Guo. *Grammar of Figures: The Art & Science of Visualizing Data for Publications*. Version X.Y.Z, YEAR. DOI/URL.

