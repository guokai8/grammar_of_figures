Here’s a polished **README.md** you can drop into the repo. It mirrors your current files (`index.md`, `Chapter 0.md` … `Chapter 10.md`), adds quick-start workflows for Python/R, and includes licensing + citation.

````markdown
# Grammar of Figures
**The Art & Science of Visualizing Data for Publications**

Author: **Kai Guo** · Contact: [guokai8@gmail.com](mailto:guokai8@gmail.com)

This repository contains the manuscript and code snippets for *Grammar of Figures*, a practical, code-first guide to designing beautiful, rigorous, and journal-ready figures in **Python** and **R**.

---

## ✨ What’s inside

- **index.md** — master table of contents and chapter summaries  
- **Chapters 0–10** — complete text covering perception, color, typography, layout, technical specs, common figure types, case studies, interactivity, and troubleshooting  
- **Code snippets** — inline in chapters (Python/R) showing best practices for reproducible exports

> No `appendices/` or `assets/` folders are referenced here. Add them later if needed and update `index.md`.

---

## 📚 Reading order

1. **Chapter 0 — Purpose, Audience & Workflow** (`Chapter 0.md`)  
2. **Chapter 1 — Foundations of Visual Perception** (`Chapter 1.md`)  
3. **Chapter 2 — The Language of Color** (`Chapter 2.md`)  
4. **Chapter 3 — Typography, Annotation & Labels** (`Chapter 3.md`)  
5. **Chapter 4 — Data Encoding & Graph Selection** (`Chapter 4.md`)  
6. **Chapter 5 — Layout, Composition & Figure Assembly** (`Chapter 5.md`)  
7. **Chapter 6 — Technical Specifications & Publication Requirements** (`Chapter 6.md`)  
8. **Chapter 7 — Common Figure Types Deep Dive** (`Chapter 7.md`)  
9. **Chapter 8 — Case Studies & Experimental Figures** (`Chapter 8.md`)  
10. **Chapter 9 — Interactive and Dynamic Figures** (`Chapter 9.md`)  
11. **Chapter 10 — Figure Troubleshooting Guide** (`Chapter 10.md`)

Or just start from **`index.md`**.

---

## 🧪 Quick start (Python)

> Requires Python ≥ 3.10. Suggested packages: `matplotlib`, `seaborn`, `pandas`, `numpy`, `colorspacious` (for CVD checks).  
> Optional: `plotnine`, `scikit-image`.

```bash
# (optional) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install matplotlib seaborn pandas numpy colorspacious
````

**Reproducible export helper (DPI × inches math):**

```python
# save_figure.py
import matplotlib.pyplot as plt

def save_figure(path, width_mm=85, height_mm=60, dpi=300, transparent=False):
    inches = (width_mm / 25.4, height_mm / 25.4)
    fig = plt.gcf()
    fig.set_size_inches(*inches)
    plt.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=transparent,
        metadata={"Creator": "Grammar of Figures"}
    )
    print(f"Saved {path} at {dpi} DPI, {inches[0]:.2f}×{inches[1]:.2f} in")

# example usage
if __name__ == "__main__":
    import seaborn as sns, pandas as pd
    df = sns.load_dataset("penguins").dropna()
    ax = sns.scatterplot(data=df, x="bill_length_mm", y="flipper_length_mm", hue="species")
    ax.set_xlabel("Bill length (mm)")
    ax.set_ylabel("Flipper length (mm)")
    save_figure("figure1.png", width_mm=85, height_mm=60, dpi=300, transparent=False)
```

---

## 📊 Quick start (R)

> Requires R ≥ 4.2. Suggested packages: `ggplot2`, `patchwork`, `scales`, `viridisLite`, `colorspace`.

```r
install.packages(c("ggplot2", "patchwork", "scales", "viridisLite", "colorspace"))
```

**Reproducible export helper:**

```r
library(ggplot2)

save_figure <- function(plot, path, width_mm = 85, height_mm = 60, dpi = 300, bg = "white") {
  ggsave(
    filename = path,
    plot = plot,
    width = width_mm / 25.4,
    height = height_mm / 25.4,
    dpi = dpi,
    bg = bg,
    units = "in",
    limitsize = FALSE
  )
  message(sprintf("Saved %s at %d DPI, %.2f×%.2f in",
                  path, dpi, width_mm/25.4, height_mm/25.4))
}

p <- ggplot(mtcars, aes(wt, mpg, color = factor(cyl))) +
  geom_point(size = 2) +
  scale_color_viridis_d() +
  labs(x = "Weight (1000 lbs)", y = "Miles/(US) gallon", color = "Cylinders")

save_figure(p, "figure1.png", width_mm = 85, height_mm = 60, dpi = 300)
```

---

## 🧰 Conventions

* **Code style:** minimal, reproducible. Python uses `matplotlib`/`seaborn` (optionally `plotnine`); R uses `ggplot2` + tidyverse. Always save with explicit inches + DPI (`plt.savefig(...)`, `ggsave(...)`).
* **Typography:** SI units; sentence case for axis titles; Title Case for figure titles (as needed).
* **Accessibility:** CVD-safe palettes; minimum contrast; legends never convey meaning by color alone; prefer redundant encodings (color + shape/linetype).
* **Terminology:** *mark* (point/line/area), *channel* (position, length, angle, area, hue, saturation, luminance, shape, texture, motion).

---

## 📁 Repository map

```
./
├─ README.md
├─ index.md
├─ Chapter 0.md
├─ Chapter 1.md
├─ Chapter 2.md
├─ Chapter 3.md
├─ Chapter 4.md
├─ Chapter 5.md
├─ Chapter 6.md
├─ Chapter 7.md
├─ Chapter 8.md
├─ Chapter 9.md
└─ Chapter 10.md
```

---

## 🧭 Suggested workflow

1. **Start with Chapter 0** to define message, audience, venue.
2. **Design with Chapter 1–5**: perception → color → typography → encoding → layout.
3. **Export with Chapter 6**: choose correct format (PNG/TIFF/PDF/SVG) & DPI at final print size.
4. **Use Chapter 7–8** for domain-specific patterns and case studies.
5. **If interactive** (Chapter 9), provide static fallbacks for manuscripts.
6. **Run the pre-submission checks** in Chapter 10 (integrity, accessibility, tech specs).

---

## 🧑‍💻 Reproducibility tips

* Pin versions (`requirements.txt` or `renv`/`pak` for R).
* Seed randomness (`numpy.random.seed`, `set.seed`).
* Keep figure dimensions & DPI fixed; avoid scaling in word processors.
* Prefer **vector** (PDF/SVG) for line art; **raster** (PNG/TIFF) for images. Avoid JPEG for scientific plots.

---

## 🤝 Contributing

Issues and PRs are welcome for:

* Corrections and clarifications
* Additional minimal code examples
* New case studies mirroring common publication pitfalls

When contributing, keep examples **small, deterministic, and export-focused**.

---

## 🧾 License

* **Text:** CC BY 4.0
* **Code:** MIT

See `LICENSE` files (add if not present).

---

## 📌 How to cite

> Kai Guo. *Grammar of Figures: The Art & Science of Visualizing Data for Publications*. Version X.Y.Z, YEAR. DOI/URL.

---

## 📫 Contact

Questions or suggestions? Open an issue or email **[guokai8@gmail.com](mailto:guokai8@gmail.com)**.

```

Want me to save this into your repo as `README.md` too?
```

