# Chapter 5: Layout, Composition & Figure Assembly

## 5.1 The Gestalt Principles Applied to Scientific Figures

### Visual Perception Fundamentals

The human visual system automatically organizes visual information according to **Gestalt principles**—innate perceptual patterns that emerged from evolutionary psychology. Understanding these principles allows you to design figures that are instantly interpretable.

**The Core Insight:**
> Readers perceive relationships between elements **before** conscious analysis. Poor layout creates false groupings; good layout guides interpretation effortlessly.

---

### Principle 1: Proximity (Nearness Implies Relatedness)

**The Rule:** Elements placed close together are perceived as belonging to the same group.

**Application in Figures:**

```
✓ CORRECT spacing:
- Panels within a figure: Close together (related)
- Figures in manuscript: Separated (distinct)
- Axis label + axis: Immediate proximity (clear association)
- Legend + plot: Adjacent (functional relationship)

❌ INCORRECT spacing:
- Equal spacing between all panels (loses grouping information)
- Legend far from plot (requires search)
- Caption separated from figure by page break
```

**Code Example (Python) - Proximity in Multi-Panel Layouts:**

```
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Generate sample data for 6 panels
data_sets = [np.random.randn(50) + i*2 for i in range(6)]

fig = plt.figure(figsize=(16, 10))

# BAD: Equal spacing (no grouping information)
fig.text(0.25, 0.95, '❌ BAD: Equal Spacing (No Visual Grouping)',
         ha='center', fontsize=14, fontweight='bold', color='red')

gs_bad = fig.add_gridspec(3, 2, left=0.05, right=0.45, top=0.85, bottom=0.55,
                          wspace=0.3, hspace=0.4)  # Equal spacing

for i in range(6):
    ax = fig.add_subplot(gs_bad[i//2, i%2])
    ax.hist(data_sets[i], bins=15, color='#3498DB', edgecolor='black', alpha=0.7)
    ax.set_title(f'Panel {chr(65+i)}', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=9)
    ax.tick_params(labelsize=8)

# Add misleading text
fig.text(0.25, 0.52, 'All panels equally spaced\n→ No visual hierarchy or grouping',
         ha='center', fontsize=9, style='italic', color='red')

# GOOD: Grouped spacing (proximity shows relationship)
fig.text(0.75, 0.95, '✓ GOOD: Grouped Spacing (Clear Relationship)',
         ha='center', fontsize=14, fontweight='bold', color='green')

# Create two groups: Control (A-C) and Treatment (D-F)
# Group 1: Tighter spacing within group
gs_good_ctrl = fig.add_gridspec(3, 1, left=0.55, right=0.68, top=0.85, bottom=0.55,
                                hspace=0.15)  # Tight spacing

# Group 2: Separated from group 1
gs_good_trt = fig.add_gridspec(3, 1, left=0.77, right=0.90, top=0.85, bottom=0.55,
                               hspace=0.15)  # Tight spacing

# Plot Control group
for i in range(3):
    ax = fig.add_subplot(gs_good_ctrl[i])
    ax.hist(data_sets[i], bins=15, color='#7F8C8D', edgecolor='black', alpha=0.7)
    ax.set_title(f'Panel {chr(65+i)}', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=9)
    ax.tick_params(labelsize=8)
    if i == 0:
        ax.text(0.5, 1.15, 'CONTROL', transform=ax.transAxes,
               ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#7F8C8D', alpha=0.3))

# Plot Treatment group
for i in range(3, 6):
    ax = fig.add_subplot(gs_good_trt[i-3])
    ax.hist(data_sets[i], bins=15, color='#E74C3C', edgecolor='black', alpha=0.7)
    ax.set_title(f'Panel {chr(65+i)}', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=9)
    ax.tick_params(labelsize=8)
    if i == 3:
        ax.text(0.5, 1.15, 'TREATMENT', transform=ax.transAxes,
               ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#E74C3C', alpha=0.3))

# Add explanatory text
fig.text(0.725, 0.52, 'Tight spacing within groups\n→ Clear visual grouping',
         ha='center', fontsize=9, style='italic', color='green')

plt.savefig('proximity_principle.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R) - Proximity in Multi-Panel Layouts:**

```
library(ggplot2)
library(patchwork)
library(grid)

set.seed(42)

# Generate data
data_list <- lapply(1:6, function(i) {
  data.frame(value = rnorm(50, mean = i*2, sd = 1))
})

# Create individual plots
plots <- lapply(1:6, function(i) {
  ggplot(data_list[[i]], aes(x = value)) +
    geom_histogram(bins = 15, fill = '#3498DB', color = 'black', alpha = 0.7) +
    labs(title = paste0('Panel ', LETTERS[i]), y = 'Frequency') +
    theme_classic(base_size = 9) +
    theme(
      plot.title = element_text(face = 'bold', size = 10, hjust = 0.5),
      axis.title = element_text(face = 'bold', size = 9)
    )
})

# BAD: Equal spacing
p_bad <- wrap_plots(plots, ncol = 2, nrow = 3) +
  plot_annotation(
    title = '❌ BAD: Equal Spacing (No Visual Grouping)',
    subtitle = 'All panels equally spaced → No visual hierarchy or grouping',
    theme = theme(
      plot.title = element_text(face = 'bold', size = 14, color = 'red', hjust = 0.5),
      plot.subtitle = element_text(face = 'italic', size = 9, color = 'red', hjust = 0.5)
    )
  )

# GOOD: Grouped spacing
# Control group (gray)
plots_ctrl <- lapply(1:3, function(i) {
  ggplot(data_list[[i]], aes(x = value)) +
    geom_histogram(bins = 15, fill = '#7F8C8D', color = 'black', alpha = 0.7) +
    labs(title = paste0('Panel ', LETTERS[i]), y = 'Frequency') +
    theme_classic(base_size = 9) +
    theme(
      plot.title = element_text(face = 'bold', size = 10, hjust = 0.5),
      axis.title = element_text(face = 'bold', size = 9)
    )
})

# Treatment group (red)
plots_trt <- lapply(4:6, function(i) {
  ggplot(data_list[[i]], aes(x = value)) +
    geom_histogram(bins = 15, fill = '#E74C3C', color = 'black', alpha = 0.7) +
    labs(title = paste0('Panel ', LETTERS[i]), y = 'Frequency') +
    theme_classic(base_size = 9) +
    theme(
      plot.title = element_text(face = 'bold', size = 10, hjust = 0.5),
      axis.title = element_text(face = 'bold', size = 9)
    )
})

# Combine with tight spacing within groups
ctrl_combined <- wrap_plots(plots_ctrl, ncol = 1) +
  plot_annotation(title = 'CONTROL',
                  theme = theme(plot.title = element_text(face = 'bold', size = 11, hjust = 0.5)))

trt_combined <- wrap_plots(plots_trt, ncol = 1) +
  plot_annotation(title = 'TREATMENT',
                  theme = theme(plot.title = element_text(face = 'bold', size = 11, hjust = 0.5)))

p_good <- (ctrl_combined | trt_combined) +
  plot_annotation(
    title = '✓ GOOD: Grouped Spacing (Clear Relationship)',
    subtitle = 'Tight spacing within groups → Clear visual grouping',
    theme = theme(
      plot.title = element_text(face = 'bold', size = 14, color = 'darkgreen', hjust = 0.5),
      plot.subtitle = element_text(face = 'italic', size = 9, color = 'darkgreen', hjust = 0.5)
    )
  )

# Save (since we can't easily combine bad and good in one figure with patchwork,
# we'll save separately or use grid)

# For demonstration, save good example
ggsave('proximity_principle_good.png', p_good, width = 14, height = 10,
       dpi = 300, bg = 'white')
```

---

### Principle 2: Similarity (Like Elements Group Together)

**The Rule:** Elements that share visual properties (color, shape, size, orientation) are perceived as related.

**Application:**

```
✓ CORRECT use:
- Same color for same experimental group across all panels
- Same marker shape for same variable type
- Consistent line styles for same condition

❌ INCORRECT:
- Different colors for same group in different panels
- Inconsistent symbols without semantic reason
```

**Example from Section 2.5:**
```
If "Drug A" is blue in Figure 1, it MUST be blue in Figures 2, 3, 4...
→ Similarity creates immediate recognition
→ Inconsistency forces mental translation (cognitive load)
```

---

### Principle 3: Closure (Mind Completes Incomplete Shapes)

**Application in Figure Layout:**

```
Use closure to create implied groupings:
- Boxes/borders around related panels (creates enclosure)
- Aligned edges create implicit boundaries
- White space separates groups (implicit border)
```

**Code Example (Python) - Using Enclosure for Grouping:**

```
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

np.random.seed(42)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Generate sample data
for i, ax in enumerate(axes.flat):
    data = np.random.randn(100) + i
    ax.hist(data, bins=15, color='#3498DB', edgecolor='black', alpha=0.7)
    ax.set_title(f'Panel {chr(65+i)}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10)
    ax.tick_params(labelsize=9)

# Add enclosure boxes to create visual groups
# Group 1: Panels A, B, C (top row - Method 1)
bbox1 = FancyBboxPatch((0.08, 0.55), 0.85, 0.38,
                       boxstyle="round,pad=0.01",
                       edgecolor='#27AE60', facecolor='none',
                       linewidth=3, transform=fig.transFigure,
                       clip_on=False, zorder=10)
fig.patches.append(bbox1)
fig.text(0.5, 0.94, 'Method 1 Results', ha='center', fontsize=12,
        fontweight='bold', color='#27AE60')

# Group 2: Panels D, E, F (bottom row - Method 2)
bbox2 = FancyBboxPatch((0.08, 0.08), 0.85, 0.38,
                       boxstyle="round,pad=0.01",
                       edgecolor='#E67E22', facecolor='none',
                       linewidth=3, transform=fig.transFigure,
                       clip_on=False, zorder=10)
fig.patches.append(bbox2)
fig.text(0.5, 0.47, 'Method 2 Results', ha='center', fontsize=12,
        fontweight='bold', color='#E67E22')

plt.suptitle('Using Enclosure to Create Visual Grouping\n(Closure Principle)',
            fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('closure_principle_grouping.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

---

### Principle 4: Continuity (Eyes Follow Paths)

**Application:**

```
Alignment creates visual flow:
✓ Align panel edges (creates reading path)
✓ Align axis limits when comparing (facilitates comparison)
✓ Use consistent panel sizes (smooth visual scanning)

❌ AVOID:
- Misaligned panels (jagged, disorganized)
- Different aspect ratios without justification
- Inconsistent spacing (disrupts flow)
```

---

### Principle 5: Figure-Ground (Foreground vs. Background)

**Application:**

```
Create clear hierarchy:
✓ Data elements (foreground): Bold, saturated colors
✓ Grid/axes (background): Light gray, thin lines
✓ Annotations (mid-ground): Medium contrast

Visual hierarchy = Information hierarchy
```

---

## 5.2 The Grid System for Scientific Figures

### Why Grids Matter

A **grid system** provides invisible structure that:
- Ensures consistent alignment
- Creates visual rhythm
- Facilitates comparison across panels
- Signals professionalism

**The Standard: Column-Based Grid**

```
Common layouts:
- 1-column: Full width (simple figures)
- 2-column: Split left/right (comparisons)
- 3-column: Triptych (sequential processes)
- 2×2 grid: Four related panels
- Mixed: Large main + smaller supporting panels
```

---

### Layout Strategy 1: Equal Weight Panels

**When to use:** All panels equally important, same data type

**Code Example (Python) - Balanced 2×2 Grid:**

```
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Generate different plot types
plot_types = [
    ('Scatter', 'scatter'),
    ('Line', 'line'),
    ('Bar', 'bar'),
    ('Box', 'box')
]

for idx, (title, ptype) in enumerate(plot_types):
    ax = axes.flat[idx]

    if ptype == 'scatter':
        x = np.random.randn(50)
        y = 2*x + np.random.randn(50)
        ax.scatter(x, y, s=50, color='#3498DB', alpha=0.7, edgecolors='black', linewidths=0.5)
        ax.set_xlabel('Variable X', fontsize=10, fontweight='bold')
        ax.set_ylabel('Variable Y', fontsize=10, fontweight='bold')

    elif ptype == 'line':
        time = np.linspace(0, 10, 50)
        signal = 5 + 2*np.sin(time) + np.random.randn(50)*0.5
        ax.plot(time, signal, 'o-', color='#27AE60', linewidth=2, markersize=4)
        ax.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Signal (mV)', fontsize=10, fontweight='bold')

    elif ptype == 'bar':
        categories = ['A', 'B', 'C', 'D']
        values = [25, 32, 28, 35]
        ax.bar(categories, values, color='#E74C3C', edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Response (AU)', fontsize=10, fontweight='bold')

    elif ptype == 'box':
        data = [np.random.normal(20+i*5, 3, 50) for i in range(4)]
        bp = ax.boxplot(data, labels=['G1', 'G2', 'G3', 'G4'], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#F39C12')
        ax.set_ylabel('Measurement', fontsize=10, fontweight='bold')

    # Panel label
    ax.text(-0.15, 1.05, chr(65+idx), transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='top')

    ax.set_title(f'{title} Plot', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Equal Weight Grid Layout (2×2)', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('equal_weight_grid.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
```

---

### Layout Strategy 2: Dominant Panel + Supporting Panels

**When to use:** One main finding + supporting evidence or details

**Structure:**
```
┌─────────────┬─────┐
│             │  B  │
│      A      ├─────┤
│  (main)     │  C  │
│             ├─────┤
└─────────────┴─────┘

A: Large (60-70% of space) - Main result
B, C: Small (15-20% each) - Supporting/methods
```

**Code Example (Python) - Dominant Layout:**

```
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

fig = plt.figure(figsize=(14, 8))

# Create grid: 2 rows, 2 columns with different sizes
gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1],
                      hspace=0.3, wspace=0.3)

# Panel A: Main result (large, spans 2 rows)
ax_main = fig.add_subplot(gs[:, 0])
x = np.random.randn(200)
y = 2.5*x + np.random.randn(200)*1.5
ax_main.scatter(x, y, s=60, color='#3498DB', alpha=0.6, edgecolors='black', linewidths=0.5)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
x_line = np.linspace(x.min(), x.max(), 100)
ax_main.plot(x_line, p(x_line), 'r--', linewidth=3, label=f'y = {z[1]:.2f} + {z[0]:.2f}x')
ax_main.set_xlabel('Treatment Dose (μM)', fontsize=12, fontweight='bold')
ax_main.set_ylabel('Cell Response (AU)', fontsize=12, fontweight='bold')
ax_main.set_title('Main Result: Dose-Response Relationship', fontsize=13, fontweight='bold')
ax_main.legend(loc='upper left', fontsize=11, frameon=True)
ax_main.grid(alpha=0.3)
ax_main.text(-0.1, 1.05, 'A', transform=ax_main.transAxes,
            fontsize=18, fontweight='bold', va='top')

# Panel B: Supporting (top right)
ax_b = fig.add_subplot(gs[0, 1])
categories = ['Control', 'Treated']
values = [25, 35]
errors = [3, 4]
ax_b.bar(categories, values, color=['#7F8C8D', '#E74C3C'],
        edgecolor='black', linewidth=1.5, width=0.5)
ax_b.errorbar(categories, values, yerr=errors, fmt='none',
             ecolor='black', capsize=6, linewidth=2)
ax_b.set_ylabel('Viability (%)', fontsize=10, fontweight='bold')
ax_b.set_title('Endpoint Viability', fontsize=10, fontweight='bold')
ax_b.set_ylim(0, 50)
ax_b.grid(axis='y', alpha=0.3)
ax_b.text(-0.25, 1.1, 'B', transform=ax_b.transAxes,
         fontsize=16, fontweight='bold', va='top')

# Panel C: Supporting (bottom right)
ax_c = fig.add_subplot(gs[1, 1])
time = np.linspace(0, 24, 50)
signal = 100 + 20*np.sin(2*np.pi*time/24) + np.random.randn(50)*3
ax_c.plot(time, signal, 'o-', color='#27AE60', linewidth=2, markersize=3)
ax_c.set_xlabel('Time (h)', fontsize=10, fontweight='bold')
ax_c.set_ylabel('Signal', fontsize=10, fontweight='bold')
ax_c.set_title('Temporal Control', fontsize=10, fontweight='bold')
ax_c.grid(alpha=0.3)
ax_c.text(-0.25, 1.1, 'C', transform=ax_c.transAxes,
         fontsize=16, fontweight='bold', va='top')

plt.suptitle('Dominant Layout: Main Result + Supporting Evidence',
            fontsize=14, fontweight='bold', y=0.98)

plt.savefig('dominant_panel_layout.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R) - Dominant Layout:**

```
library(ggplot2)
library(patchwork)

set.seed(42)

# Panel A: Main result (large)
data_main <- data.frame(
  dose = rnorm(200, 0, 1),
  response = 2.5 * rnorm(200, 0, 1) + rnorm(200, 0, 1.5)
)

p_main <- ggplot(data_main, aes(x = dose, y = response)) +
  geom_point(size = 3, color = '#3498DB', alpha = 0.6) +
  geom_smooth(method = 'lm', se = TRUE, color = 'red', linetype = 'dashed', size = 1.5) +
  labs(x = 'Treatment Dose (μM)', y = 'Cell Response (AU)',
       title = 'Main Result: Dose-Response Relationship') +
  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(face = 'bold', size = 13, hjust = 0.5),
    axis.title = element_text(face = 'bold', size = 12),
    panel.grid.major = element_line(color = 'gray90', size = 0.3)
  )

# Panel B: Supporting
data_b <- data.frame(
  category = factor(c('Control', 'Treated'), levels = c('Control', 'Treated')),
  value = c(25, 35),
  error = c(3, 4)
)

p_b <- ggplot(data_b, aes(x = category, y = value, fill = category)) +
  geom_bar(stat = 'identity', color = 'black', size = 1, width = 0.5) +
  geom_errorbar(aes(ymin = value - error, ymax = value + error),
               width = 0.2, size = 1) +
  scale_fill_manual(values = c('Control' = '#7F8C8D', 'Treated' = '#E74C3C')) +
  labs(y = 'Viability (%)', title = 'Endpoint Viability') +
  ylim(0, 50) +
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_text(face = 'bold', size = 10, hjust = 0.5),
    axis.title.x = element_blank(),
    axis.title.y = element_text(face = 'bold', size = 10),
    legend.position = 'none',
    panel.grid.major.y = element_line(color = 'gray90', size = 0.3)
  )

# Panel C: Supporting
data_c <- data.frame(
  time = seq(0, 24, length.out = 50),
  signal = 100 + 20*sin(2*pi*seq(0, 24, length.out = 50)/24) + rnorm(50, 0, 3)
)

p_c <- ggplot(data_c, aes(x = time, y = signal)) +
  geom_line(color = '#27AE60', size = 1.2) +
  geom_point(color = '#27AE60', size = 2) +
  labs(x = 'Time (h)', y = 'Signal', title = 'Temporal Control') +
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_text(face = 'bold', size = 10, hjust = 0.5),
    axis.title = element_text(face = 'bold', size = 10),
    panel.grid.major = element_line(color = 'gray90', size = 0.3)
  )

# Combine with dominant layout
layout <- "
AAB
AAC
"

combined <- p_main + p_b + p_c +
  plot_layout(design = layout) +
  plot_annotation(
    title = 'Dominant Layout: Main Result + Supporting Evidence',
    tag_levels = 'A',
    theme = theme(
      plot.title = element_text(face = 'bold', size = 14, hjust = 0.5),
      plot.tag = element_text(size = 18, face = 'bold')
    )
  )

ggsave('dominant_panel_layout.png', combined, width = 14, height = 8,
       dpi = 300, bg = 'white')
```

---

### Layout Strategy 3: Sequential/Narrative Flow

**When to use:** Showing a process, temporal sequence, or methodological pipeline

**Structure:** Left-to-right or top-to-bottom progression

```
Time 0 → Time 1 → Time 2 → Time 3
  OR
Step 1 ↓
Step 2 ↓
Step 3 ↓
Step 4 ↓
```

**Visual cues for flow:**
- Arrows between panels
- Progressive color intensity
- Consistent aspect ratio (smooth scanning)

---


## 5.3 Aspect Ratios and Panel Sizing

### The Golden Ratio and Practical Aspect Ratios

**Aspect ratio** = Width / Height

Different ratios serve different purposes and affect perception:

**Common aspect ratios in scientific figures:**

```
1:1 (Square)
- Use for: Heatmaps, correlation matrices, symmetric data
- Perception: Balanced, no directional bias
- Example: 6×6 inches

4:3 (Standard)
- Use for: General purpose, presentations
- Perception: Slightly wider, comfortable viewing
- Example: 8×6 inches

16:9 (Widescreen)
- Use for: Time series, horizontal comparisons
- Perception: Emphasizes horizontal progression
- Example: 10×5.625 inches

3:2 (Classic photography)
- Use for: Balanced figures, portraits
- Perception: Natural, versatile
- Example: 9×6 inches

Golden ratio (~1.618:1)
- Use for: Aesthetically pleasing single panels
- Perception: Harmonious, professional
- Example: 9.7×6 inches
```

---

### Matching Aspect Ratio to Data Structure

**The Principle: Aspect ratio should emphasize data relationships**

**Horizontal (Wide) Ratios:**
```
✓ BEST for:
- Time series (time flows left-to-right)
- Sequential processes
- Many categories on x-axis
- Comparisons across groups

Example: 16:9 or 2:1
```

**Vertical (Tall) Ratios:**
```
✓ BEST for:
- Hierarchical trees/dendrograms
- Stacked plots (multiple time series)
- Rank-ordered lists
- Vertical processes

Example: 1:2 or 9:16
```

**Square Ratios:**
```
✓ BEST for:
- Symmetric relationships (correlation matrices)
- Spatial data without directional bias
- Heatmaps with equal dimensions
- Network diagrams

Example: 1:1
```

---

**Code Example (Python) - Aspect Ratio Effects:**

```
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Generate time series data
time = np.linspace(0, 24, 100)
signal = 100 + 20*np.sin(2*np.pi*time/24) + np.random.randn(100)*3

fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# Aspect ratio 1: Too tall (vertical bias)
ax1 = axes[0]
ax1.plot(time, signal, 'o-', color='#3498DB', linewidth=2, markersize=3)
ax1.set_xlabel('Time (hours)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Response', fontsize=10, fontweight='bold')
ax1.set_title('❌ TOO TALL (1:3 aspect)\nEmphasizes vertical, hard to see temporal pattern',
             fontsize=11, fontweight='bold', color='red')
ax1.set_aspect('auto')
ax1.grid(alpha=0.3)

# Aspect ratio 2: Good (horizontal emphasis)
ax2 = axes[1]
ax2.plot(time, signal, 'o-', color='#27AE60', linewidth=2, markersize=3)
ax2.set_xlabel('Time (hours)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Response', fontsize=10, fontweight='bold')
ax2.set_title('✓ GOOD (3:1 aspect)\nEmphasizes horizontal flow, clear temporal pattern',
             fontsize=11, fontweight='bold', color='green')
ax2.set_aspect('auto')
ax2.grid(alpha=0.3)

# For comparison: Make ax2 appear wider by adjusting position
pos = ax2.get_position()
ax2.set_position([pos.x0, pos.y0, pos.width, pos.height*0.5])

# Aspect ratio 3: Square (no directional emphasis)
ax3 = axes[2]
ax3.plot(time, signal, 'o-', color='#E67E22', linewidth=2, markersize=3)
ax3.set_xlabel('Time (hours)', fontsize=10, fontweight='bold')
ax3.set_ylabel('Response', fontsize=10, fontweight='bold')
ax3.set_title('⚠ SQUARE (1:1 aspect)\nNo directional bias, but less ideal for time series',
             fontsize=11, fontweight='bold', color='orange')
ax3.set_aspect('auto')
ax3.grid(alpha=0.3)

plt.suptitle('Aspect Ratio Impact on Time Series Perception',
            fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('aspect_ratio_effects.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Aspect ratio affects data perception")
print("✓ Match ratio to data structure (wide for time, square for symmetric)")
```

---

### Panel Size Consistency Rules

**Within a single figure:**

```
✓ CONSISTENT sizes when:
- Comparing similar data types
- All panels equally important
- Direct visual comparison needed

✓ VARIABLE sizes when:
- Main result + supporting evidence
- Different data types (image vs. plot)
- Hierarchical importance
```

**Across figures in manuscript:**

```
✓ Keep consistent:
- Axis label font sizes
- Marker sizes
- Line widths
- Color schemes

✓ Can vary:
- Overall figure dimensions (based on content)
- Number of panels
- Layout structure
```

---

## 5.4 White Space: The Unsung Hero

### The Power of Negative Space

**White space** (or negative space) is NOT wasted space—it's functional design element that:

1. **Separates groups** (visual breathing room)
2. **Directs attention** (eyes rest on dense areas)
3. **Reduces cognitive load** (prevents overwhelming)
4. **Signals professionalism** (crowded = amateur)

---

### The Density Principle

**Rule of thumb: 40-60% white space in scientific figures**

```
Too dense (<30% white space):
→ Overwhelming, cluttered
→ Hard to identify key elements
→ Looks amateur

Too sparse (>70% white space):
→ Inefficient use of space
→ May appear incomplete
→ Poor for publications with page limits

Optimal (40-60%):
→ Clear visual hierarchy
→ Easy to scan
→ Professional appearance
```

---

**Code Example (Python) - White Space Management:**

```
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Sample data
categories = ['Group A', 'Group B', 'Group C', 'Group D']
values = [25, 32, 28, 35]
errors = [3, 4, 3.5, 4.2]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# BAD: Too cramped (insufficient white space)
ax1 = axes[0]
bars = ax1.bar(categories, values, width=0.95, color='#3498DB',  # Width = 0.95 (very wide)
              edgecolor='black', linewidth=1.5)
ax1.errorbar(categories, values, yerr=errors, fmt='none',
            ecolor='black', capsize=8, linewidth=2)
ax1.set_ylabel('Response (AU)', fontsize=11, fontweight='bold')
ax1.set_title('❌ BAD: Insufficient White Space\n(Cramped, hard to distinguish)',
             fontsize=12, fontweight='bold', color='red')
ax1.set_ylim(0, 45)
ax1.grid(axis='y', alpha=0.3)
# Add cluttered annotations
for i, (bar, val) in enumerate(zip(bars, values)):
    ax1.text(i, val + errors[i] + 1, f'{val}±{errors[i]:.1f}',
            ha='center', fontsize=8)
# Remove spines
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# GOOD: Appropriate white space
ax2 = axes[1]
bars = ax2.bar(categories, values, width=0.6, color='#27AE60',  # Width = 0.6 (breathing room)
              edgecolor='black', linewidth=1.5)
ax2.errorbar(categories, values, yerr=errors, fmt='none',
            ecolor='black', capsize=8, linewidth=2)
ax2.set_ylabel('Response (AU)', fontsize=11, fontweight='bold')
ax2.set_title('✓ GOOD: Appropriate White Space\n(Clear, easy to read)',
             fontsize=12, fontweight='bold', color='green')
ax2.set_ylim(0, 45)
ax2.grid(axis='y', alpha=0.3)
# Spaced annotations
for i, (bar, val) in enumerate(zip(bars, values)):
    ax2.text(i, 42, f'n=15', ha='center', fontsize=9, style='italic')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('white_space_importance.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

---

### Margin and Padding Guidelines

**Standard spacing hierarchy:**

```
Between figure and edge (outer margin):
- Minimum: 0.5 inches (1.27 cm)
- Recommended: 0.75-1 inch (prevents cropping)

Between panels (internal spacing):
- Related panels: 0.25-0.5 inches
- Separate groups: 0.75-1 inch

Between elements within panel:
- Axis to tick labels: ~0.1 inches
- Tick labels to axis label: ~0.15 inches
- Title to plot area: ~0.2 inches
```

**In code (Python):**

```
# Set spacing explicitly
plt.tight_layout(pad=1.5,      # Outer padding
                h_pad=2.0,     # Vertical spacing between panels
                w_pad=2.0)     # Horizontal spacing between panels

# Or manually with subplots_adjust
plt.subplots_adjust(left=0.1, right=0.95,   # Outer margins
                   top=0.92, bottom=0.08,
                   hspace=0.3, wspace=0.3)  # Internal spacing
```

---

## 5.5 Multi-Figure Consistency Across a Manuscript

### The Style Guide Approach

**Create a figure style guide at the START of manuscript preparation:**

```
Document specifications:
├─ Font family: Arial
├─ Font sizes:
│   ├─ Panel labels: 14pt bold
│   ├─ Titles: 12pt bold
│   ├─ Axis labels: 11pt bold
│   ├─ Tick labels: 9pt regular
│   └─ Annotations: 9pt regular/italic
├─ Colors:
│   ├─ Control: #7F8C8D (gray)
│   ├─ Treatment A: #3498DB (blue)
│   ├─ Treatment B: #E74C3C (red)
│   └─ Significant: #E67E22 (orange highlights)
├─ Line widths:
│   ├─ Data lines: 2.5pt
│   ├─ Axis lines: 1.5pt
│   └─ Grid lines: 0.5pt, alpha=0.3
├─ Marker sizes:
│   ├─ Scatter points: 50-80 (matplotlib units)
│   └─ Line markers: 5-6
└─ Aspect ratios:
    ├─ Time series: 16:9 or 3:1
    ├─ Comparisons: 4:3 or 3:2
    └─ Heatmaps: 1:1 or data-dependent
```

---

### Template System Implementation

**Python template:**

```
import matplotlib.pyplot as plt

# Define once, use everywhere
FIGURE_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 9,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.linewidth': 1.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'lines.linewidth': 2.5,
    'lines.markersize': 6,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5
}

# Colors (semantic consistency)
COLORS = {
    'control': '#7F8C8D',
    'treatment_a': '#3498DB',
    'treatment_b': '#E74C3C',
    'significant': '#E67E22'
}

# Apply globally
plt.rcParams.update(FIGURE_STYLE)

# Function to set consistent panel formatting
def format_panel(ax, xlabel, ylabel, title, panel_label):
    """Apply consistent formatting to all panels"""
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='top')
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

# Example usage
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot([1, 2, 3], [1, 4, 2], color=COLORS['treatment_a'])
format_panel(ax, 'Time (s)', 'Response (AU)', 'Example Plot', 'A')
plt.savefig('consistent_figure.png', dpi=300, bbox_inches='tight')
plt.close()
```

**R template:**

```
library(ggplot2)

# Define theme once
manuscript_theme <- theme_classic(base_size = 11, base_family = 'Arial') +
  theme(
    axis.title = element_text(face = 'bold', size = 11),
    plot.title = element_text(face = 'bold', size = 12, hjust = 0.5),
    axis.text = element_text(size = 9),
    legend.text = element_text(size = 9),
    legend.title = element_text(face = 'bold', size = 9),
    panel.grid.major = element_line(color = 'gray90', size = 0.3),
    plot.tag = element_text(size = 16, face = 'bold')
  )

# Colors (semantic consistency)
COLORS <- list(
  control = '#7F8C8D',
  treatment_a = '#3498DB',
  treatment_b = '#E74C3C',
  significant = '#E67E22'
)

# Example usage
p <- ggplot(data, aes(x = time, y = response)) +
  geom_line(color = COLORS$treatment_a, size = 1.5) +
  labs(x = 'Time (s)', y = 'Response (AU)', title = 'Example Plot', tag = 'A') +
  manuscript_theme

ggsave('consistent_figure.png', p, width = 7, height = 5, dpi = 300)
```

---

### Cross-Figure Checklist

**Before submitting manuscript, verify:**

- [ ] **All figures use same font family**
- [ ] **Font sizes consistent** (panel labels, axes, etc.)
- [ ] **Color schemes consistent** (Control always gray, Drug A always blue, etc.)
- [ ] **Line widths consistent** across figure types
- [ ] **Marker sizes consistent**
- [ ] **Panel label format consistent** (A, B, C... in same position)
- [ ] **Aspect ratios appropriate** for each data type
- [ ] **White space balanced** (not too cramped or too sparse)
- [ ] **Grid styles consistent** (if used)
- [ ] **Error bar styles consistent** (SEM vs SD documented)
- [ ] **Statistical annotations consistent** (*, **, *** system)
- [ ] **All figures export at same DPI** (300 minimum)
- [ ] **File formats consistent** (TIFF/PNG for publication)

---

### Version Control for Figures

**Best practices:**

```
File naming convention:
figure1_version1_20250107.png
figure1_version2_20250115.png (after revisions)
figure1_final_20250120.png

Track changes:
- Keep log of what changed between versions
- Save original high-resolution files separately
- Document software versions used
```

---

**Summary of Chapter 5 so far:**

✓ **Gestalt principles**: Proximity, similarity, closure guide perception
✓ **Grid systems**: Structured layouts (equal weight, dominant, sequential)
✓ **Aspect ratios**: Match to data (wide for time, square for symmetric)
✓ **White space**: 40-60% optimal, prevents crowding
✓ **Consistency**: Style guide + templates ensure manuscript coherence

---

**End of Chapter 5: Layout, Composition & Figure Assembly**

**Final Figure Assembly Workflow:**

```
1. Define style guide (fonts, colors, sizes)
2. Create templates (Python rcParams / R themes)
3. Generate individual panels with consistent formatting
4. Assemble using grid system (appropriate layout strategy)
5. Balance white space (check density)
6. Add panel labels (A, B, C...) consistently
7. Verify aspect ratios match data types
8. Cross-check against manuscript figures for consistency
9. Export at publication quality (300 DPI, TIFF/PNG)
10. Review entire figure set side-by-side before submission
```

---

---

# Scientific Figure Design: Comprehensive Quick Reference Guide

## Part I: Pre-Design Decision Framework

### Step 1: Define Your Message
```
Ask yourself:
1. What is the ONE key finding this figure must communicate?
2. Who is the audience? (Specialists vs. general scientists)
3. What comparison is most important? (Groups, time, relationships)
4. What level of detail is required? (Overview vs. granular)

→ Answer determines: Plot type, layout complexity, detail level
```

### Step 2: Assess Your Data Structure
```
Data characteristics checklist:
 Continuous or categorical variables?
 How many variables? (1, 2, 3+)
 Sample size? (n < 20, 20-100, >100)
 Distribution shape? (Normal, skewed, bimodal)
 Temporal component? (Time series, sequential)
 Hierarchical or grouped structure?
 Comparisons needed? (Between groups, over time, correlations)

→ Determines: Plot type selection (see Part II)
```

### Step 3: Choose Plot Type
*See detailed decision tree in Part II*

### Step 4: Establish Style Specifications
```
Before creating ANY figures, document:
 Font family (Arial/Helvetica recommended)
 Font size hierarchy (panel labels > titles > axis labels > ticks)
 Color scheme (3 colors maximum, semantic consistency)
 Line widths (data: 2-3pt, axes: 1-1.5pt, grid: 0.5pt)
 Marker sizes (scatter: 50-80, line markers: 5-6)
 Error bar type (SEM vs SD - must be consistent)

→ Create template file to enforce consistency
```

---

## Part II: Plot Type Selection Matrix

### For Comparing Groups

| Scenario | Best Plot | Key Considerations |
|----------|-----------|-------------------|
| **Few groups (<5), one variable** | Bar chart | Start at zero, show error bars, include n |
| **Few groups, show distribution** | Box plot or Violin plot | Box = quartiles, Violin = density shape |
| **Many groups (>5)** | Small multiples or Heatmap | Avoid single cluttered panel |
| **Bimodal/complex distributions** | Violin plot | Box plot will hide multiple peaks |

### For Showing Distributions

| Scenario | Best Plot | Key Considerations |
|----------|-----------|-------------------|
| **Single distribution** | Histogram or Density plot | Choose bin width carefully (Sturges', FD) |
| **Compare 2-3 distributions** | Overlapping density plots | Transparency helps |
| **Compare many distributions** | Ridgeline plot or Box plot grid | Maintain consistent scales |

### For Relationships

| Scenario | Best Plot | Key Considerations |
|----------|-----------|-------------------|
| **Two continuous variables** | Scatter plot | Always plot raw data (Anscombe warning!) |
| **Dense scatter (>1000 points)** | Hexbin or 2D density | Avoid point occlusion |
| **Correlation matrix** | Heatmap with hierarchical clustering | Symmetric diverging colormap |

### For Temporal Data

| Scenario | Best Plot | Key Considerations |
|----------|-----------|-------------------|
| **Single time series** | Line graph | Lines imply continuity |
| **Few time series (2-4)** | Line graph, different colors | Add legend or direct labels |
| **Many time series (>5)** | Small multiples OR Highlight one + gray others | Avoid spaghetti plot |
| **Time series with error** | Shaded error band (SEM/SD) | Cleaner than error bars at every point |

### What to AVOID

| Never Use | Why | Use Instead |
|-----------|-----|-------------|
| **Pie charts** | Angles harder to compare than lengths | Bar chart (horizontal if long labels) |
| **3D effects on 2D data** | Perspective distortion, occlusion | Standard 2D with color/size for 3rd dimension |
| **Dual y-axes (usually)** | Easily manipulated to mislead | Separate panels or normalize to same scale |
| **Truncated bar charts** | Exaggerates small differences | Always start bars at zero |

---

## Part III: Color Decision Framework

### Step 1: Identify Data Type

```
Sequential (ordered, one direction):
→ Use: Single-hue gradient (light → dark)
→ Colormap: viridis, plasma, YlOrRd, Blues
→ Example: Gene expression (0 to max)

Diverging (ordered, two directions from center):
→ Use: Two-hue gradient with neutral center
→ Colormap: RdBu, PiYG, BrBG (ensure symmetric!)
→ Example: Fold change (-∞ to +∞, center at 0)

Categorical (unordered groups):
→ Use: Distinct hues, equal saturation
→ Colormap: Okabe-Ito palette, Set2, Dark2
→ Example: Different cell types, treatment groups
```

### Step 2: Apply Semantic Color Logic

```
Control/Baseline: Gray (#7F8C8D)
Experimental conditions: Colors (distinct hues)
Significant results: Deeper saturation
Non-significant: Lighter saturation or gray

CRITICAL: Same group = same color across ALL figures
```

### Step 3: Accessibility Check

```
Mandatory checks before finalizing:
 Uses colorblind-safe palette (Okabe-Ito recommended)
 Redundant encoding applied (color + shape/line style)
 Tested with Color Oracle or Coblis simulator
 Works in grayscale (print test)
 Text contrast meets WCAG AA (4.5:1 minimum)
```

### Step 4: Special Cases

```
All values significant?
→ Don't use binary deep/light colors
→ Encode magnitude with continuous gradient

Missing data (NA)?
→ NEVER use a color from your gradient
→ Use distinct color (gray with black border) OR pattern (crosshatch)

Cyclic data (time of day, angles)?
→ Use cyclic colormap (twilight, hsv)
→ Start and end colors must be similar
```

---

## Part IV: Typography & Annotation Checklist

### Font Specifications

```
Font hierarchy (apply consistently):
Panel labels:     14-16pt, bold, black
Titles:           12-13pt, bold, black
Axis labels:      11pt, bold, black  ← ALWAYS include units: "Variable (unit)"
Tick labels:      9pt, regular, dark gray (#333)
Legend text:      9pt, regular, black
Annotations:      9-10pt, regular/italic, black or dark gray

Font family: Sans-serif (Arial, Helvetica, Calibri)
```

### Axis Label Requirements

```
Format: "Variable Name (units)"

Examples:
✓ "Temperature (°C)"
✓ "Time (hours)"
✓ "Expression Level (FPKM)"
✓ "Fold Change (log₂)"

❌ NEVER:
- "Temperature" (missing units)
- "Expression" (ambiguous scale)
- "Time" (seconds? hours? days?)
```

### Statistical Annotations

```
Standard notation:
*     p < 0.05
**    p < 0.01
***   p < 0.001
n.s.  p ≥ 0.05

Required in caption:
- Statistical test used (e.g., "two-way ANOVA with Tukey post-hoc")
- Exact p-values if critical (e.g., "p = 0.003")
- Sample sizes (e.g., "n = 15 per group")
```

### Caption Structure

```
[Figure #]. [One-sentence summary of main finding].
(A) [Panel A description: what + how + n + stats].
(B) [Panel B description: what + how + n + stats].
[Error bar definition]. [Statistical methods]. [Abbreviations].

Example:
"Figure 2. Treatment A reduces tumor volume in xenograft mice.
(A) Tumor volume over time (n=8 mice per group, mean ± SEM,
two-way ANOVA, **p<0.01). (B) Survival curves (log-rank test,
***p<0.001). Error bars: SEM. * p<0.05, ** p<0.01, *** p<0.001."
```

---

## Part V: Layout & Composition Checklist

### Panel Arrangement

```
Equal weight (all panels equally important):
└─ Use: 2×2, 3×3, or uniform grid
└─ Spacing: Equal margins between all panels

Dominant panel (one main + supporting):
└─ Use: Large panel (60-70%) + small panels (15-20%)
└─ Spacing: Main panel prominent, supporting clustered nearby

Sequential/narrative (process flow):
└─ Use: Left-to-right or top-to-bottom progression
└─ Spacing: Tight within sequence, clear breaks between stages
```

### Aspect Ratio Selection

```
Data Type          → Recommended Aspect Ratio
Time series        → 16:9 or 3:1 (wide)
Group comparisons  → 4:3 or 3:2 (standard)
Heatmaps           → 1:1 or data-dependent (square/rectangular)
Hierarchical trees → 1:2 or 9:16 (tall)
Spatial maps       → Match geographic proportions
```

### White Space Guidelines

```
Target: 40-60% white space

Spacing hierarchy (smallest to largest):
1. Within panel: 0.1-0.2 inches (axis to labels)
2. Between related panels: 0.25-0.5 inches
3. Between panel groups: 0.75-1 inch
4. Figure outer margin: 0.75-1 inch

Bar chart specific:
- Bar width : gap ratio ≈ 3:1
- Leave breathing room between bars
```

---

## Part VI: Pre-Submission Quality Control

### Technical Specifications

```
Resolution:
 Minimum 300 DPI for publication
 600 DPI if images contain fine details (microscopy)

File formats:
 TIFF or PNG for final submission (lossless)
 Vector formats (PDF, EPS) for line graphs if allowed
 RGB color mode (unless journal requires CMYK)

Dimensions:
 Check journal specifications (often 3.5" or 7" column width)
 Export at intended print size (don't rely on scaling)
```

### Cross-Figure Consistency Check

```
Verify across ALL manuscript figures:
 Font family identical
 Font sizes consistent (panel labels, axes, ticks)
 Color schemes consistent (Control always gray, Drug A always blue, etc.)
 Line widths uniform
 Marker sizes uniform
 Panel label format consistent (A, B, C... placement)
 Error bar definition consistent (all SEM or all SD)
 Statistical notation consistent (*, **, ***)
 Grid styles consistent (if used)
 Legend positions logical
```

### Accessibility Verification

```
Final checks before submission:
 Colorblind-safe palette used (Okabe-Ito, ColorBrewer CVD-safe)
 Redundant encoding present (color + shape/line for critical distinctions)
 Tested with Color Oracle or Coblis
 Prints clearly in grayscale
 Text contrast sufficient (WCAG AA: 4.5:1)
 Font sizes readable when printed at target size
 No reliance on color alone for critical information
```

### Ethical Compliance

```
Image integrity (microscopy, gels, photos):
 No selective brightness/contrast adjustments
 All adjustments applied uniformly to all comparison images
 Linear adjustments only (no gamma correction)
 Original, unprocessed images available if requested
 All processing documented in Methods section

Color scale integrity:
 Symmetric scales for diverging data (no manipulation)
 No truncated axes (unless explicitly justified)
 Colormap choices documented in caption
 Missing data handled appropriately (not hidden)
```

---

## Part VII: Common Mistakes Quick Reference

### Top 10 Errors to Avoid

| Error | Why It's Wrong | Correct Approach |
|-------|---------------|------------------|
| **1. Truncated bar chart y-axis** | Exaggerates small differences | Always start at zero for bar charts |
| **2. Using pie charts** | Angles harder to compare than lengths | Use bar chart instead |
| **3. Missing units on axes** | Reader can't interpret scale | "Variable (unit)" format required |
| **4. Inconsistent colors across figures** | Confuses readers, breaks semantic meaning | Define color scheme once, use everywhere |
| **5. No sample size reported** | Can't assess reliability | State n in caption or on figure |
| **6. Error bar type undefined** | SEM vs SD gives different interpretation | Always specify "Error bars: SEM" or "SD" |
| **7. Missing colorblind accessibility** | Excludes 8% of male readers | Use Okabe-Ito palette + redundant encoding |
| **8. All-significant data with binary coloring** | Loses magnitude information | Use continuous gradient for effect size |
| **9. Overlapping text labels** | Illegible, unprofessional | Use adjustText (Python) or ggrepel (R) |
| **10. 3D effects on 2D data** | Perspective distortion, looks gimmicky | Use standard 2D with color/size for 3rd variable |

---

## Part VIII: Software-Specific Quick Commands

### Python (Matplotlib) Essential Commands

```
import matplotlib.pyplot as plt

# Global style setup (run once at start)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 9,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'lines.linewidth': 2.5,
    'grid.alpha': 0.3
})

# Okabe-Ito colorblind-safe palette
OKABE_ITO = {
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'bluish_green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermilion': '#D55E00',
    'reddish_purple': '#CC79A7',
    'black': '#000000'
}

# Remove top and right spines (cleaner look)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save at publication quality
plt.savefig('figure.png', dpi=300, bbox_inches='tight', facecolor='white')
```

### R (ggplot2) Essential Commands

```
library(ggplot2)

# Reusable theme
manuscript_theme <- theme_classic(base_size = 11, base_family = 'Arial') +
  theme(
    axis.title = element_text(face = 'bold', size = 11),
    plot.title = element_text(face = 'bold', size = 12, hjust = 0.5),
    axis.text = element_text(size = 9),
    panel.grid.major = element_line(color = 'gray90', size = 0.3),
    plot.tag = element_text(size = 16, face = 'bold')
  )

# Okabe-Ito palette
OKABE_ITO <- c(
  orange = '#E69F00',
  sky_blue = '#56B4E9',
  bluish_green = '#009E73',
  yellow = '#F0E442',
  blue = '#0072B2',
  vermilion = '#D55E00',
  reddish_purple = '#CC79A7',
  black = '#000000'
)

# Apply to plot
p <- ggplot(...) +
  geom_point(color = OKABE_ITO['sky_blue']) +
  manuscript_theme

# Save at publication quality
ggsave('figure.png', p, width = 7, height = 5, dpi = 300, bg = 'white')
```

---

## Part IX: Journal-Specific Considerations

### Nature Family

```
Requirements:
- Figures must be self-explanatory
- Concise but complete captions
- Define all abbreviations
- 300 DPI minimum
- RGB color mode
- TIFF preferred for final submission

Caption style:
- Brief title sentence
- Panel descriptions (a, b, c in lowercase)
- Statistical methods in caption or Methods
- Scale bars required for all images
```

### Cell Family

```
Requirements:
- Similar to Nature
- Emphasize reproducibility (n, replicates clearly stated)
- Can use supplementary figures liberally
- 300-600 DPI
- Color figures encouraged (no extra charge)

Caption style:
- Related panels can share description
- "(A-C)" notation for grouped panels
- Error bar type must be specified
```

### PLOS Family

```
Requirements:
- Very detailed captions encouraged
- Must be understandable without main text
- Can include more methods detail in caption than Nature/Cell
- 300 DPI
- Figures published under CC-BY license (open access)

Caption style:
- More verbose acceptable
- Extensive statistical details in caption
- All abbreviations defined at first use
```

---

## Part X: Emergency Troubleshooting

### "My figure looks cluttered"

```
Solutions:
1. Reduce number of elements (combine categories, show fewer time points)
2. Increase white space (widen margins, reduce bar width)
3. Use small multiples instead of overlaying everything
4. Remove non-essential grid lines
5. Simplify color palette (remove gradients, use solid colors)
6. Increase font sizes (yes, bigger is often clearer)
```

### "Colors don't distinguish well"

```
Solutions:
1. Test with Color Oracle simulator
2. Switch to Okabe-Ito palette
3. Add redundant encoding (shapes, line styles)
4. Increase saturation difference
5. Check contrast ratio (aim for >4.5:1)
6. Avoid red-green combinations
```

### "Text is illegible"

```
Solutions:
1. Increase font size (minimum 9pt for tick labels)
2. Use bold for critical text (axis labels, panel labels)
3. Increase contrast (black text on white background)
4. Remove text overlap (use adjustText or ggrepel)
5. Rotate long axis labels 45° if necessary
6. Simplify category names if too long
```

### "Figure doesn't fit journal specifications"

```
Solutions:
1. Check journal's figure guidelines (width, DPI, format)
2. Redesign for column width (single vs. double column)
3. Reduce number of panels (move some to supplement)
4. Adjust aspect ratio to match journal template
5. Export at exact required dimensions (don't rely on scaling)
```

---

## Part XI: Final Submission Checklist

```
Before submitting manuscript:

Figure Files:
 All figures exported at 300+ DPI
 Correct file format (TIFF/PNG or as specified)
 RGB color mode (unless CMYK required)
 Files named according to journal guidelines
 Correct dimensions (match journal column widths)

Visual Consistency:
 All figures use identical font family and sizes
 Color schemes consistent across all figures
 Panel labels (A, B, C...) in consistent position
 Line widths and marker sizes uniform
 Error bar styles and definitions consistent

Scientific Content:
 All axes have labels with units
 All panels have clear titles or descriptions
 Sample sizes (n) stated in captions
 Statistical methods documented
 Error bar types specified (SEM vs SD)
 Significance markers defined (*, **, ***)
 Scale bars present on all images (microscopy, maps)

Accessibility:
 Colorblind-safe palettes used
 Redundant encoding applied where needed
 Tested with colorblind simulator
 Works in grayscale
 Text contrast sufficient (WCAG AA)

Captions:
 Self-contained (understandable without main text)
 All panels described
 Sample sizes and statistics included
 Abbreviations defined
 Technical details provided (scale bars, magnifications, etc.)

Ethics:
 No image manipulation beyond linear adjustments
 All adjustments applied uniformly to comparison images
 Original unprocessed images available
 Processing documented in Methods
 No misleading color scale manipulation
```

---

## Part XII: Resources and Tools

### Color Tools

```
Palette generators:
- ColorBrewer 2.0: colorbrewer2.org (CVD-safe filter available)
- Coolors: coolors.co (palette generator)
- Adobe Color: color.adobe.com

Accessibility testing:
- Color Oracle: colororacle.org (desktop, free, real-time CVD simulation)
- Coblis: coblis.blogspot.com (web-based, image upload)
- WebAIM Contrast Checker: webaim.org/resources/contrastchecker/

Recommended palettes:
- Okabe-Ito: Universally colorblind-safe (8 colors)
- Viridis: Perceptually uniform, colorblind-safe (sequential)
- RdBu: Diverging, colorblind-safe (with proper use)
```

### Layout and Design References

```
Books:
- "The Visual Display of Quantitative Information" by Edward Tufte
- "Fundamentals of Data Visualization" by Claus O. Wilke (free online)
- "Better Presentations" by Jonathan Schwabish

Online guides:
- Fundamentals of Data Visualization: clauswilke.com/dataviz/
- Data-to-Viz: data-to-viz.com (plot type decision tree)
- From Data to Viz: python graph gallery: python-graph-gallery.com
```

### Software Documentation

```
Python:
- Matplotlib: matplotlib.org/stable/tutorials/
- Seaborn: seaborn.pydata.org/tutorial.html
- Plotly: plotly.com/python/

R:
- ggplot2: ggplot2.tidyverse.org
- patchwork (multi-panel): patchwork.data-imaginist.com
- cowplot (publication-ready): wilkelab.org/cowplot/

Statistics:
- GraphPad Prism tutorials: graphpad.com/guides/
- JASP (free): jasp-stats.org
```

---

## Conclusion: The Path to Publication-Quality Figures

**Remember the hierarchy of priorities:**

1. **Scientific accuracy** (correct data representation)
2. **Clarity** (immediate comprehension of key message)
3. **Accessibility** (colorblind-safe, high contrast, clear labels)
4. **Consistency** (uniform style across manuscript)
5. **Aesthetics** (professional appearance, but never at expense of 1-4)

**When in doubt:**
- Ask: "Does this choice help or hinder understanding?"
- Simplify rather than embellish
- Test on colleagues unfamiliar with your work
- Refer back to journal requirements
- Prioritize data over decoration

**Scientific figures are functional communication tools, not art.**
Every design choice must serve the goal of clear, honest, accessible data presentation.

---
