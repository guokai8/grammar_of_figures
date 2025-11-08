# Chapter 10: Figure Troubleshooting Guide

## 10.1 Diagnostic Framework: "My Figure Looks Wrong"

### Systematic Troubleshooting Approach

**When your figure doesn't look right, follow this diagnostic tree:**

```
STEP 1: Identify the problem category
├─ Technical issues (resolution, colors, file format)
├─ Design issues (layout, spacing, readability)
├─ Data representation issues (plot type, axes, scale)
└─ Compliance issues (journal requirements, ethical standards)

STEP 2: Apply category-specific solutions (see sections below)

STEP 3: Verify fix doesn't create new problems

STEP 4: Document what worked for future reference
```

---

## 10.2 Technical Issues

### Problem 1: "My figure looks blurry,pixelated when zoomed"

**Symptoms:**

- Text appears fuzzy
- Lines have jagged edges
- Image quality degrades when enlarged

**Diagnosis:**

```
from PIL import Image
img = Image.open('your_figure.png')
print(f"Size: {img.size} pixels")
print(f"DPI: {img.info.get('dpi', 'Not set')}")

# Calculate effective DPI
width_inches = 7  # Intended print width
dpi_effective = img.size[0] / width_inches
print(f"Effective DPI at {width_inches}\" width: {dpi_effective:.1f}")

```

**Solutions:**

```
# Solution 1: Re-export at higher DPI
import matplotlib.pyplot as plt

# ... your plotting code ...

# Save at 300 DPI (publication standard)
plt.savefig('figure_300dpi.png', dpi=300, bbox_inches='tight')

# For microscopy/detailed images, use 600 DPI
plt.savefig('figure_600dpi.png', dpi=600, bbox_inches='tight')
```

**Common mistake: Upsampling after creation**

```
# ❌ WRONG: This doesn't add information
from PIL import Image
img = Image.open('low_res.png')
img_upsampled = img.resize((img.width*2, img.height*2), Image.LANCZOS)
# → Still blurry, just larger file

# ✓ CORRECT: Create at target resolution from start
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)  # Set DPI at creation
```

---

### Problem 2: "Colors look different on screen vs print"

Cause: RGB (screen) vs CMYK (print) color space mismatch

**Diagnosis:**

```
from PIL import Image
img = Image.open('figure.png')
print(f"Color mode: {img.mode}")
# RGB = screen colors, CMYK = print colors
```

**Solutions:**

```
# Preventive: Use print-safe RGB colors from the start
# Avoid highly saturated colors that can't be printed

PRINT_SAFE_PALETTE = {
    'blue': '#0066CC',      # Instead of pure #0000FF
    'red': '#CC0000',       # Instead of pure #FF0000
    'green': '#009900',     # Instead of pure #00FF00
    'orange': '#CC6600',
    'purple': '#9933CC'
}

# Test colors in grayscale (simulate print preview)
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Original colors
data = np.random.rand(5, 5)
im1 = axes[0].imshow(data, cmap='viridis')
axes[0].set_title('Original (RGB)')

# Grayscale preview (simulates black & white print)
im2 = axes[1].imshow(data, cmap='gray')
axes[1].set_title('Grayscale Preview')

plt.tight_layout()
plt.savefig('color_test.png', dpi=300)
plt.close()
```

**Journal-specific fix:**

```
# Some journals require CMYK conversion
# Do this ONLY if explicitly required

from PIL import Image

# Convert RGB to CMYK
img_rgb = Image.open('figure_rgb.png')
img_cmyk = img_rgb.convert('CMYK')
img_cmyk.save('figure_cmyk.tif')  # TIFF supports CMYK

print("⚠ Warning: Colors may shift during conversion")
print("→ Preview before submission")
```

---

### Problem 3: "File size too large for submission"

**Diagnosis:**

```
import os

file_path = 'large_figure.png'
size_mb = os.path.getsize(file_path) / (1024 * 1024)
print(f"Current file size: {size_mb:.2f} MB")

# Check journal limit (common limits: 5-20 MB per figure)
limit_mb = 10
if size_mb > limit_mb:
    print(f"❌ Exceeds {limit_mb} MB limit")
```

**Solutions (in order of preference):**

```
# Solution 1: Optimize compression (lossless)
from PIL import Image

img = Image.open('large_figure.png')

# PNG optimization
img.save('optimized.png', optimize=True, compress_level=9)

# Check new size
new_size = os.path.getsize('optimized.png') / (1024 * 1024)
print(f"Optimized size: {new_size:.2f} MB ({(1-new_size/size_mb)*100:.1f}% reduction)")

# Solution 2: Reduce dimensions (if too large)
# Only if figure dimensions exceed journal specifications
target_width = 7  # inches
current_dpi = 300
img_resized = img.resize((int(target_width * current_dpi),
                         int(target_width * current_dpi * img.height / img.width)),
                        Image.LANCZOS)
img_resized.save('resized.png', optimize=True)

# Solution 3: Use TIFF with LZW compression (lossless)
img.save('compressed.tif', compression='tiff_lzw')

# Solution 4: Split into multiple panels (if complex)
# Save panels separately, combine in manuscript text
```

### Problem 4: "Fonts look different after export"

**Cause:** Font embedding issues

**Solutions:**

```
# Matplotlib: Ensure fonts are embedded
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42  # TrueType (embeds fonts)
plt.rcParams['ps.fonttype'] = 42

# Or rasterize text (converts to pixels, always works)
plt.savefig('figure.pdf', dpi=300, bbox_inches='tight')  # Text as vectors

# If font still missing, use rasterization
plt.savefig('figure.png', dpi=300, bbox_inches='tight')  # Text as pixels
```

**R:**

```
library(ggplot2)

# Ensure fonts are embedded in PDF
ggsave('figure.pdf', plot,
       width = 7, height = 5,
       device = cairo_pdf,  # Uses Cairo for better font handling
       dpi = 300)

# Or save as PNG (rasterizes everything)
ggsave('figure.png', plot,
       width = 7, height = 5,
       dpi = 300)
```

**Fallback: Use system fonts only**

```
# If custom fonts cause issues, stick to universals
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
# System will use first available font

```

---

## 10.3 Design Issues

### Problem 5: "Text is overlapping and illegible"

**Common in:** Scatter plots with many labels, crowded axes

**Solutions:**

```
# Solution 1: Use adjustText library (automatic label adjustment)
from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.random.rand(30)
y = np.random.rand(30)
labels = [f'Gene_{i}' for i in range(30)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Without adjustment (overlapping)
ax1 = axes[0]
ax1.scatter(x, y, s=50, color='#3498DB', alpha=0.7)
for i, label in enumerate(labels):
    ax1.text(x[i], y[i], label, fontsize=8)
ax1.set_title('❌ Without Adjustment\n(Overlapping labels)', color='red', fontweight='bold')

# With adjustment (clean)
ax2 = axes[1]
ax2.scatter(x, y, s=50, color='#3498DB', alpha=0.7)
texts = [ax2.text(x[i], y[i], label, fontsize=8) for i, label in enumerate(labels)]
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
ax2.set_title('✓ With Adjustment\n(Clear labels)', color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('label_overlap_fix.png', dpi=300, bbox_inches='tight')
plt.close()
```

**R equivalent:**

```
library(ggplot2)
library(ggrepel)  # Automatic label repelling

# Without repelling (overlapping)
p1 <- ggplot(data, aes(x = x, y = y)) +
  geom_point(size = 3, color = '#3498DB', alpha = 0.7) +
  geom_text(aes(label = label), size = 3) +  # Overlaps
  labs(title = '❌ Without Repelling')

# With repelling (clean)
p2 <- ggplot(data, aes(x = x, y = y)) +
  geom_point(size = 3, color = '#3498DB', alpha = 0.7) +
  geom_text_repel(aes(label = label), size = 3,  # Automatic adjustment
                  box.padding = 0.5, max.overlaps = 20) +
  labs(title = '✓ With Repelling')

library(patchwork)
p1 | p2
ggsave('label_overlap_fix.png', width = 14, height = 6, dpi = 300)
```

**Solution 2: Selective labeling**

```
# Label only significant/important points
significant_idx = [0, 5, 12, 20, 28]  # Indices of important points

ax.scatter(x, y, s=50, color='#3498DB', alpha=0.7)
for i in significant_idx:
    ax.annotate(labels[i], (x[i], y[i]),
               xytext=(5, 5), textcoords='offset points',
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               arrowprops=dict(arrowstyle='->', lw=1))
```

---

### Problem 6: "Figure looks cluttered/too busy"

**Symptoms:**

- Hard to identify key message
- Too many colors/patterns
- Dense gridlines overwhelm data

**Solutions:**

```
# Decluttering checklist:

# 1. Remove unnecessary grid lines
ax.grid(False)  # Remove all grids
# Or: Use subtle grid
ax.grid(axis='y', alpha=0.3, linewidth=0.5, color='gray')

# 2. Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 3. Reduce color palette
# ❌ BAD: Too many colors
colors = plt.cm.tab20(np.linspace(0, 1, 15))  # 15 different colors!

# ✓ GOOD: 3-4 colors maximum
colors = ['#7F8C8D', '#3498DB', '#E74C3C']  # Gray, blue, red

# 4. Increase white space
plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)

# 5. Use small multiples instead of overlaying everything
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Split into 4 panels
# Instead of cramming all data into one plot

# 6. Remove redundant legends
# If panel labels (A, B, C) are clear, legend may not be needed
```

**Before/After Example:**

```
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.linspace(0, 10, 100)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# CLUTTERED VERSION
ax1 = axes[0]
for i in range(10):
    y = np.sin(x + i/2) + np.random.rand(100)*0.1
    ax1.plot(x, y, linewidth=1, label=f'Series {i+1}')
ax1.grid(True, linewidth=1, alpha=0.7)  # Heavy grid
ax1.legend(loc='best', fontsize=8, ncol=2)
ax1.set_title('❌ Cluttered: Too Many Elements', color='red', fontweight='bold')

# CLEAN VERSION
ax2 = axes[1]
# Show only 3 key series
for i in [0, 4, 9]:
    y = np.sin(x + i/2) + np.random.rand(100)*0.1
    ax2.plot(x, y, linewidth=2.5, label=f'Series {i+1}', alpha=0.8)
# Others in background (gray)
for i in [1, 2, 3, 5, 6, 7, 8]:
    y = np.sin(x + i/2) + np.random.rand(100)*0.1
    ax2.plot(x, y, color='#CCCCCC', linewidth=0.8, alpha=0.3)
ax2.grid(axis='y', alpha=0.3, linewidth=0.5)  # Subtle grid
ax2.legend(loc='upper right', frameon=True, fontsize=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_title('✓ Clean: Focused Message', color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('declutter_fix.png', dpi=300, bbox_inches='tight')
plt.close()
```

---

### Problem 7: "Panels are misaligned"

**Cause:** Inconsistent axis ranges, label sizes, or spacing

**Solutions:**

```
# Solution 1: Use consistent axis limits
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Set same limits for all comparable plots
xlim_range = (0, 10)
ylim_range = (0, 100)

for ax in axes.flat:
    # ... plot data ...
    ax.set_xlim(xlim_range)
    ax.set_ylim(ylim_range)

# Solution 2: Use GridSpec for precise control
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, figure=fig,
             width_ratios=[1, 1], height_ratios=[1, 1],
             hspace=0.3, wspace=0.3)  # Consistent spacing

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Solution 3: Match axis label sizes
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlabel('X Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y Label', fontsize=11, fontweight='bold')
    ax.tick_params(labelsize=9)
```

**R equivalent:**

```
library(ggplot2)
library(patchwork)

# Create plots with consistent limits
p1 <- ggplot(data1, aes(x, y)) + geom_point() +
  xlim(0, 10) + ylim(0, 100)

p2 <- ggplot(data2, aes(x, y)) + geom_point() +
  xlim(0, 10) + ylim(0, 100)  # Same limits

p3 <- ggplot(data3, aes(x, y)) + geom_point() +
  xlim(0, 10) + ylim(0, 100)

p4 <- ggplot(data4, aes(x, y)) + geom_point() +
  xlim(0, 10) + ylim(0, 100)

# Combine with patchwork (auto-aligns)
(p1 | p2) / (p3 | p4) +
  plot_layout(heights = c(1, 1), widths = c(1, 1))
```

---

## 10.4 Data Representation Issues

### Problem 8: "My bar chart doesn't show the difference clearly"

**Diagnosis:** Likely truncated y-axis or wrong plot type

**Solutions:**

```
import matplotlib.pyplot as plt
import numpy as np

data = [98, 100, 102, 104]  # Small differences
categories = ['A', 'B', 'C', 'D']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# WRONG: Truncated axis (exaggerates difference)
ax1 = axes[0]
ax1.bar(categories, data, color='#3498DB', edgecolor='black', linewidth=1.5)
ax1.set_ylim(95, 105)  # Truncated!
ax1.set_ylabel('Value')
ax1.set_title('❌ Misleading: Truncated Axis', color='red', fontweight='bold')
ax1.text(0.5, 0.5, 'Exaggerates\ndifferences!', transform=ax1.transAxes,
        ha='center', fontsize=12, color='red', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# CORRECT: Start at zero (shows true scale)
ax2 = axes[1]
ax2.bar(categories, data, color='#3498DB', edgecolor='black', linewidth=1.5)
ax2.set_ylim(0, 110)  # Starts at zero
ax2.set_ylabel('Value')
ax2.set_title('✓ Correct: Zero Baseline', color='green', fontweight='bold')

# ALTERNATIVE: Dot plot with range (if differences are small but real)
ax3 = axes[2]
ax3.plot(categories, data, 'o', markersize=12, color='#3498DB')
ax3.set_ylim(95, 105)  # Can use truncated axis for dot plots
ax3.axhline(100, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Baseline')
ax3.set_ylabel('Value')
ax3.set_title('✓ Alternative: Dot Plot\n(Truncation OK here)', color='green', fontweight='bold')
ax3.legend()

plt.tight_layout()
plt.savefig('bar_chart_fix.png', dpi=300, bbox_inches='tight')
plt.close()
```

**Key rule:**

```
Bar charts: ALWAYS start at zero
Dot plots/line plots: Truncation acceptable if clearly indicated
```

---

### Problem 9: "P-values overlap or are unreadable"

**Common in:** Plots with many statistical comparisons

**Solutions:**

```
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

categories = ['A', 'B', 'C', 'D', 'E']
values = [25, 32, 28, 40, 35]
errors = [3, 4, 3, 5, 4]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# BAD: All comparisons shown (cluttered)
ax1 = axes[0]
bars = ax1.bar(categories, values, yerr=errors, capsize=5,
              color='#3498DB', edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Response (AU)', fontsize=11, fontweight='bold')
ax1.set_ylim(0, 60)

# Draw all pairwise comparisons (too many!)
y_max = 50
for i in range(len(categories)-1):
    y_max += 3
    ax1.plot([i, i+1], [y_max, y_max], 'k-', linewidth=1.5)
    ax1.text((i + i+1)/2, y_max + 0.5, '**', ha='center', fontsize=10)

ax1.set_title('❌ Cluttered: Too Many Comparisons', color='red', fontweight='bold')

# GOOD: Show only key comparisons
ax2 = axes[1]
bars = ax2.bar(categories, values, yerr=errors, capsize=5,
              color='#3498DB', edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Response (AU)', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 55)

# Show only comparisons vs. control (A) or most important
# Comparison: D vs A
ax2.plot([0, 3], [48, 48], 'k-', linewidth=2)
ax2.text(1.5, 49, '** p<0.01', ha='center', fontsize=11, fontweight='bold')

# Comparison: E vs A
ax2.plot([0, 4], [52, 52], 'k-', linewidth=2)
ax2.text(2, 53, '* p<0.05', ha='center', fontsize=11, fontweight='bold')

ax2.set_title('✓ Clean: Key Comparisons Only', color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('statistical_annotation_fix.png', dpi=300, bbox_inches='tight')
plt.close()

# Use letters instead of brackets (common in biology)
ax.bar(categories, values, ...)

# Add letters above bars
letters = ['a', 'ab', 'ab', 'c', 'bc']  # Different letters = significant difference
for i, (bar, letter) in enumerate(zip(bars, letters)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + errors[i] + 1,
           letter, ha='center', va='bottom', fontsize=12, fontweight='bold')

# Caption: "Bars with different letters are significantly different (p<0.05, Tukey HSD)"
```

---

### Problem 10: "Data points are too small or large"

**Solutions:**

```
# Guideline: Marker sizes for readability
recommended_sizes = {
    'few_points': (10, 50),      # <20 points: 50-150 in plt units
    'moderate': (50, 100),       # 20-100 points: 30-80
    'many': (100, 500),          # 100-500 points: 20-50
    'dense': (500, 1000),        # >500 points: 10-30 or use hexbin
    'very_dense': (1000, 10000)  # >1000 points: 5-15 or 2D density
}

# Adjust based on figure size
fig_inches = 7
dpi = 300
fig_pixels = fig_inches * dpi

# Rule of thumb: marker diameter should be 1-3% of figure dimension
marker_size_min = (0.01 * fig_pixels)**2  # matplotlib uses area, not diameter
marker_size_max = (0.03 * fig_pixels)**2

print(f"Recommended marker size range: {marker_size_min:.0f} - {marker_size_max:.0f}")

# Example
ax.scatter(x, y, s=50, ...)  # s = area in points^2
```

---

## 10.5 Compliance Issues

### Problem 11: "Journal rejected my figure for image manipulation"

**Common violations:**
1. Selective brightness/contrast adjustment
2. Lane splicing in Western blots without disclosure
3. Background removal
4. Duplicated/cloned elements

**Prevention:**

```
# Document ALL processing steps
"""
Image Processing Protocol (for Methods section):

1. Microscopy images acquired with:
   - Microscope: [Model]
   - Objective: [Magnification, NA]
   - Camera: [Model]
   - Exposure: [milliseconds]
   - Gain: [value]

2. Post-acquisition processing:
   - Software: ImageJ v1.53
   - Brightness: +10% (applied uniformly to all images)
   - Contrast: Linear adjustment (min=50, max=200, applied uniformly)
   - No gamma adjustment
   - No background subtraction
   - No cloning or content-aware fill

3. Cropping:
   - Representative regions shown
   - Original unprocessed images available upon request

4. Figure assembly:
   - Software: Adobe Illustrator CC 2024
   - No modifications beyond cropping and labeling
"""

# Save original unprocessed images separately
# Name: raw_image_01.tif, raw_image_02.tif, ...
# Keep permanently for potential requests from journal/reviewers
```

---

### Problem 12: "Figure doesn't meet journal specifications"

**Systematic check:**

```
def validate_figure_specs(file_path, journal_specs):
    """
    Validate figure against journal requirements

    journal_specs = {
        'max_width_inches': 7,
        'min_dpi': 300,
        'allowed_formats': ['TIFF', 'PNG', 'PDF'],
        'max_file_size_mb': 10,
        'color_mode': 'RGB'
    }
    """
    from PIL import Image
    import os

    img = Image.open(file_path)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    issues = []

    # Check format
    if img.format not in journal_specs['allowed_formats']:
        issues.append(f"❌ Format {img.format} not allowed. Use: {journal_specs['allowed_formats']}")

    # Check DPI
    dpi = img.info.get('dpi', (72, 72))[0]
    if dpi < journal_specs['min_dpi']:
        issues.append(f"❌ DPI {dpi} < required {journal_specs['min_dpi']}")

    # Check dimensions
    width_inches = img.size[0] / dpi
    if width_inches > journal_specs['max_width_inches']:
        issues.append(f"❌ Width {width_inches:.2f}\" > max {journal_specs['max_width_inches']}\"")

    # Check file size
    if file_size_mb > journal_specs['max_file_size_mb']:
        issues.append(f"❌ File size {file_size_mb:.2f} MB > max {journal_specs['max_file_size_mb']} MB")

    # Check color mode
    if img.mode != journal_specs['color_mode']:
        issues.append(f"⚠ Color mode {img.mode}, required: {journal_specs['color_mode']}")

    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✓ All specifications met")
        return True

# Example usage
nature_specs = {
    'max_width_inches': 7.2,
    'min_dpi': 300,
    'allowed_formats': ['TIFF', 'PDF', 'EPS'],
    'max_file_size_mb': 10,
    'color_mode': 'RGB'
}

validate_figure_specs('my_figure.png', nature_specs)
```

---

## 10.6 Quick Fix Cheat Sheet

**One-line fixes for common problems:**

```
# Problem: Text too small
plt.rcParams['font.size'] = 11  # Increase base font size

# Problem: Legend overlaps data
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))  # Outside plot area

# Problem: Margins too tight
plt.tight_layout(pad=2.0)  # Increase padding

# Problem: Axes labels cut off
plt.savefig('figure.png', bbox_inches='tight')  # Auto-adjust bounds

# Problem: Colorbar too wide
plt.colorbar(fraction=0.046, pad=0.04)  # Standard narrow colorbar

# Problem: Tick labels overlap
ax.tick_params(axis='x', rotation=45)  # Rotate 45°

# Problem: Grid too prominent
ax.grid(alpha=0.3, linewidth=0.5)  # Subtle grid

# Problem: Too much white space
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

# Problem: Inconsistent line widths
ax.plot(x, y, linewidth=2.5)  # Standard data line
ax.spines['left'].set_linewidth(1.5)  # Standard axis line

# Problem: Can't see error bars
ax.errorbar(x, y, yerr=errors, capsize=6, capthick=2, linewidth=2)
```

---

**End of Chapter 10: Figure Troubleshooting Guide**

**Emergency Checklist (use when deadline is tight):**

```
 DPI ≥ 300 (check with print preview at 100%)
 File format correct for journal (usually TIFF/PNG)
 Color mode RGB (unless CMYK explicitly required)
 File size < journal limit
 All text readable at intended print size
 Axes labeled with units
 Panel labels (A, B, C...) present
 Scale bars on all microscopy images
 Statistical annotations clear
 Color-blind safe palette used
 Figure works in grayscale
 Caption complete (methods, n, statistics)
 Consistent style across all manuscript figures
```

---


# MASTER CHECKLIST: Publication-Quality Scientific Figures

**Version 1.0 | Complete Reference Guide**

---

## SECTION 1: PRE-DESIGN PLANNING

### 1.1 Define Your Message
```
 One key finding per figure identified
 Target audience defined (specialists vs. general)
 Primary comparison clearly stated
 Figure type selected based on data structure
```

### 1.2 Data Assessment
```
 Data type classified (continuous, categorical, temporal, spatial)
 Sample sizes documented (n for each group)
 Distribution assessed (normal, skewed, multimodal)
 Missing data identified and handling method determined
 Statistical tests planned
```

### 1.3 Journal Requirements Research
```
 Submission guidelines read carefully
 Figure dimensions noted (single/double column width)
 Required DPI documented
 File format requirements noted
 File size limits recorded
 Color mode specified (RGB vs CMYK)
 Special requirements noted (e.g., separate panel files)
```

---

## SECTION 2: COLOR DESIGN

### 2.1 Color Palette Selection
```
 Palette type matches data:
   • Sequential (ordered, one direction): Viridis, Blues, YlOrRd
   • Diverging (ordered, two directions): RdBu, PiYG, BrBG
   • Categorical (unordered): Okabe-Ito, Set2, Dark2

 Colorblind-safe palette used (test with Color Oracle)
 Maximum 3-5 colors for categorical data
 Semantic consistency applied:
   • Control always same color across figures
   • Treatment A always same color across figures
   • Statistical significance encoded consistently
```

### 2.2 Color Accessibility
```
 Tested with colorblind simulator (Color Oracle, Coblis)
 Works in grayscale (print test performed)
 Redundant encoding added where critical (color + shape/line)
 Text contrast meets WCAG AA (4.5:1 minimum)
 No red-green combinations as sole distinguisher
```

### 2.3 Color Scale Integrity
```
 Symmetric diverging scales (e.g., -3 to +3, center at 0)
 Colorbar included with units
 Missing data encoded distinctly (not from main palette)
 No rainbow colormap (jet, hsv) used
 Perceptually uniform colormap used for sequential data
```

---

## SECTION 3: TYPOGRAPHY & LABELS

### 3.1 Font Specifications
```
 Font family consistent across all figures: Arial/Helvetica
 Font hierarchy implemented:
   • Panel labels: 14-16pt, bold
   • Titles: 12-13pt, bold
   • Axis labels: 11pt, bold
   • Tick labels: 9pt, regular
   • Legend: 9pt, regular
   • Annotations: 9-10pt, regular/italic
```

### 3.2 Axis Labels
```
 All axes labeled with variable name and units: "Variable (unit)"
 Examples checked:
   ✓ "Temperature (°C)" not "Temperature"
   ✓ "Time (hours)" not "Time"
   ✓ "Expression (FPKM)" not "Expression"
   ✓ "Fold Change (log₂)" not "Fold Change"
```

### 3.3 Statistical Annotations
```
 Notation consistent:
   • * p < 0.05
   • ** p < 0.01
   • *** p < 0.001
   • n.s. p ≥ 0.05
 Statistical test stated in caption
 Exact p-values provided when critical
 Sample sizes (n) stated
```

### 3.4 Text Readability
```
 No overlapping labels (adjustText or ggrepel used)
 All text readable at intended print size
 High contrast: black text on white background
 No text smaller than 6pt after reduction
 Bold used for emphasis (axis labels, panel labels)
```

---

## SECTION 4: PLOT TYPE SELECTION

### 4.1 Comparing Groups
```
Data: Few groups (<5), continuous variable
 Bar chart selected (if showing means/totals)
 Box plot selected (if showing distributions)
 Violin plot selected (if distributions complex/bimodal)
 Zero baseline used for bar charts
 Error bars included (SEM or SD specified)
 Individual data points overlaid (if n small)
```

### 4.2 Distributions
```
Data: Distribution of continuous variable
 Histogram used (with appropriate bin width)
 Density plot used (for smooth estimate)
 Box/violin plot used (for compact view)
 Bin width justified (Sturges', Freedman-Diaconis, or manual)
```

### 4.3 Relationships
```
Data: Two continuous variables
 Scatter plot used (always plot raw data)
 Regression line added if appropriate
 Correlation coefficient reported
 Hexbin or 2D density used if >1000 points
```

### 4.4 Time Series
```
Data: Variable measured over time
 Line graph used (implies continuity)
 Small multiples used if >5 series
 Shaded error bands used (cleaner than error bars)
 Time axis starts at meaningful baseline
```

### 4.5 What to AVOID
```
 NO pie charts (use bar chart instead)
 NO 3D effects on 2D data
 NO dual y-axes (unless absolutely justified)
 NO truncated bar charts (always start at zero)
 NO rainbow colormaps (jet, hsv)
```

---

## SECTION 5: LAYOUT & COMPOSITION

### 5.1 Panel Arrangement
```
 Layout strategy chosen:
   • Equal weight: All panels equally important (2×2, 3×3)
   • Dominant: Main panel (60-70%) + supporting (15-20%)
   • Sequential: Left-to-right or top-to-bottom flow

 Gestalt principles applied:
   • Proximity: Related panels close together
   • Similarity: Same colors for same groups
   • Closure: Boxes/borders for grouping
```

### 5.2 Aspect Ratios
```
 Ratio matches data structure:
   • Time series: 16:9 or 3:1 (wide)
   • Comparisons: 4:3 or 3:2 (standard)
   • Heatmaps: 1:1 or data-dependent (square)
   • Trees: 1:2 (tall)
```

### 5.3 White Space
```
 White space: 40-60% of figure area
 Margins adequate:
   • Outer margin: 0.75-1 inch
   • Between panels: 0.25-0.5 inch (related), 0.75-1 inch (separate)
 Not too cramped (<30% white space)
 Not too sparse (>70% white space)
```

### 5.4 Panel Labels
```
 All panels labeled: A, B, C...
 Labels in consistent position (top-left or top-right)
 Labels bold, large (14-16pt)
 Labels easily visible (white on dark, black on light)
```

---

## SECTION 6: TECHNICAL SPECIFICATIONS

### 6.1 Resolution
```
 DPI appropriate for content:
   • Line art: 600-1000 DPI
   • Photos/microscopy: 300-600 DPI
   • Combination: 600 DPI
 Created at target resolution (not upsampled later)
 Effective DPI calculated: pixels / intended width (inches)
```

### 6.2 File Format
```
 Format matches journal requirements:
   • TIFF: Publication standard (lossless)
   • PNG: Good alternative (lossless, smaller files)
   • PDF: Vector graphics
   • EPS: Legacy vector format
 NO JPEG used (lossy compression)
 Color mode correct: RGB (most common)
```

### 6.3 Dimensions
```
 Width matches journal specifications:
   • Single column: typically 3.5" (89mm)
   • Double column: typically 7" (178mm)
 Height within limits (often ≤10")
 Aspect ratio appropriate for data
```

### 6.4 File Size
```
 File size < journal limit (often 5-20 MB)
 Optimization applied if needed:
   • PNG with compress_level=9
   • TIFF with LZW compression
   • Dimensions not larger than necessary
 Fallback: Split into multiple files if too large
```

---

## SECTION 7: FIELD-SPECIFIC REQUIREMENTS

### 7.1 Microscopy Images
```
 Scale bar present on ALL images
 Scale bar sized appropriately (10 µm, 50 µm, 100 µm)
 Scale bar color contrasts with image (white on dark, black on light)
 Channel labels included for fluorescence:
   • Channel name (e.g., "DAPI")
   • Wavelength (e.g., "405 nm")
   • Merge clearly labeled
 Image processing documented:
   • Microscope model and settings
   • All adjustments listed
   • Uniform processing applied
 "Representative images shown" stated in caption
 Quantification included (n fields/images analyzed)
```

### 7.2 Western Blots
```
 Full lanes visible (no splicing without disclosure)
 Molecular weight markers labeled
 Loading control included (β-actin, GAPDH, etc.)
 Loading control in SAME sample order
 Quantification bar chart included
 Normalized to loading control
 Error bars from biological replicates (n≥3)
 Statistical test stated
 "Representative blot from X experiments" in caption
 Original unprocessed images available
```

### 7.3 Flow Cytometry
```
 Axes labeled: Marker name + fluorophore (e.g., "CD4-FITC")
 Scale type indicated (linear or logarithmic)
 Gates clearly visible (red lines typical)
 Population percentages shown
 Total cell count stated (n = X cells)
 Gating strategy defined in Methods or caption
 Compensation described in Methods
```

### 7.4 Phylogenetic Trees
```
 Scale bar with units (substitutions/site, years)
 Bootstrap or posterior probability values at nodes
 Branch lengths proportional to distance
 Root indicated (outgroup stated)
 Tip labels clear
 Tree-building method stated in caption
```

### 7.5 Network Diagrams
```
 Node encoding defined (size, color, shape)
 Edge encoding defined (width, color, style)
 Layout algorithm stated
 Network statistics reported (nodes, edges, density)
 Labels non-overlapping
 Not too dense (hairball avoided)
```

---

## SECTION 8: SPECIALIZED PLOT TYPES

### 8.1 Heatmaps
```
 Colormap matches data type:
   • Sequential: Viridis, Plasma (one direction)
   • Diverging: RdBu, PiYG (two directions from center)
 Symmetric diverging scale if applicable (-3 to +3, center at 0)
 Colorbar present with label and units
 Normalization stated (row Z-score, raw values, etc.)
 Clustering method stated if used
 Dendrogram shown if hierarchical clustering
 NO rainbow colormap
```

### 8.2 Volcano Plots
```
 X-axis: Log₂ fold change
 Y-axis: -Log₁₀ p-value
 BOTH fold change AND p-value thresholds shown
 Threshold lines visible (typically |log₂FC| > 1, p < 0.05)
 Three color groups:
   • Not significant (gray)
   • Upregulated significant (red)
   • Downregulated significant (blue)
 Gene counts per category in legend
```

### 8.3 PCA Plots
```
 Variance explained in axis labels: "PC1 (45.3% variance)"
 Color/shape encodes biological variable
 95% confidence ellipses shown for groups
 Scree plot included (shows variance by PC)
 Grid lines at x=0, y=0
 Legend with sample sizes
```

### 8.4 Survival Curves (Kaplan-Meier)
```
 Y-axis: 0 to 1 (survival probability), starts at 1.0
 X-axis: Time, starts at 0
 Step function (not smooth curve)
 95% confidence bands shown (shaded or dashed)
 Censored data marked (tick marks or + symbols)
 Log-rank test p-value reported
 Median survival times noted
 Number at risk table (or stated in caption)
```

### 8.5 ROC Curves
```
 X-axis: False Positive Rate (1 - Specificity)
 Y-axis: True Positive Rate (Sensitivity)
 Diagonal reference line (y=x) shown
 AUC with 95% confidence interval reported
 Square aspect ratio (equal axes)
 Optimal operating point marked (if applicable)
 Sample size stated (n positives, n negatives)
```

---

## SECTION 9: CAPTIONS

### 9.1 Caption Structure
```
 Format: [Figure #]. [One-sentence summary].
   (A) [Panel A description]. (B) [Panel B description]. ...
   [Error bar definition]. [Statistical methods]. [Abbreviations].

 Caption is self-contained (understandable without main text)
 All panels described (A, B, C...)
 Sample sizes stated for each group (n = X)
 Error bar type specified (SEM or SD)
 Statistical methods documented:
   • Test used (t-test, ANOVA, etc.)
   • Significance thresholds (*, **, ***)
   • Post-hoc tests if applicable
 Abbreviations defined at first use
 Technical details included:
   • Microscopy: scale bars, magnifications
   • Blots: antibodies, molecular weights
   • Flow: cell counts, gating strategy
```

---

## SECTION 10: CROSS-FIGURE CONSISTENCY

### 10.1 Manuscript-Wide Checks
```
 Font family identical across all figures
 Font sizes consistent (panel labels, axes, ticks)
 Color scheme consistent:
   • Control always same color
   • Treatment A always same color
   • Treatment B always same color
 Line widths uniform (data: 2-3pt, axes: 1-1.5pt)
 Marker sizes uniform
 Panel label format consistent (position, size)
 Error bar style consistent (all SEM or all SD)
 Statistical notation consistent (*, **, ***)
 Grid styles consistent (if used)
 All figures export at same DPI (300 minimum)
 File formats consistent
```

### 10.2 Style Guide Documentation
```
 Style guide created and saved:
   • Font family and sizes
   • Color palette (hex codes)
   • Line widths
   • Marker sizes
   • Aspect ratios by plot type
 Template files created (Python rcParams, R themes)
 Version control for figures:
   • File naming: figure1_v1_20250108.png
   • Change log maintained
   • Original high-res files saved separately
```

---

## SECTION 11: ETHICAL COMPLIANCE

### 11.1 Image Integrity
```
 NO selective brightness/contrast adjustments
 All adjustments applied uniformly to comparison images
 Linear adjustments only (no gamma without justification)
 NO background removal beyond uniform adjustments
 NO cloning or content-aware fill
 NO selective cropping to remove unwanted features
 Original unprocessed images saved and available
 All processing documented in Methods section
```

### 11.2 Data Integrity
```
 NO data point removal without justification
 Outliers handled transparently (method stated)
 NO selective reporting (all replicates shown or summarized)
 NO p-hacking (multiple testing corrected)
 NO misleading truncation of axes (bar charts start at zero)
 NO manipulation of scales to exaggerate effects
```

### 11.3 Documentation
```
 Methods section documents:
   • Image acquisition settings
   • Statistical tests and software
   • Sample sizes and replicates
   • Inclusion/exclusion criteria
   • All image processing steps
 Raw data available (in supplement or upon request)
 Code available (if computational)
```

---

## SECTION 12: PRE-SUBMISSION FINAL CHECKS

### 12.1 Technical Quality
```
 Print test at 100% scale performed (readability check)
 DPI verified (≥300 for publication)
 File format correct for journal
 File size < journal limit
 Color mode correct (RGB unless CMYK specified)
 All fonts embedded (if PDF/EPS)
 No compression artifacts (JPEG avoided)
 Files named per journal convention
```

### 12.2 Visual Quality
```
 All text readable at intended size
 No overlapping labels
 Colors distinguish well
 Grid lines subtle (not overwhelming)
 White space balanced (40-60%)
 Panels aligned properly
 Consistent styling across all manuscript figures
```

### 12.3 Content Completeness
```
 All axes labeled with units
 All panels labeled (A, B, C...)
 Scale bars on all microscopy images
 Molecular weight markers on all blots
 Statistical annotations clear
 Legends complete
 Captions self-contained
 Supplementary figures numbered separately
```

### 12.4 Accessibility
```
 Colorblind-safe palette used
 Tested with colorblind simulator
 Works in grayscale
 Redundant encoding present (color + shape/line)
 Text contrast sufficient (WCAG AA)
 Alt text prepared (if online publication)
```

### 12.5 Journal-Specific
```
 Dimensions match specifications
 Resolution meets requirements
 File format allowed
 File size within limits
 Special requirements met (e.g., separate panel files)
 Submission checklist completed
```

---

## SECTION 13: INTERACTIVE FIGURES (if applicable)

### 13.1 Interactive Figure Requirements
```
 Static fallback version created (for print/PDF)
 Alt text provided (describes figure content)
 Colorblind-safe palette used (same as static)
 Tested on multiple browsers (Chrome, Firefox, Safari)
 Download option for underlying data provided
 Software/library versions documented in Methods
 File size optimized (<10 MB target for web)
```

### 13.2 Caption for Interactive
```
 Instructions included:
   • How to interact (hover, click, drag)
   • What features are available (zoom, pan, filter)
   • URL to interactive version
 Static version described: "Static version shows [state]"
 Data availability noted (supplementary table, etc.)
```

---

## SECTION 14: TROUBLESHOOTING QUICK FIXES

### 14.1 One-Line Fixes
```
# Text too small
plt.rcParams['font.size'] = 11

# Legend overlaps data
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

# Margins too tight
plt.tight_layout(pad=2.0)

# Axes labels cut off
plt.savefig('figure.png', bbox_inches='tight')

# Tick labels overlap
ax.tick_params(axis='x', rotation=45)

# Grid too prominent
ax.grid(alpha=0.3, linewidth=0.5)

# Inconsistent line widths
ax.plot(x, y, linewidth=2.5)  # Data lines
ax.spines['left'].set_linewidth(1.5)  # Axes

# Error bars not visible
ax.errorbar(x, y, yerr=err, capsize=6, capthick=2, linewidth=2)
```

### 14.2 Common Problems → Solutions
```
Blurry figure → Re-export at 300+ DPI (don't upsample)
Colors look different in print → Use print-safe RGB palette
File too large → Optimize compression, reduce dimensions
Fonts missing → Embed fonts (fonttype=42) or use system fonts
Labels overlapping → Use adjustText (Python) or ggrepel (R)
Figure cluttered → Reduce elements, increase white space
Panels misaligned → Use consistent limits and GridSpec
Bar chart misleading → Always start at zero
P-values unreadable → Show only key comparisons or use letter notation
Journal rejection → Document all processing, provide originals
```

---

## SECTION 15: MANUSCRIPT SUBMISSION CHECKLIST

### 15.1 All Figures
```
 Figures numbered sequentially (Figure 1, 2, 3...)
 All figures cited in text in order
 All figure files in correct format
 All figure files named correctly (e.g., Figure1.tif)
 All figure captions in separate document (if required)
 Supplementary figures numbered separately (Figure S1, S2...)
 High-resolution originals saved separately
 Figure permissions obtained (if reusing published material)
```

### 15.2 Cover Letter
```
 Figure creation software stated
 Statistical software stated
 Any special considerations mentioned (e.g., large file sizes)
 Confirmation of original data and no manipulation
```

### 15.3 Methods Section
```
 Figure creation software and versions
 Statistical tests with software
 Image acquisition parameters
 Image processing steps
 Color encodings explained
 Any custom code deposited (GitHub, Zenodo)
```

---

## SECTION 16: POST-SUBMISSION

### 16.1 Revision Response
```
 Original figure files saved (for comparison)
 Revised figures clearly marked (v2, v3, etc.)
 Change log maintained (what changed and why)
 Response letter documents figure changes
 Reviewer comments addressed systematically
```

### 16.2 Data Archival
```
 Raw data deposited (journal repository or public database)
 Figure source code deposited (if computational)
 Original unprocessed images archived
 Processing scripts archived
 README file with metadata
```

---

## EMERGENCY DEADLINE CHECKLIST (30-min review)

**If you have 30 minutes before submission:**

```
 DPI ≥ 300 (zoom to 100% and check readability)
 File format correct (TIFF or PNG, NOT JPEG)
 All axes labeled with units
 Panel labels present (A, B, C...)
 Scale bars on microscopy images
 Color-blind safe palette (test with Color Oracle)
 Works in grayscale (File → Print Preview)
 Caption complete (n, statistics, error bars defined)
 File size < limit
 Consistent style across figures
 Statistical annotations clear
 Spelling checked (titles, labels, captions)
```

---

## RESOURCES & TOOLS

### Software
```
Python: matplotlib, seaborn, plotly
R: ggplot2, patchwork, pheatmap
Color tools: Color Oracle, Coblis, ColorBrewer
Image processing: ImageJ/Fiji, Adobe Photoshop (with caution)
Vector graphics: Adobe Illustrator, Inkscape
```

### Color Palettes
```
Colorblind-safe:
- Okabe-Ito (8 colors): Universal
- Viridis family: Sequential, perceptually uniform
- ColorBrewer: CVD-safe filter available
```

### References
```
- Tufte, E. "The Visual Display of Quantitative Information"
- Wilke, C. O. "Fundamentals of Data Visualization" (online)
- Nature Methods "Points of View" series
- Journal-specific figure guidelines
```

### Online Validators
```
- Color contrast: WebAIM Contrast Checker
- Colorblind simulation: Color Oracle, Coblis
- Image integrity: Check for duplications (manually or software)
```

---