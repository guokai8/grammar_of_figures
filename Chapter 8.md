# Chapter 8: Specialized Field Visualizations

## 8.1 Microscopy Images: Best Practices for Publication

### Unique Challenges in Microscopy Imaging

**Unlike other scientific figures, microscopy images are:**
- **Raw data** (not statistical summaries)
- **Subject to strict ethical guidelines** (no manipulation allowed)
- **High-resolution** (often >10 MB per image)
- **Multi-channel** (fluorescence, confocal, etc.)

**Critical ethical principle:**
> Any adjustment (brightness, contrast) must be applied **uniformly** to all images in a comparison set. Selective editing is scientific misconduct.

---

### Essential Microscopy Figure Elements

**Element 1: Scale Bar**

```
✓ MANDATORY for all microscopy images
Size: Should represent standard unit (e.g., 10 µm, 50 µm, 100 µm)
Placement: Bottom-left or bottom-right corner
Color: White on dark images, black on light images
Width: Bold enough to be visible (2-3 pixels minimum)

❌ NEVER submit microscopy without scale bars
→ Reader cannot interpret size, magnification, or spatial relationships
```

**Element 2: Channel Labels**

```
For multi-channel fluorescence:
✓ Label each channel clearly
✓ Use channel name (not just color): "DAPI (nuclei)" not just "Blue"
✓ Include wavelength if relevant: "GFP (488 nm)"

Merged images:
✓ Show individual channels + merge
✓ Label merge clearly: "Merge" or "Overlay"
```

**Element 3: Image Processing Documentation**

```
In Methods section, MUST document:
- Microscope type and model
- Objective lens magnification and numerical aperture (NA)
- Camera/detector specifications
- Acquisition settings (exposure, gain, binning)
- ALL post-processing applied (brightness, contrast, gamma)
- Software used for processing

If images are cropped, state: "Representative cropped regions shown"
```

---

**Code Example (Python) - Adding Scale Bars to Microscopy Images:**

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib_scalebar.scalebar import ScaleBar

# Simulate microscopy image (grayscale)
np.random.seed(42)
image = np.random.rand(512, 512) * 0.3  # Background noise

# Add some "cells" (bright circular regions)
for _ in range(15):
    x, y = np.random.randint(50, 462, 2)
    r = np.random.randint(20, 40)
    Y, X = np.ogrid[:512, :512]
    mask = (X - x)**2 + (Y - y)**2 <= r**2
    image[mask] += np.random.rand() * 0.5 + 0.3

# Clip to 0-1 range
image = np.clip(image, 0, 1)

# Image metadata
pixel_size_um = 0.5  # 0.5 µm per pixel
scale_bar_um = 50  # 50 µm scale bar

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# === PANEL A: BAD - No scale bar ===
ax1 = axes[0]
ax1.imshow(image, cmap='gray', interpolation='nearest')
ax1.axis('off')
ax1.set_title('❌ BAD: No Scale Bar\n(Cannot determine size)',
             fontsize=12, fontweight='bold', color='red', pad=10)

# === PANEL B: GOOD - Manual scale bar ===
ax2 = axes[1]
ax2.imshow(image, cmap='gray', interpolation='nearest')
ax2.axis('off')

# Calculate scale bar length in pixels
scale_bar_pixels = scale_bar_um / pixel_size_um

# Add scale bar as rectangle
scale_bar_height = 5  # pixels
scale_bar_x = 20  # pixels from left
scale_bar_y = 512 - 30  # pixels from bottom

rect = Rectangle((scale_bar_x, scale_bar_y), scale_bar_pixels, scale_bar_height,
                linewidth=0, edgecolor='none', facecolor='white')
ax2.add_patch(rect)

# Add scale bar label
ax2.text(scale_bar_x + scale_bar_pixels/2, scale_bar_y - 10,
        f'{scale_bar_um} µm',
        color='white', fontsize=10, fontweight='bold', ha='center', va='top')

ax2.set_title('✓ GOOD: Manual Scale Bar\n(Clear size reference)',
             fontsize=12, fontweight='bold', color='green', pad=10)

# === PANEL C: BEST - Using matplotlib-scalebar library ===
ax3 = axes[2]
ax3.imshow(image, cmap='gray', interpolation='nearest')
ax3.axis('off')

# Add scalebar using library (more professional)
scalebar = ScaleBar(pixel_size_um, "um", length_fraction=0.2,
                   location='lower left', box_alpha=0.5, color='white',
                   font_properties={'size': 10, 'weight': 'bold'})
ax3.add_artist(scalebar)

ax3.set_title('✓ BEST: Professional Scale Bar\n(Automated, consistent)',
             fontsize=12, fontweight='bold', color='green', pad=10)

# Add panel labels
for i, ax in enumerate(axes):
    ax.text(0.02, 0.98, chr(65+i), transform=ax.transAxes,
           fontsize=16, fontweight='bold', color='white', va='top',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

plt.tight_layout()
plt.savefig('microscopy_scale_bars.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Microscopy scale bar examples created")
print(f"Image size: {image.shape[0]} × {image.shape[1]} pixels")
print(f"Pixel size: {pixel_size_um} µm/pixel")
print(f"Scale bar: {scale_bar_um} µm = {scale_bar_pixels} pixels")
```

**Code Example (Python) - Multi-Channel Fluorescence Imaging:**

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Simulate 3-channel fluorescence image
size = 256

# Channel 1: DAPI (nuclei, blue)
dapi = np.zeros((size, size))
for _ in range(30):
    x, y = np.random.randint(20, size-20, 2)
    r = np.random.randint(8, 15)
    Y, X = np.ogrid[:size, :size]
    mask = (X - x)**2 + (Y - y)**2 <= r**2
    dapi[mask] = np.random.rand() * 0.8 + 0.2

# Channel 2: GFP (cytoplasm, green)
gfp = np.zeros((size, size))
for _ in range(25):
    x, y = np.random.randint(15, size-15, 2)
    r = np.random.randint(15, 25)
    Y, X = np.ogrid[:size, :size]
    mask = (X - x)**2 + (Y - y)**2 <= r**2
    gfp[mask] = np.random.rand() * 0.6 + 0.1

# Channel 3: RFP (marker, red)
rfp = np.zeros((size, size))
for _ in range(20):
    x, y = np.random.randint(20, size-20, 2)
    r = np.random.randint(5, 10)
    Y, X = np.ogrid[:size, :size]
    mask = (X - x)**2 + (Y - y)**2 <= r**2
    rfp[mask] = np.random.rand() * 0.9 + 0.1

# Create RGB composite
merged = np.zeros((size, size, 3))
merged[:, :, 2] = dapi  # Blue channel
merged[:, :, 1] = gfp   # Green channel
merged[:, :, 0] = rfp   # Red channel

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Individual channels
channel_data = [
    (dapi, 'Blues', 'DAPI (Nuclei)', '#3498DB'),
    (gfp, 'Greens', 'GFP (Cytoplasm)', '#27AE60'),
    (rfp, 'Reds', 'RFP (Marker)', '#E74C3C')
]

for i, (data, cmap, title, color) in enumerate(channel_data):
    ax = axes[0, i]
    ax.imshow(data, cmap=cmap, interpolation='nearest')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', color=color, pad=10)

    # Add channel wavelength
    wavelengths = ['405 nm', '488 nm', '561 nm']
    ax.text(0.5, 0.02, wavelengths[i], transform=ax.transAxes,
           fontsize=9, ha='center', color='white', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Panel label
    ax.text(0.02, 0.98, chr(65+i), transform=ax.transAxes,
           fontsize=16, fontweight='bold', color='white', va='top',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

# Merged image
ax_merge = axes[1, 0]
ax_merge.imshow(merged, interpolation='nearest')
ax_merge.axis('off')
ax_merge.set_title('D. Merge (All Channels)', fontsize=12, fontweight='bold', pad=10)
ax_merge.text(0.02, 0.98, 'D', transform=ax_merge.transAxes,
             fontsize=16, fontweight='bold', color='white', va='top',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

# Merge with only DAPI + GFP
merge_dg = np.zeros((size, size, 3))
merge_dg[:, :, 2] = dapi
merge_dg[:, :, 1] = gfp
ax_dg = axes[1, 1]
ax_dg.imshow(merge_dg, interpolation='nearest')
ax_dg.axis('off')
ax_dg.set_title('E. DAPI + GFP', fontsize=12, fontweight='bold', pad=10)
ax_dg.text(0.02, 0.98, 'E', transform=ax_dg.transAxes,
          fontsize=16, fontweight='bold', color='white', va='top',
          bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

# Quantification example
ax_quant = axes[1, 2]
categories = ['DAPI+', 'GFP+', 'RFP+', 'Colocalized']
counts = [30, 25, 20, 15]
colors_bar = ['#3498DB', '#27AE60', '#E74C3C', '#F39C12']

bars = ax_quant.bar(range(len(categories)), counts, color=colors_bar,
                    edgecolor='black', linewidth=1.5, width=0.6)
ax_quant.set_xticks(range(len(categories)))
ax_quant.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
ax_quant.set_ylabel('Cell Count', fontsize=11, fontweight='bold')
ax_quant.set_title('F. Quantification', fontsize=12, fontweight='bold', pad=10)
ax_quant.spines['top'].set_visible(False)
ax_quant.spines['right'].set_visible(False)
ax_quant.grid(axis='y', alpha=0.3)

# Add counts on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax_quant.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax_quant.text(0.02, 0.98, 'F', transform=ax_quant.transAxes,
             fontsize=16, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('fluorescence_multichannel.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Multi-channel fluorescence example created")
```

---

### Microscopy Figure Best Practices Checklist

```
Essential elements (mandatory):
 Scale bar on ALL images (size + unit label)
 Consistent scale bars across comparison images
 Channel labels for fluorescence (name + wavelength)
 Panel labels (A, B, C...)
 Image acquisition settings in Methods

Image processing (ethical requirements):
 All adjustments documented in Methods
 Brightness/contrast applied uniformly to all comparison images
 No selective editing (e.g., removing unwanted cells)
 Original unprocessed images available if requested
 Linear adjustments only (no gamma correction without justification)

Representative images:
 State in caption: "Representative images shown"
 State how many images/fields were analyzed (n)
 Include quantification (if applicable)

Common mistakes to avoid:
❌ Missing scale bars
❌ Inconsistent scale bars (different sizes in comparison panels)
❌ Unlabeled fluorescence channels
❌ Selective brightness/contrast adjustments
❌ No quantification (single image without statistics)
❌ Aspect ratio distortion (stretched/compressed images)
```

---

## 8.2 Western Blots and Gel Images

### Special Considerations for Gel/Blot Images

**Western blots and gels are semi-quantitative data, NOT just images.**

**Strict rules:**
1. **Show full lanes** (no lane splicing without disclosure)
2. **No background removal** beyond uniform linear adjustments
3. **Molecular weight markers must be visible**
4. **Loading controls required** (e.g., β-actin, GAPDH)

---

### Western Blot Figure Structure

**Standard layout:**

```
Panel A: Target protein blot
- All lanes visible
- Molecular weight marker labeled
- Sample order clear

Panel B: Loading control blot
- Same samples, same order
- Housekeeping protein (β-actin, GAPDH, tubulin)
- Confirms equal loading

Panel C: Quantification
- Bar chart of normalized band intensities
- Target / Loading control ratio
- Error bars (from biological replicates)
- Statistical comparisons
```

---

**Code Example (Python) - Simulated Western Blot Figure:**

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

np.random.seed(42)

# Simulate gel/blot images
def generate_blot_lane(target_level, loading_level=1.0, width=30, height=200):
    """Generate simulated Western blot lane"""
    lane = np.zeros((height, width))

    # Target protein band (variable intensity)
    target_pos = 80  # Position from top
    target_width = 25
    for i in range(target_width):
        y_start = max(0, target_pos - i)
        y_end = min(height, target_pos + i)
        intensity = target_level * np.exp(-i/5) + np.random.rand(1)[0] * 0.1
        lane[y_start:y_end, :] = np.maximum(lane[y_start:y_end, :], intensity)

    # Loading control band (consistent intensity)
    loading_pos = 150
    loading_width = 20
    for i in range(loading_width):
        y_start = max(0, loading_pos - i)
        y_end = min(height, loading_pos + i)
        intensity = loading_level * np.exp(-i/4) + np.random.rand(1)[0] * 0.05
        lane[y_start:y_end, :] = np.maximum(lane[y_start:y_end, :], intensity)

    return lane

# Generate blot with 5 lanes
n_lanes = 5
lane_width = 30
lane_spacing = 10
total_width = n_lanes * lane_width + (n_lanes - 1) * lane_spacing

# Target protein levels (control vs treatments)
target_levels = [0.5, 0.5, 0.8, 1.2, 1.5]  # Relative to control
labels = ['Ctrl 1', 'Ctrl 2', 'Low', 'Med', 'High']

# Create blot image
blot_image = np.zeros((200, total_width))
lane_positions = []

for i, target_level in enumerate(target_levels):
    lane = generate_blot_lane(target_level, loading_level=1.0)
    x_start = i * (lane_width + lane_spacing)
    x_end = x_start + lane_width
    blot_image[:, x_start:x_end] = lane
    lane_positions.append((x_start, x_end))

# Quantification data
normalized_intensities = np.array(target_levels)
errors = np.random.rand(n_lanes) * 0.15

# Create figure
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)

# === PANEL A: Target protein blot ===
ax_target = fig.add_subplot(gs[0, :])
ax_target.imshow(blot_image, cmap='gray_r', aspect='auto', interpolation='nearest')
ax_target.axis('off')
ax_target.set_title('A. Target Protein (e.g., phospho-ERK)',
                   fontsize=12, fontweight='bold', pad=10)

# Add lane labels
for i, (x_start, x_end) in enumerate(lane_positions):
    x_mid = (x_start + x_end) / 2
    ax_target.text(x_mid, blot_image.shape[0] + 10, labels[i],
                  ha='center', fontsize=9, fontweight='bold')

# Add molecular weight marker
ax_target.text(-15, 80, '50 kDa', fontsize=8, ha='right', va='center')
ax_target.plot([-10, -5], [80, 80], 'k-', linewidth=2)

# Panel label
ax_target.text(0.02, 0.98, 'A', transform=ax_target.transAxes,
              fontsize=16, fontweight='bold', va='top')

# === PANEL B: Loading control blot ===
# Generate loading control image (consistent across lanes)
loading_control = np.zeros((200, total_width))
for i in range(n_lanes):
    lane = generate_blot_lane(0.3, loading_level=0.8)  # Consistent loading
    x_start = i * (lane_width + lane_spacing)
    x_end = x_start + lane_width
    loading_control[:, x_start:x_end] = lane

ax_loading = fig.add_subplot(gs[1, :])
ax_loading.imshow(loading_control, cmap='gray_r', aspect='auto', interpolation='nearest')
ax_loading.axis('off')
ax_loading.set_title('B. Loading Control (β-actin)',
                    fontsize=12, fontweight='bold', pad=10)

# Add lane labels
for i, (x_start, x_end) in enumerate(lane_positions):
    x_mid = (x_start + x_end) / 2
    ax_loading.text(x_mid, loading_control.shape[0] + 10, labels[i],
                   ha='center', fontsize=9, fontweight='bold')

# Add molecular weight marker
ax_loading.text(-15, 150, '42 kDa', fontsize=8, ha='right', va='center')
ax_loading.plot([-10, -5], [150, 150], 'k-', linewidth=2)

ax_loading.text(0.02, 0.98, 'B', transform=ax_loading.transAxes,
               fontsize=16, fontweight='bold', va='top')

# === PANEL C: Quantification ===
ax_quant = fig.add_subplot(gs[2, 0])

categories = ['Control', 'Control', 'Low Dose', 'Med Dose', 'High Dose']
x_pos = np.arange(len(categories))
colors_quant = ['#7F8C8D', '#7F8C8D', '#3498DB', '#3498DB', '#E74C3C']

bars = ax_quant.bar(x_pos, normalized_intensities, yerr=errors,
                   color=colors_quant, edgecolor='black', linewidth=1.5,
                   capsize=8, width=0.6, error_kw={'linewidth': 2})

ax_quant.set_xticks(x_pos)
ax_quant.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
ax_quant.set_ylabel('Normalized Intensity\n(Target / Loading Control)',
                   fontsize=11, fontweight='bold')
ax_quant.set_title('C. Quantification (n=3 biological replicates)',
                  fontsize=12, fontweight='bold')
ax_quant.set_ylim(0, 2.0)
ax_quant.grid(axis='y', alpha=0.3)
ax_quant.spines['top'].set_visible(False)
ax_quant.spines['right'].set_visible(False)

# Add statistical comparisons
ax_quant.plot([0.5, 4], [1.8, 1.8], 'k-', linewidth=1.5)
ax_quant.text(2.25, 1.85, '**', ha='center', fontsize=14, fontweight='bold')

ax_quant.text(0.02, 0.98, 'C', transform=ax_quant.transAxes,
             fontsize=16, fontweight='bold', va='top')

# === PANEL D: Representative image note ===
ax_note = fig.add_subplot(gs[2, 1])
ax_note.axis('off')

note_text = """
D. Experimental Details

• Representative blot from 3
  independent experiments

• Quantification based on 3
  biological replicates

• Statistical test: One-way ANOVA
  with Dunnett's post-hoc

• ** p < 0.01 vs. Control

• Blot image: Linear adjustments
  only, applied uniformly

• Full unprocessed blots available
  in supplementary materials
"""

ax_note.text(0.1, 0.9, note_text, transform=ax_note.transAxes,
            fontsize=9, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax_note.text(0.02, 0.98, 'D', transform=ax_note.transAxes,
            fontsize=16, fontweight='bold', va='top')

plt.suptitle('Western Blot Analysis: Publication-Ready Format',
            fontsize=14, fontweight='bold', y=0.98)

plt.savefig('western_blot_figure.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Western blot figure created")
print("Key components:")
print("  - Target protein blot (Panel A)")
print("  - Loading control blot (Panel B)")
print("  - Quantification with statistics (Panel C)")
print("  - Methods documentation (Panel D)")
```

---

### Western Blot Best Practices Checklist

```
Image integrity (mandatory):
 Show FULL lanes (no splicing without clear indication)
 Molecular weight markers visible and labeled
 Loading control included (same sample order)
 All lanes from SAME gel/membrane (if comparing)
 Linear adjustments only, applied uniformly
 Original unprocessed images available

Quantification (required for claims):
 Densitometry performed on multiple replicates (n≥3)
 Normalized to loading control
 Error bars shown (SEM or SD specified)
 Statistical test stated
 "Representative image" noted in caption

Caption must include:
 Protein names and antibodies used
 Molecular weights
 Sample sizes and replicates
 How bands were quantified
 Statistical methods

Ethical violations (NEVER do):
❌ Splice lanes from different gels without disclosure
❌ Selectively adjust brightness/contrast for specific lanes
❌ Remove background non-uniformly
❌ Duplicate lanes
❌ Clone/copy-paste bands
```

---

## 8.3 Flow Cytometry Plots

### Standard Flow Cytometry Display Formats

**Common plot types:**
1. **Histogram:** Single parameter distribution
2. **Dot plot:** Two-parameter comparison (most common)
3. **Contour plot:** Density representation
4. **Overlay histogram:** Compare multiple samples

---

### Flow Cytometry Figure Essentials

**Element 1: Gating Strategy**

```
Show the gating hierarchy:
1. Forward scatter (FSC) vs. Side scatter (SSC) → Cell population
2. Doublet discrimination → Single cells
3. Viability gate → Live cells
4. Marker-positive gates → Populations of interest

✓ Show representative gates with percentages
```

**Element 2: Axes and Scales**

```
X/Y axes must show:
- Parameter name (e.g., "CD4-FITC")
- Scale type (linear vs. log)
- Units if applicable

Most fluorescence: Logarithmic scale
Scatter parameters: Linear scale
```

**Element 3: Quadrant/Gate Statistics**

```
Each gate/quadrant should show:
- Percentage of parent population
- Absolute count (if relevant)
- Gate name/definition
```

---

**Code Example (Python) - Flow Cytometry Figure:**

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Polygon

np.random.seed(42)

# Simulate flow cytometry data (2 populations)
n_cells = 10000

# Population 1: CD4+ cells (high CD4, low CD8)
cd4_pos = np.random.lognormal(mean=3.5, sigma=0.5, size=4000)
cd8_neg = np.random.lognormal(mean=1.5, sigma=0.5, size=4000)

# Population 2: CD8+ cells (low CD4, high CD8)
cd4_neg = np.random.lognormal(mean=1.5, sigma=0.5, size=3000)
cd8_pos = np.random.lognormal(mean=3.5, sigma=0.5, size=3000)

# Population 3: Double negative (low both)
cd4_dn = np.random.lognormal(mean=1.5, sigma=0.5, size=2000)
cd8_dn = np.random.lognormal(mean=1.5, sigma=0.5, size=2000)

# Population 4: Double positive (high both) - rare
cd4_dp = np.random.lognormal(mean=3.5, sigma=0.5, size=1000)
cd8_dp = np.random.lognormal(mean=3.5, sigma=0.5, size=1000)

# Combine all populations
cd4_all = np.concatenate([cd4_pos, cd4_neg, cd4_dn, cd4_dp])
cd8_all = np.concatenate([cd8_neg, cd8_pos, cd8_dn, cd8_dp])

# Log transform for display
cd4_log = np.log10(cd4_all)
cd8_log = np.log10(cd8_all)

# Define gates (in log space)
cd4_threshold = 2.5  # 10^2.5
cd8_threshold = 2.5

# Calculate quadrant percentages
q1 = np.sum((cd4_log < cd4_threshold) & (cd8_log > cd8_threshold)) / len(cd4_log) * 100  # CD4- CD8+
q2 = np.sum((cd4_log > cd4_threshold) & (cd8_log > cd8_threshold)) / len(cd4_log) * 100  # CD4+ CD8+
q3 = np.sum((cd4_log < cd4_threshold) & (cd8_log < cd8_threshold)) / len(cd4_log) * 100  # CD4- CD8-
q4 = np.sum((cd4_log > cd4_threshold) & (cd8_log < cd8_threshold)) / len(cd4_log) * 100  # CD4+ CD8-

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# === PANEL A: Dot plot (standard) ===
ax1 = axes[0, 0]

# Plot cells as scatter
ax1.scatter(cd4_log, cd8_log, s=1, alpha=0.3, c='#3498DB', rasterized=True)

# Add gates (cross-hairs)
ax1.axhline(cd8_threshold, color='red', linewidth=2, linestyle='--')
ax1.axvline(cd4_threshold, color='red', linewidth=2, linestyle='--')

# Add quadrant percentages
ax1.text(1.5, 3.5, f'CD4⁻ CD8⁺\n{q1:.1f}%', fontsize=10, fontweight='bold',
        ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax1.text(3.5, 3.5, f'CD4⁺ CD8⁺\n{q2:.1f}%', fontsize=10, fontweight='bold',
        ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax1.text(1.5, 1.5, f'CD4⁻ CD8⁻\n{q3:.1f}%', fontsize=10, fontweight='bold',
        ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax1.text(3.5, 1.5, f'CD4⁺ CD8⁻\n{q4:.1f}%', fontsize=10, fontweight='bold',
        ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax1.set_xlabel('CD4-FITC (log₁₀)', fontsize=11, fontweight='bold')
ax1.set_ylabel('CD8-PE (log₁₀)', fontsize=11, fontweight='bold')
ax1.set_title('A. Dot Plot with Quadrant Gates', fontsize=12, fontweight='bold')
ax1.set_xlim(0.5, 4.5)
ax1.set_ylim(0.5, 4.5)
ax1.grid(alpha=0.3)

# Panel label
ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes,
        fontsize=16, fontweight='bold', va='top')

# === PANEL B: Contour plot (density) ===
ax2 = axes[0, 1]

# Create 2D histogram for contour
from scipy.stats import gaussian_kde

# Subsample for KDE (faster computation)
sample_idx = np.random.choice(len(cd4_log), size=2000, replace=False)
cd4_sample = cd4_log[sample_idx]
cd8_sample = cd8_log[sample_idx]

# Create grid
x_grid = np.linspace(0.5, 4.5, 100)
y_grid = np.linspace(0.5, 4.5, 100)
X, Y = np.meshgrid(x_grid, y_grid)
positions = np.vstack([X.ravel(), Y.ravel()])

# Calculate density
kernel = gaussian_kde([cd4_sample, cd8_sample])
Z = kernel(positions).reshape(X.shape)

# Plot contours
contours = ax2.contour(X, Y, Z, levels=8, colors='#3498DB', linewidths=1.5)
ax2.contourf(X, Y, Z, levels=8, cmap='Blues', alpha=0.6)

# Add gates
ax2.axhline(cd8_threshold, color='red', linewidth=2, linestyle='--')
ax2.axvline(cd4_threshold, color='red', linewidth=2, linestyle='--')

# Add quadrant percentages
ax2.text(3.5, 1.5, f'{q4:.1f}%', fontsize=11, fontweight='bold',
        ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax2.set_xlabel('CD4-FITC (log₁₀)', fontsize=11, fontweight='bold')
ax2.set_ylabel('CD8-PE (log₁₀)', fontsize=11, fontweight='bold')
ax2.set_title('B. Contour Plot (Density)', fontsize=12, fontweight='bold')
ax2.set_xlim(0.5, 4.5)
ax2.set_ylim(0.5, 4.5)

ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes,
        fontsize=16, fontweight='bold', va='top')

# === PANEL C: Histogram overlay (CD4 expression) ===
ax3 = axes[1, 0]

# Separate populations
cd4_pos_cells = cd4_log[cd4_log > cd4_threshold]
cd4_neg_cells = cd4_log[cd4_log <= cd4_threshold]

ax3.hist(cd4_neg_cells, bins=50, alpha=0.6, color='#7F8C8D',
        edgecolor='black', linewidth=0.5, label=f'CD4⁻ ({q3+q1:.1f}%)')
ax3.hist(cd4_pos_cells, bins=50, alpha=0.6, color='#E74C3C',
        edgecolor='black', linewidth=0.5, label=f'CD4⁺ ({q4+q2:.1f}%)')

# Add gate threshold
ax3.axvline(cd4_threshold, color='red', linewidth=2, linestyle='--')

ax3.set_xlabel('CD4-FITC (log₁₀)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Cell Count', fontsize=11, fontweight='bold')
ax3.set_title('C. Histogram Overlay: CD4 Expression', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right', frameon=True, fontsize=10)
ax3.grid(axis='y', alpha=0.3)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

ax3.text(0.02, 0.98, 'C', transform=ax3.transAxes,
        fontsize=16, fontweight='bold', va='top')

# === PANEL D: Quantification bar chart ===
ax4 = axes[1, 1]

populations = ['CD4⁺\nCD8⁻', 'CD4⁻\nCD8⁺', 'CD4⁺\nCD8⁺', 'CD4⁻\nCD8⁻']
percentages = [q4, q1, q2, q3]
colors_pop = ['#E74C3C', '#3498DB', '#9B59B6', '#7F8C8D']

bars = ax4.bar(range(len(populations)), percentages, color=colors_pop,
              edgecolor='black', linewidth=1.5, width=0.6)

# Add percentage labels on bars
for bar, pct in zip(bars, percentages):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax4.set_xticks(range(len(populations)))
ax4.set_xticklabels(populations, fontsize=10, fontweight='bold')
ax4.set_ylabel('% of Total Cells', fontsize=11, fontweight='bold')
ax4.set_title('D. Population Frequencies', fontsize=12, fontweight='bold')
ax4.set_ylim(0, max(percentages) + 10)
ax4.grid(axis='y', alpha=0.3)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

ax4.text(0.02, 0.98, 'D', transform=ax4.transAxes,
        fontsize=16, fontweight='bold', va='top')

plt.suptitle(f'Flow Cytometry Analysis (n = {len(cd4_log):,} cells)',
            fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('flow_cytometry_figure.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Flow cytometry figure created")
print(f"Total cells analyzed: {len(cd4_log):,}")
print(f"\nPopulation frequencies:")
print(f"  CD4+ CD8- (Helper T cells): {q4:.2f}%")
print(f"  CD4- CD8+ (Cytotoxic T cells): {q1:.2f}%")
print(f"  CD4+ CD8+ (Double positive): {q2:.2f}%")
print(f"  CD4- CD8- (Double negative): {q3:.2f}%")
```

---

### Flow Cytometry Best Practices Checklist

```
Plot requirements:
 Axes labeled with marker name and fluorophore (e.g., "CD4-FITC")
 Scale type indicated (linear or logarithmic)
 Gates clearly visible (contrasting color, typically red)
 Population percentages shown in each gate/quadrant
 Total cell count stated (n = X cells)

Gating strategy:
 Show sequential gating steps (if applicable)
 Define gating criteria in Methods or caption
 Include negative/isotype controls (in supplement if not main figure)
 State how gates were determined (fluorescence minus one, isotype, etc.)

Caption requirements:
 Antibody clones and fluorophores
 Gating strategy summary
 Number of cells analyzed
 Number of biological replicates
 Representative or pooled data (specify)

Common mistakes:
❌ Unlabeled axes or missing fluorophore names
❌ No scale indication (linear vs. log unclear)
❌ Gates without percentages
❌ No compensation information in Methods
❌ Comparing plots with different scales
```

---

## 8.4 Phylogenetic Trees

### Tree Visualization Essentials

**Common in:** Evolutionary biology, microbiology, bioinformatics

**Key components:**

```
1. Branch lengths: Represent evolutionary distance
2. Node labels: Bootstrap values (confidence)
3. Tip labels: Species/strain/sequence names
4. Scale bar: Units of substitution or time
5. Root: Outgroup or midpoint
```

---

**Code Example (Python) - Phylogenetic Tree:**

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy.cluster.hierarchy import dendrogram, linkage

np.random.seed(42)

# Simulate sequence distance matrix (phylogenetic distances)
species = ['Human', 'Chimp', 'Gorilla', 'Orangutan', 'Macaque',
          'Mouse', 'Rat', 'Dog', 'Cat', 'Cow']
n_species = len(species)

# Create distance matrix (lower triangle)
distances = np.random.rand(n_species, n_species) * 0.5
distances = (distances + distances.T) / 2  # Make symmetric
np.fill_diagonal(distances, 0)

# Adjust to reflect known phylogeny (primates closer to each other)
# Primates (0-4): closer
distances[0:5, 0:5] *= 0.3
# Rodents (5-6): closer
distances[5:7, 5:7] *= 0.3
# Carnivores (7-9): closer
distances[7:9, 7:9] *= 0.3

# Perform hierarchical clustering
Z = linkage(distances[np.triu_indices(n_species, k=1)], method='average')

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# === PANEL A: Standard dendrogram ===
ax1 = axes[0]

dendro = dendrogram(Z, labels=species, orientation='right', ax=ax1,
                   color_threshold=0.5, above_threshold_color='#3498DB')

ax1.set_xlabel('Evolutionary Distance (substitutions/site)', fontsize=11, fontweight='bold')
ax1.set_title('A. Phylogenetic Tree\n(UPGMA clustering)',
             fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add scale bar
scale_bar_length = 0.1
ax1.plot([0, scale_bar_length], [-1, -1], 'k-', linewidth=3)
ax1.text(scale_bar_length/2, -1.5, '0.1 substitutions/site',
        ha='center', fontsize=9, fontweight='bold')

# Add bootstrap values (simulated) at major nodes
# In real analysis, these come from resampling
node_positions = [(0.15, 3.5, 95), (0.12, 6.5, 88), (0.25, 7.5, 78)]
for x, y, boot in node_positions:
    ax1.text(x, y, str(boot), fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Highlight clades
# Primates
primate_y = [i for i, sp in enumerate(dendro['ivl']) if sp in ['Human', 'Chimp', 'Gorilla', 'Orangutan', 'Macaque']]
if primate_y:
    y_min, y_max = min(primate_y), max(primate_y)
    rect = FancyBboxPatch((-0.02, y_min-0.5), 0.02, y_max-y_min+1,
                         boxstyle="round,pad=0.02",
                         edgecolor='#E74C3C', facecolor='none',
                         linewidth=2, transform=ax1.transData)
    ax1.add_patch(rect)
    ax1.text(-0.04, (y_min+y_max)/2, 'Primates', rotation=90, va='center',
            fontsize=9, fontweight='bold', color='#E74C3C')

ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes,
        fontsize=16, fontweight='bold', va='top')

# === PANEL B: Circular tree (radial layout) ===
ax2 = axes[1]
ax2.set_aspect('equal')
ax2.axis('off')

# Simplified circular tree representation
n_leaves = len(species)
angles = np.linspace(0, 2*np.pi, n_leaves, endpoint=False)

# Plot leaf labels
radius = 1.0
for i, (angle, sp) in enumerate(zip(angles, species)):
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)

    # Leaf node
    ax2.plot(x, y, 'o', markersize=8, color='#3498DB')

    # Label
    angle_deg = np.degrees(angle)
    if -90 <= angle_deg <= 90:
        ha = 'left'
        rotation = angle_deg
        x_text = x * 1.15
    else:
        ha = 'right'
        rotation = angle_deg + 180
        x_text = x * 1.15

    ax2.text(x_text, y * 1.15, sp, ha=ha, va='center',
            rotation=rotation, fontsize=9, fontweight='bold')

# Draw simplified branches (radial from center)
for i, angle in enumerate(angles):
    x_inner = 0.3 * np.cos(angle)
    y_inner = 0.3 * np.sin(angle)
    x_outer = radius * np.cos(angle)
    y_outer = radius * np.sin(angle)
    ax2.plot([0, x_inner], [0, y_inner], 'k-', linewidth=1, alpha=0.5)
    ax2.plot([x_inner, x_outer], [y_inner, y_outer], color='#3498DB', linewidth=2)

# Central node
ax2.plot(0, 0, 'ko', markersize=10)

# Title
ax2.set_title('B. Circular Phylogenetic Tree\n(Alternative layout)',
             fontsize=12, fontweight='bold')
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)

ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes,
        fontsize=16, fontweight='bold', va='top')

plt.suptitle('Phylogenetic Analysis: Tree Visualization',
            fontsize=14, fontweight='bold', y=0.96)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('phylogenetic_tree.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Phylogenetic tree figure created")
print(f"Species analyzed: {n_species}")
print("Tree construction: UPGMA clustering method")
```

---

### Phylogenetic Tree Best Practices

```
Essential elements:
 Scale bar with units (substitutions/site, years, etc.)
 Branch lengths proportional to distance (unless cladogram)
 Bootstrap or posterior probability values at nodes
 Clear tip labels (species/strain names)
 Root indicated (outgroup or midpoint)
 Tree construction method stated in caption

Layout choices:
✓ Rectangular (standard): Best for detailed branch lengths
✓ Circular/radial: Good for large trees, emphasizes relationships
✓ Unrooted: When root position unknown

Caption requirements:
 Alignment method (if sequences)
 Tree-building algorithm (NJ, ML, Bayesian, etc.)
 Bootstrap replicates (if applicable)
 Outgroup used for rooting
 Software used

Avoid:
❌ Unlabeled scale bars
❌ No bootstrap values (can't assess confidence)
❌ Inconsistent branch lengths without justification
❌ Missing outgroup (unclear root)
```

---

## 8.5 Network Diagrams

### When to Use Network Visualizations

**Common in:**

- Protein-protein interaction networks
- Gene regulatory networks
- Metabolic pathways
- Co-occurrence networks

**Key components:**

```
Nodes: Entities (genes, proteins, metabolites)
Edges: Relationships (interactions, correlations)
Node size: Often encodes degree or importance
Node color: Often encodes category or expression
Edge width: Often encodes interaction strength
```

---

**Code Example (Python) - Network Diagram:**

```python
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

np.random.seed(42)

# Create a gene regulatory network (simplified)
G = nx.Graph()

# Add nodes (genes)
genes = ['GENE_A', 'GENE_B', 'GENE_C', 'GENE_D', 'GENE_E',
        'GENE_F', 'GENE_G', 'GENE_H', 'GENE_I', 'GENE_J']

# Gene categories (functional groups)
categories = {
    'GENE_A': 'Transcription Factor',
    'GENE_B': 'Transcription Factor',
    'GENE_C': 'Signaling',
    'GENE_D': 'Signaling',
    'GENE_E': 'Signaling',
    'GENE_F': 'Metabolic',
    'GENE_G': 'Metabolic',
    'GENE_H': 'Structural',
    'GENE_I': 'Structural',
    'GENE_J': 'Receptor'
}

# Expression levels (fold change)
expression = {
    'GENE_A': 2.5, 'GENE_B': 1.8, 'GENE_C': -1.5,
    'GENE_D': 3.2, 'GENE_E': -2.1, 'GENE_F': 1.2,
    'GENE_G': -1.8, 'GENE_H': 0.5, 'GENE_I': 2.8, 'GENE_J': 1.5
}

# Add nodes with attributes
for gene in genes:
    G.add_node(gene, category=categories[gene], expression=expression[gene])

# Add edges (interactions)
interactions = [
    ('GENE_A', 'GENE_C', 0.9),  # (gene1, gene2, correlation)
    ('GENE_A', 'GENE_D', 0.85),
    ('GENE_B', 'GENE_C', 0.75),
    ('GENE_B', 'GENE_E', 0.8),
    ('GENE_C', 'GENE_F', 0.7),
    ('GENE_D', 'GENE_G', 0.65),
    ('GENE_E', 'GENE_H', 0.6),
    ('GENE_F', 'GENE_I', 0.9),
    ('GENE_G', 'GENE_J', 0.7),
    ('GENE_H', 'GENE_I', 0.85),
    ('GENE_I', 'GENE_J', 0.75),
    ('GENE_A', 'GENE_B', 0.95)
]

for gene1, gene2, weight in interactions:
    G.add_edge(gene1, gene2, weight=weight)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# === PANEL A: Network colored by category ===
ax1 = axes[0]

# Layout
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

# Node colors by category
category_colors = {
    'Transcription Factor': '#E74C3C',
    'Signaling': '#3498DB',
    'Metabolic': '#27AE60',
    'Structural': '#F39C12',
    'Receptor': '#9B59B6'
}

node_colors = [category_colors[categories[node]] for node in G.nodes()]

# Node sizes by degree (number of connections)
node_sizes = [G.degree(node) * 300 for node in G.nodes()]

# Edge widths by correlation
edge_widths = [G[u][v]['weight'] * 3 for u, v in G.edges()]

# Draw network
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                      edgecolors='black', linewidths=2, ax=ax1)
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray', ax=ax1)
nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax1)

ax1.set_title('A. Gene Regulatory Network\n(Colored by functional category)',
             fontsize=12, fontweight='bold')
ax1.axis('off')

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, edgecolor='black', label=cat)
                  for cat, color in category_colors.items()]
ax1.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=9,
          title='Functional Category', title_fontsize=10)

ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes,
        fontsize=16, fontweight='bold', va='top')

# === PANEL B: Network colored by expression ===
ax2 = axes[1]

# Node colors by expression level
expr_values = [expression[node] for node in G.nodes()]
node_colors_expr = plt.cm.RdBu_r([(x + 3) / 6 for x in expr_values])  # Normalize to 0-1

# Draw network
nx.draw_networkx_nodes(G, pos, node_color=node_colors_expr, node_size=node_sizes,
                      edgecolors='black', linewidths=2, ax=ax2)
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray', ax=ax2)
nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax2)

ax2.set_title('B. Gene Regulatory Network\n(Colored by fold change)',
             fontsize=12, fontweight='bold')
ax2.axis('off')

# Colorbar for expression
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Create colorbar axis
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
norm = Normalize(vmin=-3, vmax=3)
cb = ColorbarBase(cbar_ax, cmap=cm.RdBu_r, norm=norm, orientation='vertical')
cb.set_label('Log₂ Fold Change', fontsize=10, fontweight='bold')

ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes,
        fontsize=16, fontweight='bold', va='top')

plt.suptitle(f'Network Analysis: {len(G.nodes())} genes, {len(G.edges())} interactions',
            fontsize=14, fontweight='bold', y=0.95)

plt.tight_layout(rect=[0, 0, 0.9, 0.93])
plt.savefig('network_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Network diagram created")
print(f"Nodes (genes): {len(G.nodes())}")
print(f"Edges (interactions): {len(G.edges())}")
print(f"Network density: {nx.density(G):.3f}")
print(f"Average clustering coefficient: {nx.average_clustering(G):.3f}")
```

---

### Network Diagram Best Practices

```
Node encoding:
 Size: Often represents degree, importance, or expression level
 Color: Category (discrete) or value (continuous)
 Shape: Secondary category (if needed)
 Label: Clear, non-overlapping

Edge encoding:
 Width: Interaction strength, correlation, weight
 Color: Interaction type (activation, inhibition, etc.)
 Style: Solid vs. dashed for different evidence levels

Layout considerations:
✓ Force-directed (spring): Good for general networks
✓ Hierarchical: Good for directed networks with clear levels
✓ Circular: Good for highlighting cycles
✓ Manual: When biological layout is known

Caption requirements:
 Node and edge definitions
 Network statistics (nodes, edges, density)
 Layout algorithm used
 Data source and filtering criteria
 Software used

Avoid:
❌ Overlapping labels (use algorithms like adjustText)
❌ Too many nodes (>50 becomes cluttered; consider subnetwork)
❌ Unlabeled encoding (what does node size mean?)
❌ Hairball (too dense; filter by importance/threshold)
```

---

### **8.6 Novel Plot Types: Providing Context**

**Principle:** Any non-standard plot type requires extra explanation to ensure reader understanding.

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge, Circle

np.random.seed(42)

fig = plt.figure(figsize=(16, 12))

# Example: Circular plot (not commonly used in biology)
# MUST provide sufficient context

# Panel A: Novel plot WITHOUT explanation (confusing)
ax1 = plt.subplot(2, 2, 1, projection='polar')
theta = np.linspace(0, 2*np.pi, 24)
r = 50 + 30*np.sin(3*theta) + np.random.randn(24)*5

ax1.plot(theta, r, 'o-', linewidth=2, markersize=8, color='#3498DB')
ax1.fill(theta, r, alpha=0.3, color='#3498DB')
ax1.set_theta_zero_location('N')
ax1.set_title('❌ BAD: No Explanation\n(What am I looking at?)',
              fontsize=13, fontweight='bold', color='red', pad=20)

# Panel B: Same plot WITH clear explanation
ax2 = plt.subplot(2, 2, 2, projection='polar')
ax2.plot(theta, r, 'o-', linewidth=2.5, markersize=8, color='#27AE60')
ax2.fill(theta, r, alpha=0.3, color='#27AE60')
ax2.set_theta_zero_location('N')
ax2.set_theta_direction(-1)  # Clockwise

# Add time labels
hour_labels = [f'{h}:00' for h in range(0, 24, 3)]
ax2.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
ax2.set_xticklabels(hour_labels, fontsize=10)

# Add radial axis label
ax2.set_ylabel('Activity Level', fontsize=11, fontweight='bold', labelpad=30)

# Add annotations
ax2.text(0, 85, 'Peak\nActivity', ha='center', fontsize=10,
        fontweight='bold', color='red',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

ax2.set_title('✓ GOOD: Clear Context\n"24-hour Circadian Activity Pattern"',
              fontsize=13, fontweight='bold', color='green', pad=20)

# Panel C: Novel plot with legend/key explaining elements
ax3 = plt.subplot(2, 2, 3)

# Example: Alluvial/Sankey-style plot
# Simulate patient flow between disease states
states = ['Healthy', 'Stage I', 'Stage II', 'Stage III']
time_points = ['Baseline', 'Month 3', 'Month 6', 'Month 9']

# Create flow diagram
from matplotlib.sankey import Sankey

# Simplified representation (actual Sankey would be more complex)
ax3.text(0.5, 0.95, 'Disease Progression Flow Diagram',
        ha='center', va='top', transform=ax3.transAxes,
        fontsize=12, fontweight='bold')

# Add explanatory text boxes
explanations = [
    "Arrow width = Number of patients",
    "Color = Disease severity",
    "Left → Right = Time progression",
    "Splits show state transitions"
]

y_pos = 0.8
for exp in explanations:
    ax3.text(0.05, y_pos, f'• {exp}', transform=ax3.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    y_pos -= 0.15

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')
ax3.set_title('✓ GOOD: Novel Plot with Key\n(All elements explained)',
              fontsize=13, fontweight='bold', color='green', pad=20)

# Panel D: Checklist for novel plots
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

checklist_text = """
✓ CHECKLIST FOR NOVEL PLOT TYPES:

 Title clearly states what plot shows
   Example: "Circular Heatmap of Temporal Gene Expression"

 All axes labeled with units
   • Radial axis: What does distance mean?
   • Angular axis: What does angle represent?

 Legend explains all visual encodings
   • Line thickness → Sample size
   • Color → Statistical significance
   • Pattern → Experimental group

 Caption provides interpretation guide
   "In this polar plot, each point represents
    hourly measurements over 24 hours. Radial
    distance indicates activity level (AU)."

 Reference to similar plot if published
   "Similar to circular genome plots (Zhang et al. 2020)"

 Full methods in supplementary
   • Software used
   • Parameter settings
   • Data processing steps
"""

ax4.text(0.05, 0.95, checklist_text, transform=ax4.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

ax4.set_title('Novel Plot Checklist',
              fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('novel_plots_explanation.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

---

**End of Chapter 8: Specialized Field Visualizations**

**Summary Table:**

| Figure Type | Field | Key Elements | Common Pitfalls |
|-------------|-------|--------------|-----------------|
| **Microscopy** | All biology | Scale bar, channel labels | Missing scale bar, selective editing |
| **Western Blot** | Molecular biology | Full lanes, loading control, quantification | Lane splicing, no loading control |
| **Flow Cytometry** | Immunology | Gates with %, axes labeled, scale type | Unlabeled axes, no gate percentages |
| **Phylogenetic Tree** | Evolution | Scale bar, bootstrap values, root | No scale bar, no confidence values |
| **Network Diagram** | Systems biology | Node/edge encoding, layout, stats | Overlapping labels, hairball, unlabeled encoding |

---