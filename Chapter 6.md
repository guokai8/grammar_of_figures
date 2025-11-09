# Chapter 6: Technical Specifications & Publication Requirements

## 6.1 Resolution and Image Quality

### Understanding DPI (Dots Per Inch)

**The Critical Rule:** Resolution must match final print size, not screen appearance.

**What DPI means:**
```
DPI = Dots (pixels) Per Inch when printed

Example:
- Image: 3000 × 2000 pixels
- Printed at 10" × 6.67": 3000/10 = 300 DPI ✓
- Printed at 20" × 13.33": 3000/20 = 150 DPI ❌ (too low)

Key insight: Same image file, different DPI depending on print size
```

---

### Publication DPI Standards

**Minimum requirements by content type:**

```
Line art (graphs, charts, diagrams):
- Minimum: 300 DPI
- Recommended: 600 DPI
- Reasoning: Sharp edges, crisp text

Photographs/Microscopy:
- Minimum: 300 DPI
- Recommended: 600 DPI for fine details
- Maximum practical: 1200 DPI (diminishing returns beyond this)

Combination (line art + photos):
- Minimum: 300 DPI
- Recommended: 600 DPI
- Reasoning: Must satisfy highest requirement

Screen/web only:
- 72-96 DPI acceptable
- But export print versions at 300 DPI anyway
```

---

### Common Resolution Mistakes

**Mistake 1: Upsampling low-resolution images**
```
❌ WRONG approach:
1. Create figure at 100 DPI (screen resolution)
2. "Increase resolution" to 300 DPI in image editor
→ Doesn't add information, just enlarges pixels (blurry)

✓ CORRECT approach:
1. Create figure at 300 DPI from the start
2. Set figure size in inches at creation
→ Generates true high-resolution image
```

**Mistake 2: Relying on vector formats incorrectly**
```
Vector formats (PDF, EPS, SVG):
✓ Good for: Line graphs, bar charts, scatter plots
✓ Scales infinitely without quality loss
✓ Small file sizes

❌ Not good for: Rasterized components (imagesasterized components (images, complex gradients)
→ Embedded images still need 300 DPI

Hybrid approach:
- Save graph as vector
- Ensure embedded images are high-resolution
```

**Code Example (Python) - Setting DPI Correctly:**

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Method 1: Set figure size and DPI at creation
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)  # 7"×5" at 300 DPI = 2100×1500 pixels

x = np.random.randn(100)
y = 2*x + np.random.randn(100)
ax.scatter(x, y, s=50, color='#3498DB', alpha=0.7, edgecolors='black', linewidths=0.5)
ax.set_xlabel('Variable X (units)', fontsize=11, fontweight='bold')
ax.set_ylabel('Variable Y (units)', fontsize=11, fontweight='bold')
ax.set_title('High-Resolution Figure (300 DPI)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save with explicit DPI (this overrides figure DPI if different)
plt.savefig('figure_300dpi.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('figure_600dpi.png', dpi=600, bbox_inches='tight', facecolor='white')  # For fine details

plt.close()

# Check file sizes
import os
size_300 = os.path.getsize('figure_300dpi.png') / 1024  # KB
size_600 = os.path.getsize('figure_600dpi.png') / 1024  # KB

print(f"300 DPI file size: {size_300:.1f} KB")
print(f"600 DPI file size: {size_600:.1f} KB")
print(f"Ratio: {size_600/size_300:.1f}x larger at 600 DPI")
```

**Code Example (R) - Setting DPI Correctly:**

```r
library(ggplot2)

set.seed(42)
data <- data.frame(
  x = rnorm(100),
  y = 2*rnorm(100) + rnorm(100)
)

p <- ggplot(data, aes(x = x, y = y)) +
  geom_point(size = 3, color = '#3498DB', alpha = 0.7) +
  labs(x = 'Variable X (units)',
       y = 'Variable Y (units)',
       title = 'High-Resolution Figure (300 DPI)') +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = 'bold', size = 12, hjust = 0.5),
    axis.title = element_text(face = 'bold'),
    panel.grid.major = element_line(color = 'gray90', size = 0.3)
  )

# Save at different DPI levels
ggsave('figure_300dpi.png', p,
       width = 7, height = 5, dpi = 300, bg = 'white')

ggsave('figure_600dpi.png', p,
       width = 7, height = 5, dpi = 600, bg = 'white')

# Check file sizes
file_info_300 <- file.info('figure_300dpi.png')
file_info_600 <- file.info('figure_600dpi.png')

cat(sprintf("300 DPI file size: %.1f KB\n", file_info_300$size / 1024))
cat(sprintf("600 DPI file size: %.1f KB\n", file_info_600$size / 1024))
cat(sprintf("Ratio: %.1fx larger at 600 DPI\n",
            file_info_600$size / file_info_300$size))
```

---

## 6.2 File Formats for Publication

### Format Decision Tree

```
Is your figure purely vector (lines, shapes, text)?
├─ YES → Use vector format
│   ├─ PDF (most universal)
│   ├─ EPS (legacy journals)
│   └─ SVG (web/interactive)
│
└─ NO (contains images/raster elements) → Use raster format
    ├─ TIFF (publication standard, lossless)
    ├─ PNG (good alternative, lossless, smaller files)
    └─ AVOID JPEG (lossy compression, artifacts)
```

---

### Format Comparison

| Format | Lossless? | Best For | Pros | Cons |
|--------|-----------|----------|------|------|
| **TIFF** | Yes | Final publication submission | Industry standard, supports layers/alpha | Large files |
| **PNG** | Yes | Web, supplementary materials | Smaller than TIFF, transparency support | Limited metadata |
| **PDF** | Depends | Vector graphics, multi-page | Universal, scalable, embeds fonts | Can include low-res images if not careful |
| **EPS** | Yes | Vector for print | PostScript standard, scalable | Being phased out by PDF |
| **SVG** | Yes | Web/interactive graphics | Scalable, editable, small files | Limited journal support |
| **JPEG** | **NO** ❌ | AVOID for science | Small files | **Lossy compression creates artifacts** |

---

### Why JPEG is Dangerous for Scientific Figures

**JPEG compression creates artifacts that can distort data:**

**Code Example (Python) - JPEG Artifact Demonstration:**

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Create a figure with sharp edges
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original high-quality image
ax1 = axes[0]
data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
ax1.imshow(data)
ax1.set_title('Original (Random Pixels)', fontsize=12, fontweight='bold')
ax1.axis('off')

# Save and reload as PNG (lossless)
plt.savefig('temp_lossless.png', dpi=300, bbox_inches='tight')
img_png = Image.open('temp_lossless.png')
ax2 = axes[1]
ax2.imshow(img_png)
ax2.set_title('✓ PNG (Lossless)\nNo artifacts', fontsize=12, fontweight='bold', color='green')
ax2.axis('off')

# Save and reload as JPEG (lossy)
plt.savefig('temp_lossy.jpg', dpi=300, bbox_inches='tight', quality=85)
img_jpg = Image.open('temp_lossy.jpg')
ax3 = axes[2]
ax3.imshow(img_jpg)
ax3.set_title('❌ JPEG (Lossy)\nCompression artifacts visible',
             fontsize=12, fontweight='bold', color='red')
ax3.axis('off')

plt.tight_layout()
plt.savefig('jpeg_artifacts_demo.png', dpi=300, bbox_inches='tight')
plt.close()

# Clean up temp files
import os
os.remove('temp_lossless.png')
os.remove('temp_lossy.jpg')

print("JPEG artifacts demonstration created")
print("Notice: Blocky patterns (8×8 pixel blocks) visible in JPEG version")
```

**Visual evidence of JPEG problems:**
```
Sharp edges → Ringing artifacts (halos around edges)
Flat colors → Blockiness (8×8 pixel compression blocks)
Fine details → Loss of subtle features
Text → Blurry/fuzzy edges
```

**Journals that explicitly FORBID JPEG:**
- Nature family (Nature, Nature Methods, etc.)
- Cell family (Cell, Molecular Cell, etc.)
- Science
- PLOS family (strongly discouraged)

---

## 6.3 Color Modes: RGB vs. CMYK

### Understanding Color Spaces

**RGB (Red, Green, Blue):**
```
Use: Screen display, digital publication
Color range: Wide gamut (more colors available)
Standard for: Online journals, supplementary materials, presentations

How it works:
- Additive color (light-based)
- Black = no light (0, 0, 0)
- White = all light (255, 255, 255)
```

**CMYK (Cyan, Magenta, Yellow, Black):**
```
Use: Print publication
Color range: Smaller gamut (some RGB colors can't be printed)
Standard for: Traditional print journals

How it works:
- Subtractive color (ink-based)
- White = no ink (paper color)
- Black = all inks mixed
```

---

### RGB vs. CMYK Decision

**Modern reality: Most journals want RGB**

```
✓ Use RGB when:
- Journal specifies RGB (most do now)
- Figures will be online (increasing standard)
- Unsure (RGB is safer default)
- Your figures use vivid colors (wider gamut)

✓ Use CMYK when:
- Journal explicitly requires CMYK
- Print-only publication (rare)
- Preparing for commercial printing
```

**Common mistake: Converting RGB → CMYK yourself**
```
❌ WRONG:
You convert to CMYK → Colors shift → Looks dull

✓ CORRECT:
Submit RGB → Journal's printer converts → Optimized conversion
```

**How to check your image color mode:**

```python
# Python
from PIL import Image
img = Image.open('figure.png')
print(f"Color mode: {img.mode}")  # Should be 'RGB' or 'RGBA'
```

```r
# R
library(png)
img <- readPNG('figure.png')
if (dim(img)[3] == 3) {
  cat("Color mode: RGB\n")
} else if (dim(img)[3] == 4) {
  cat("Color mode: RGBA (RGB + alpha channel)\n")
}
```

---

## 6.4 Journal-Specific Requirements

### Reading and Interpreting Figure Guidelines

**Every journal has a "Guide for Authors" or "Figure Guidelines"—READ IT CAREFULLY**

**Common specification categories:**

```
1. Dimensions
   - Single column width (usually 3.5 inches / 89 mm)
   - Double column width (usually 7 inches / 178 mm)
   - Full page width (varies)
   - Maximum height (often 9-10 inches)

2. Resolution
   - Line art: 600-1000 DPI
   - Photos: 300-600 DPI
   - Combination: 300-600 DPI

3. File format
   - Preferred: TIFF, EPS, PDF
   - Acceptable: PNG (some journals)
   - Avoid: JPEG, GIF, BMP

4. Color mode
   - RGB (most common now)
   - CMYK (traditional print journals)

5. File size limits
   - Per figure: 5-20 MB common
   - Total submission: 50-100 MB typical

6. Fonts
   - Embedded/outlined required
   - Minimum size: 6-8 pt after reduction
   - Preferred: Arial, Helvetica, Times

7. Special requirements
   - Separate files for each panel (some journals)
   - Layered files (editable)
   - Color charge (rare now, but check)
```

---

### Example Journal Requirements

**Nature:**
```
- Dimensions: Single column 89 mm, double 183 mm, full 247 mm
- Format: TIFF, EPS, or PDF
- Resolution: 300 DPI photos, 600 DPI line art
- Color: RGB preferred
- Fonts: Must be embedded
- Labels: Bold sans-serif, minimum 7 pt final size
- Special: Submit figures at intended publication size
```

**Cell:**
```
- Dimensions: 1 column 85 mm, 2 columns 178 mm
- Format: TIFF, EPS, PDF, or AI (Adobe Illustrator)
- Resolution: 300 DPI minimum, 600 preferred for line art
- Color: RGB
- Fonts: Arial or Helvetica, embedded
- Special: Figures may be reduced up to 75% for publication
```

**PLOS ONE:**
```
- Dimensions: Width 670-2010 pixels
- Format: TIFF, EPS, PDF, PNG acceptable
- Resolution: 300-600 DPI
- Color: RGB
- Fonts: No specific requirement (but Arial/Helvetica recommended)
- Special: Figures published under CC-BY license (open access)
```

**Science:**
```
- Dimensions: 1 column 5.5 cm, 2 columns 12 cm, 3 columns 18.3 cm
- Format: PDF, TIFF, EPS, PNG
- Resolution: 300 DPI minimum
- Color: RGB or CMYK acceptable
- Fonts: Sans-serif preferred, minimum 6 pt after reduction
- Special: Very strict image integrity policies
```

---

### Pre-Submission Checklist Builder

**Generate custom checklist based on target journal:**

```python
def generate_journal_checklist(journal_name):
    """
    Create custom pre-submission checklist for specific journal
    """
    guidelines = {
        'Nature': {
            'dimensions': ['89mm (single column)', '183mm (double)', '247mm (full)'],
            'format': ['TIFF', 'EPS', 'PDF'],
            'resolution': '300 DPI (photos), 600 DPI (line art)',
            'color': 'RGB',
            'font_min': '7pt after reduction',
            'special': 'Submit at intended size, embed fonts'
        },
        'Cell': {
            'dimensions': ['85mm (1 col)', '178mm (2 col)'],
            'format': ['TIFF', 'EPS', 'PDF', 'AI'],
            'resolution': '300-600 DPI',
            'color': 'RGB',
            'font_min': 'Readable after 75% reduction',
            'special': 'May be reduced up to 75%'
        },
        'PLOS ONE': {
            'dimensions': ['670-2010 pixels width'],
            'format': ['TIFF', 'EPS', 'PDF', 'PNG'],
            'resolution': '300-600 DPI',
            'color': 'RGB',
            'font_min': 'No minimum specified',
            'special': 'CC-BY license (open access)'
        }
    }

    if journal_name not in guidelines:
        return "Journal not in database. Check journal website."

    specs = guidelines[journal_name]

    checklist = f"""
    === {journal_name} Figure Submission Checklist ===

     Dimensions match journal specs: {', '.join(specs['dimensions'])}
     File format: {', '.join(specs['format'])}
     Resolution: {specs['resolution']}
     Color mode: {specs['color']}
     Font size: {specs['font_min']}
     Special requirements: {specs['special']}

     All panels labeled (A, B, C...)
     Scale bars present (if applicable)
     Captions complete and self-contained
     Supplementary figures numbered separately
     Files named according to journal convention
     Image integrity: No manipulation beyond linear adjustments
    """

    return checklist

# Example usage
print(generate_journal_checklist('Nature'))
```

---

## 6.5 Image Compression and File Optimization

### Balancing Quality and File Size

**The challenge:** High-resolution images = large files, but journals have size limits

**Optimization strategies:**

**1. Choose appropriate file format**
```
For line graphs/charts:
✓ PDF (vector, small file, scalable)
✓ EPS (if journal requires)

For photos/images:
✓ PNG with compression level 6-9 (lossless but smaller)
✓ TIFF with LZW compression (lossless)

For combinations:
✓ PDF with high-quality raster settings
✓ TIFF with LZW compression
```

**2. Optimize image dimensions**
```
Don't create unnecessarily large images:
❌ WASTEFUL: 10,000 × 10,000 pixels for 7-inch figure
✓ OPTIMAL: 2,100 × 1,500 pixels (7" × 5" at 300 DPI)

Rule: Pixels = Inches × DPI
```

**3. Flatten layers if not needed**
```
Layered TIFF files can be 10x larger:
✓ Keep layers if journal requires editable files
✓ Flatten layers for final submission if allowed
```

**Code Example (Python) - Image Optimization:**

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

np.random.seed(42)

# Create sample figure
fig, ax = plt.subplots(figsize=(7, 5))
x = np.random.randn(100)
y = 2*x + np.random.randn(100)
ax.scatter(x, y, s=50, color='#3498DB', alpha=0.7)
ax.set_xlabel('Variable X', fontsize=11, fontweight='bold')
ax.set_ylabel('Variable Y', fontsize=11, fontweight='bold')
ax.set_title('File Size Optimization Demo', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Save with different formats and settings
formats = {
    'PNG_default': {'format': 'png', 'dpi': 300},
    'PNG_compressed': {'format': 'png', 'dpi': 300, 'optimize': True},
    'TIFF_uncompressed': {'format': 'tiff', 'dpi': 300},
    'TIFF_LZW': {'format': 'tiff', 'dpi': 300, 'compression': 'tiff_lzw'},
    'PDF_vector': {'format': 'pdf', 'dpi': 300}
}

file_sizes = {}

for name, kwargs in formats.items():
    filename = f'test_{name}.{kwargs["format"]}'
    plt.savefig(filename, bbox_inches='tight', facecolor='white', **kwargs)
    file_sizes[name] = os.path.getsize(filename) / 1024  # KB

    # Clean up
    os.remove(filename)

plt.close()

# Display comparison
print("File Size Comparison (same figure, different formats):")
print("=" * 50)
for name, size in sorted(file_sizes.items(), key=lambda x: x[1]):
    print(f"{name:20} : {size:6.1f} KB")

# Calculate savings
largest = max(file_sizes.values())
smallest = min(file_sizes.values())
print(f"\nOptimization savings: {(1 - smallest/largest)*100:.1f}% size reduction")
```

**Expected output:**

```
File Size Comparison (same figure, different formats):
==================================================
PDF_vector           :   45.2 KB  ← Smallest (vector)
PNG_compressed       :  156.8 KB  ← Good balance
PNG_default          :  203.5 KB
TIFF_LZW            :  287.3 KB
TIFF_uncompressed   :  892.1 KB  ← Largest

Optimization savings: 94.9% size reduction
```

---

### When File Sizes Are Too Large

**Problem:** Figure exceeds journal's file size limit (e.g., 10 MB)

**Solutions (in order of preference):**

```
1. Optimize compression (lossless)
   - Use PNG with optimization
   - Use TIFF with LZW compression
   - Flatten unnecessary layers

2. Reduce unnecessary resolution
   - If figure is 600 DPI but journal requires 300 DPI, downsample
   - If figure is larger than needed, resize to exact print dimensions

3. Split complex figures
   - Break into multiple files (Figure 1A, 1B, 1C as separate files)
   - Some journals prefer this anyway

4. Reduce color depth (carefully!)
   - If grayscale image, save as 8-bit grayscale (not RGB)
   - If indexed color appropriate, convert (rare in science)

5. Move to supplementary materials
   - Less critical figures → supplement
   - High-resolution originals → supplement, lower-res in main text
```

**Code Example (Python) - Batch Image Optimization:**

```python
from PIL import Image
import os

def optimize_image(input_path, output_path, max_size_mb=10, target_dpi=300):
    """
    Optimize image while preserving quality

    Parameters:
    - input_path: Source image file
    - output_path: Destination file
    - max_size_mb: Maximum file size in MB
    - target_dpi: Target DPI (will downsample if higher)
    """
    img = Image.open(input_path)

    # Get current DPI
    dpi = img.info.get('dpi', (300, 300))
    current_dpi = dpi[0] if isinstance(dpi, tuple) else dpi

    # Downsample if exceeds target DPI
    if current_dpi > target_dpi:
        scale_factor = target_dpi / current_dpi
        new_size = tuple(int(dim * scale_factor) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        print(f"Downsampled from {current_dpi} to {target_dpi} DPI")

    # Save with optimization
    save_kwargs = {
        'dpi': (target_dpi, target_dpi),
        'optimize': True
    }

    if output_path.lower().endswith('.png'):
        save_kwargs['compress_level'] = 9  # Maximum PNG compression
    elif output_path.lower().endswith(('.tif', '.tiff')):
        save_kwargs['compression'] = 'tiff_lzw'  # LZW compression for TIFF

    img.save(output_path, **save_kwargs)

    # Check file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)

    if size_mb > max_size_mb:
        print(f"Warning: File size {size_mb:.2f} MB exceeds {max_size_mb} MB")
        print("Consider: Reducing dimensions, splitting figure, or moving to supplement")
    else:
        print(f"✓ Optimized file size: {size_mb:.2f} MB (within {max_size_mb} MB limit)")

    return size_mb

# Example usage
# optimize_image('large_figure.png', 'optimized_figure.png', max_size_mb=10, target_dpi=300)
```

---

**End of Chapter 6: Technical Specifications & Publication Requirements**

**Key Takeaways:**
- **300 DPI minimum** for publication (600 DPI for line art)
- **Create at target size**, don't upsample later
- **TIFF or PNG** for raster images (never JPEG)
- **PDF** for vector graphics
- **RGB color mode** (unless journal specifies CMYK)
- **Read journal guidelines** carefully—requirements vary
- **Optimize file sizes** without losing quality (compression, appropriate dimensions)
- **Check before submission**: DPI, format, color mode, file size, dimensions

