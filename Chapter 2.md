# Chapter 2: The Language of Color

## 2.1 Color Theory Foundations

### Why Color Matters in Scientific Visualization

Color is one of the most powerful visual channels available, yet it's also one of the most misused. Unlike position or length, which have inherent quantitative meaning, color's interpretation is:

- **Culturally dependent** (red means "danger" in some contexts, "celebration" in others)
- **Context-dependent** (same color looks different on different backgrounds)
- **Perceptually non-uniform** (equal numeric differences ≠ equal perceptual differences)
- **Biologically variable** (8% of males, 0.5% of females have color vision deficiency)

**The Goal of This Chapter:**
Teach you to use color **scientifically** — as a precise encoding tool, not decoration.

---

### Color Models and Color Spaces

To use color effectively, we must understand how to specify and manipulate it. Different color models serve different purposes.

#### RGB: The Additive Model (Screens)

**Definition:** Colors created by mixing Red, Green, and Blue light.

**Range:** Each channel: 0-255 (8-bit) or 0.0-1.0 (normalized)

**Examples:**

```
Pure Red:    RGB(255, 0, 0)   or RGB(1.0, 0.0, 0.0)
Pure Green:  RGB(0, 255, 0)   or RGB(0.0, 1.0, 0.0)
Pure Blue:   RGB(0, 0, 255)   or RGB(0.0, 0.0, 1.0)
White:       RGB(255, 255, 255)
Black:       RGB(0, 0, 0)
Gray:        RGB(128, 128, 128)
```

**Strengths:**
- Native to digital displays
- Direct hardware mapping
- Simple for programming

**Weaknesses:**
- **Not perceptually uniform**: Equal RGB steps don't produce equal visual differences
- **Hard to reason about**: "What's halfway between red and blue?" → Purple? Magenta?
- **Doesn't separate color dimensions**: Hue and brightness are entangled

**When to Use:**
- Specifying colors for screen display
- Digital-only figures
- When starting from hex codes (e.g., #FF5733)

**Code Example (Python):**

```
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Demonstrate RGB color space
fig, ax = plt.subplots(figsize=(10, 4))

# Create color samples
colors_rgb = [
    (1.0, 0.0, 0.0, 'Red\nRGB(255,0,0)'),
    (0.0, 1.0, 0.0, 'Green\nRGB(0,255,0)'),
    (0.0, 0.0, 1.0, 'Blue\nRGB(0,0,255)'),
    (1.0, 1.0, 0.0, 'Yellow\nRGB(255,255,0)'),
    (1.0, 0.0, 1.0, 'Magenta\nRGB(255,0,255)'),
    (0.0, 1.0, 1.0, 'Cyan\nRGB(0,255,255)'),
]

for i, (r, g, b, label) in enumerate(colors_rgb):
    rect = mpatches.Rectangle((i, 0), 1, 1, facecolor=(r, g, b))
    ax.add_patch(rect)
    ax.text(i+0.5, 0.5, label, ha='center', va='center',
            fontsize=9, color='white' if (r+g+b) < 1.5 else 'black')

ax.set_xlim(0, len(colors_rgb))
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('RGB Color Model: Additive Primaries', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('rgb_color_model.png', dpi=300, bbox_inches='tight')
plt.close()
```

**Code Example (R):**

```
library(ggplot2)
library(dplyr)

# Create RGB color samples
rgb_colors <- data.frame(
  x = 1:6,
  color = c('red', 'green', 'blue', 'yellow', 'magenta', 'cyan'),
  rgb_code = c('RGB(255,0,0)', 'RGB(0,255,0)', 'RGB(0,0,255)',
               'RGB(255,255,0)', 'RGB(255,0,255)', 'RGB(0,255,255)'),
  hex = c('#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF')
)

ggplot(rgb_colors, aes(x = x, y = 1)) +
  geom_tile(aes(fill = hex), color = 'black', size = 1, width = 0.9, height = 0.8) +
  geom_text(aes(label = paste(color, rgb_code, sep='\n')),
            color = 'white', fontface = 'bold', size = 3.5) +
  scale_fill_identity() +
  labs(title = 'RGB Color Model: Additive Primaries') +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = 'bold'))

ggsave('rgb_color_model.png', width = 10, height = 3, dpi = 300)
```

---

#### CMYK: The Subtractive Model (Print)

**Definition:** Colors created by mixing Cyan, Magenta, Yellow, and Black (K) inks.

**Range:** Each channel: 0-100%

**Why It Exists:**
- Ink on paper **absorbs** light (subtractive)
- Different from light emission (additive RGB)
- Professional printing uses CMYK

**Example:**
```
Pure Red:     CMYK(0%, 100%, 100%, 0%)   [No cyan, full magenta+yellow]
Pure Green:   CMYK(100%, 0%, 100%, 0%)   [Full cyan+yellow, no magenta]
Pure Blue:    CMYK(100%, 100%, 0%, 0%)   [Full cyan+magenta]
Black:        CMYK(0%, 0%, 0%, 100%)     [Pure black ink]
```

**When to Use:**
- Preparing figures for print publication
- Journal requires CMYK submission
- Proofing how colors will appear in print

**Important Note:**
- RGB → CMYK conversion can shift colors (especially bright blues, greens)
- Always preview in target color space
- Some RGB colors have no CMYK equivalent (out of gamut)

**Code Example (Python):**

```
from PIL import Image
import matplotlib.pyplot as plt

# Note: Matplotlib doesn't natively support CMYK, but we can demonstrate conversion

# Create RGB image
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# RGB version
rgb_colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
for i, color in enumerate(rgb_colors):
    rect = plt.Rectangle((i, 0), 1, 1, facecolor=color)
    axes[0].add_patch(rect)

axes[0].set_xlim(0, len(rgb_colors))
axes[0].set_ylim(0, 1)
axes[0].set_title('RGB (Screen Display)', fontsize=12, fontweight='bold')
axes[0].axis('off')

# Simulated "CMYK" appearance (with gamut limitations)
# In reality, you'd convert via color management
cmyk_note = """CMYK (Print):
- Some colors shift
- Gamut smaller than RGB
- Preview before submission"""

axes[1].text(0.5, 0.5, cmyk_note, ha='center', va='center',
             fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].axis('off')
axes[1].set_title('CMYK Considerations', fontsize=12, fontweight='bold')

plt.suptitle('RGB vs CMYK Color Spaces', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('rgb_vs_cmyk.png', dpi=300, bbox_inches='tight')
plt.close()
```

**Code Example (R):**

```
library(ggplot2)

# RGB to CMYK conversion approximation
rgb_to_cmyk <- function(r, g, b) {
  # Normalize RGB to 0-1
  r <- r/255; g <- g/255; b <- b/255

  # Calculate K (black)
  k <- 1 - max(c(r, g, b))

  if (k == 1) {
    return(c(C=0, M=0, Y=0, K=100))
  }

  # Calculate CMY
  c <- (1 - r - k) / (1 - k)
  m <- (1 - g - k) / (1 - k)
  y <- (1 - b - k) / (1 - k)

  return(c(C=round(c*100), M=round(m*100), Y=round(y*100), K=round(k*100)))
}

# Example colors
colors_demo <- data.frame(
  name = c('Red', 'Green', 'Blue', 'Cyan', 'Magenta', 'Yellow'),
  rgb = c('#FF0000', '#00FF00', '#0000FF', '#00FFFF', '#FF00FF', '#FFFF00'),
  r = c(255, 0, 0, 0, 255, 255),
  g = c(0, 255, 0, 255, 0, 255),
  b = c(0, 0, 255, 255, 255, 0)
)

# Add CMYK values
colors_demo$cmyk <- apply(colors_demo[, c('r', 'g', 'b')], 1,
                          function(x) paste(rgb_to_cmyk(x[1], x[2], x[3]), collapse=','))

# Plot
ggplot(colors_demo, aes(x = name, y = 1)) +
  geom_tile(aes(fill = rgb), color = 'black', width = 0.8, height = 0.6) +
  geom_text(aes(label = paste0('RGB: ', rgb, '\nCMYK: ', cmyk)),
            size = 2.5, color = 'white', fontface = 'bold') +
  scale_fill_identity() +
  labs(title = 'RGB to CMYK Conversion',
       subtitle = 'Note: Actual print colors may differ') +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = 'bold'),
        plot.subtitle = element_text(hjust = 0.5, size = 10, face = 'italic'))

ggsave('rgb_to_cmyk_conversion.png', width = 10, height = 4, dpi = 300)
```

---

#### HSV/HSB: Hue, Saturation, Value (Brightness)

**Definition:** A more intuitive color model based on human perception.

**Components:**
- **Hue (H)**: The "pure" color (0-360°)
  - 0° = Red, 120° = Green, 240° = Blue
  - Circular: 360° wraps back to red

- **Saturation (S)**: Color intensity (0-100%)
  - 0% = Gray (no color)
  - 100% = Pure, vivid color

- **Value/Brightness (V)**: Lightness (0-100%)
  - 0% = Black
  - 100% = Brightest version of that hue

**Advantages:**
- **Intuitive**: "I want a light, desaturated blue" → easy to specify
- **Easy manipulation**: Adjust brightness without changing hue
- **Good for color picking**: Natural way humans think about color

**Disadvantages:**
- Still not perceptually uniform
- Brightness perception varies with hue (yellow appears brighter than blue at same V)

**Code Example (Python):**

```
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import hsv_to_rgb
import numpy as np

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Demonstrate Hue variation (S=1, V=1)
hues = np.linspace(0, 1, 12, endpoint=False)
for i, hue in enumerate(hues):
    color = hsv_to_rgb([hue, 1.0, 1.0])
    rect = mpatches.Rectangle((i, 0), 1, 1, facecolor=color)
    axes[0].add_patch(rect)
    axes[0].text(i+0.5, 0.5, f'{int(hue*360)}°', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')

axes[0].set_xlim(0, 12)
axes[0].set_ylim(0, 1)
axes[0].set_title('Hue Variation (Saturation=100%, Value=100%)',
                  fontsize=12, fontweight='bold')
axes[0].axis('off')

# Demonstrate Saturation variation (H=0.6 [blue], V=1)
saturations = np.linspace(0, 1, 10)
for i, sat in enumerate(saturations):
    color = hsv_to_rgb([0.6, sat, 1.0])
    rect = mpatches.Rectangle((i, 0), 1, 1, facecolor=color)
    axes[1].add_patch(rect)
    axes[1].text(i+0.5, 0.5, f'{int(sat*100)}%', ha='center', va='center',
                fontsize=9, color='black' if sat < 0.5 else 'white', fontweight='bold')

axes[1].set_xlim(0, 10)
axes[1].set_ylim(0, 1)
axes[1].set_title('Saturation Variation (Hue=216°, Value=100%)',
                  fontsize=12, fontweight='bold')
axes[1].axis('off')

plt.suptitle('HSV Color Model', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('hsv_color_model.png', dpi=300, bbox_inches='tight')
plt.close()
```

**Code Example (R):**

```
library(ggplot2)
library(dplyr)

# Hue variation
hue_demo <- data.frame(
  x = 1:12,
  hue = seq(0, 330, by = 30),
  saturation = 1,
  value = 1
)

hue_demo$color <- hsv(hue_demo$hue/360, hue_demo$saturation, hue_demo$value)

p1 <- ggplot(hue_demo, aes(x = x, y = 1)) +
  geom_tile(aes(fill = color), color = 'black', width = 0.9, height = 0.8) +
  geom_text(aes(label = paste0(hue, '°')), color = 'white',
            fontface = 'bold', size = 3.5) +
  scale_fill_identity() +
  labs(title = 'Hue Variation (Saturation=100%, Value=100%)') +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = 'bold'))

# Saturation variation
sat_demo <- data.frame(
  x = 1:10,
  hue = 0.6,  # Blue
  saturation = seq(0, 1, length.out = 10),
  value = 1
)

sat_demo$color <- hsv(sat_demo$hue, sat_demo$saturation, sat_demo$value)
sat_demo$label <- paste0(round(sat_demo$saturation*100), '%')

p2 <- ggplot(sat_demo, aes(x = x, y = 1)) +
  geom_tile(aes(fill = color), color = 'black', width = 0.9, height = 0.8) +
  geom_text(aes(label = label),
            color = ifelse(sat_demo$saturation < 0.5, 'black', 'white'),
            fontface = 'bold', size = 3.5) +
  scale_fill_identity() +
  labs(title = 'Saturation Variation (Hue=216°, Value=100%)') +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = 'bold'))

# Combine plots
library(patchwork)
combined <- p1 / p2 + plot_annotation(
  title = 'HSV Color Model',
  theme = theme(plot.title = element_text(hjust = 0.5, size = 14, face = 'bold'))
)

ggsave('hsv_color_model.png', combined, width = 12, height = 6, dpi = 300)
```

---

#### CIELAB (L\*a\*b\*): Perceptually Uniform

**Definition:** A color space designed to be **perceptually uniform** — equal distances in the space correspond to equal perceived color differences.

**Components:**
- **L\* (Lightness)**: 0 (black) to 100 (white)
- **a\***: Green (-) to Red (+) axis
- **b\***: Blue (-) to Yellow (+) axis

**Why It Matters:**
```
Problem with RGB/HSV:
- RGB(100, 0, 0) → RGB(110, 0, 0): small perceptual change
- RGB(200, 0, 0) → RGB(210, 0, 0): even smaller perceptual change
→ Not uniform!

Solution with CIELAB:
- Moving 10 units in any direction produces approximately equal perceptual change
→ Uniform!
```

**Applications:**
- **Designing perceptually uniform colormaps** (e.g., viridis)
- **Measuring color similarity** (Delta E metric)
- **Quality control in printing**
- **Accessible color palette design**

**When to Use:**
- Creating custom sequential colormaps
- Ensuring smooth gradients
- Comparing colors quantitatively
- Accessibility testing

**Code Example (Python) - Perceptual Uniformity:**

```
import matplotlib.pyplot as plt
import numpy as np
from skimage import color

# Generate gradients in RGB and LAB
n_steps = 50

# RGB gradient (red to blue)
rgb_gradient = np.zeros((1, n_steps, 3))
rgb_gradient[0, :, 0] = np.linspace(1, 0, n_steps)  # Red decreases
rgb_gradient[0, :, 2] = np.linspace(0, 1, n_steps)  # Blue increases

# Convert to LAB
lab_gradient = color.rgb2lab(rgb_gradient[0])

# Calculate perceptual differences
rgb_diffs = np.diff(rgb_gradient[0], axis=0)
rgb_distances = np.linalg.norm(rgb_diffs, axis=1)

lab_diffs = np.diff(lab_gradient, axis=0)
lab_distances = np.linalg.norm(lab_diffs, axis=1)

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Show gradient
axes[0].imshow(rgb_gradient, aspect='auto')
axes[0].set_title('Red to Blue Gradient', fontsize=12, fontweight='bold')
axes[0].axis('off')

# RGB distances (non-uniform)
axes[1].plot(rgb_distances, 'o-', color='steelblue', linewidth=2)
axes[1].set_ylabel('Step Size (RGB)', fontsize=10)
axes[1].set_title('RGB: Non-Uniform Perceptual Steps', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)
axes[1].axhline(rgb_distances.mean(), color='red', linestyle='--',
                label=f'Mean={rgb_distances.mean():.3f}')
axes[1].legend()

# LAB distances (more uniform)
axes[2].plot(lab_distances, 'o-', color='coral', linewidth=2)
axes[2].set_xlabel('Gradient Step', fontsize=10)
axes[2].set_ylabel('Step Size (LAB)', fontsize=10)
axes[2].set_title('LAB: More Uniform Perceptual Steps', fontsize=12, fontweight='bold')
axes[2].grid(alpha=0.3)
axes[2].axhline(lab_distances.mean(), color='red', linestyle='--',
                label=f'Mean={lab_distances.mean():.3f}')
axes[2].legend()

plt.tight_layout()
plt.savefig('perceptual_uniformity.png', dpi=300, bbox_inches='tight')
plt.close()
```

**Code Example (R):**

```
library(ggplot2)
library(colorspace)
library(patchwork)

# Create gradients
n_steps <- 50

# RGB gradient (red to blue)
red_vals <- seq(1, 0, length.out = n_steps)
blue_vals <- seq(0, 1, length.out = n_steps)
rgb_colors <- rgb(red_vals, 0, blue_vals)

# Convert to LAB
rgb_mat <- col2rgb(rgb_colors)/255
lab_colors <- t(apply(rgb_mat, 2, function(x) {
  as(RGB(x[1], x[2], x[3]), "LAB")@coords
}))

# Calculate perceptual distances
rgb_distances <- sqrt(diff(red_vals)^2 + diff(blue_vals)^2)
lab_distances <- sqrt(rowSums(diff(lab_colors)^2))

# Plot gradient
grad_data <- data.frame(
  x = 1:n_steps,
  y = 1,
  color = rgb_colors
)

p1 <- ggplot(grad_data, aes(x = x, y = y, fill = color)) +
  geom_tile() +
  scale_fill_identity() +
  labs(title = 'Red to Blue Gradient') +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5, face = 'bold'))

# RGB distances
rgb_df <- data.frame(
  step = 1:(n_steps-1),
  distance = rgb_distances
)

p2 <- ggplot(rgb_df, aes(x = step, y = distance)) +
  geom_line(color = 'steelblue', size = 1) +
  geom_point(color = 'steelblue', size = 2) +
  geom_hline(yintercept = mean(rgb_distances), color = 'red',
             linetype = 'dashed', size = 1) +
  annotate('text', x = n_steps/2, y = mean(rgb_distances)*1.1,
           label = paste0('Mean = ', round(mean(rgb_distances), 3)),
           color = 'red', fontface = 'bold') +
  labs(title = 'RGB: Non-Uniform Perceptual Steps',
       y = 'Step Size (RGB)') +
  theme_classic() +
  theme(plot.title = element_text(face = 'bold'))

# LAB distances
lab_df <- data.frame(
  step = 1:(n_steps-1),
  distance = lab_distances
)

p3 <- ggplot(lab_df, aes(x = step, y = distance)) +
  geom_line(color = 'coral', size = 1) +
  geom_point(color = 'coral', size = 2) +
  geom_hline(yintercept = mean(lab_distances), color = 'red',
             linestyle = 'dashed', size = 1) +
  annotate('text', x = n_steps/2, y = mean(lab_distances)*1.1,
           label = paste0('Mean = ', round(mean(lab_distances), 1)),
           color = 'red', fontface = 'bold') +
  labs(title = 'LAB: More Uniform Perceptual Steps',
       x = 'Gradient Step', y = 'Step Size (LAB)') +
  theme_classic() +
  theme(plot.title = element_text(face = 'bold'))

# Combine
combined <- p1 / p2 / p3 + plot_annotation(
  title = 'Perceptual Uniformity: RGB vs CIELAB',
  theme = theme(plot.title = element_text(hjust = 0.5, size = 14, face = 'bold'))
)

ggsave('perceptual_uniformity.png', combined, width = 12, height = 10, dpi = 300)
```

---

### Summary: Which Color Space to Use?

| Task | Best Color Space | Why |
|------|-----------------|-----|
| **Digital display** | RGB | Native to screens |
| **Print publication** | CMYK | Required by printers |
| **Color picking/adjustment** | HSV | Intuitive manipulation |
| **Designing gradients** | CIELAB | Perceptually uniform |
| **Accessibility testing** | CIELAB | Quantify perceptual differences |
| **General programming** | RGB or Hex | Simplest, most compatible |

**Best Practice Workflow:**
```
1. Think in HSV (intuitive)
2. Implement in RGB (compatible)
3. Validate in CIELAB (uniform)
4. Convert to CMYK if printing (required)
```

---

### Exercise 2.1.1: Color Space Exploration

**Objective:** Experience how different color spaces represent the same color

**Task:**

1. **Pick a color** from a published figure you admire

2. **Convert it across spaces:**
   - Use a color picker to get RGB values
   - Convert to HSV: [https://www.rapidtables.com/convert/color/rgb-to-hsv.html](https://www.rapidtables.com/convert/color/rgb-to-hsv.html)
   - Convert to CIELAB: [https://colormine.org/convert/rgb-to-lab](https://colormine.org/convert/rgb-to-lab)
   - Convert to CMYK: [https://www.rapidtables.com/convert/color/rgb-to-cmyk.html](https://www.rapidtables.com/convert/color/rgb-to-cmyk.html)

3. **Document:**
   ```
   ```

4. **Reflection:**
   - Which representation is most intuitive to you?
   - If you wanted a "lighter version" of this color, which space makes that easiest?
   - How much does CMYK shift from RGB? (Compare visually)

---

## 2.2 Choosing Color Palettes

Now that we understand color spaces, let's apply them to the most critical decision: **choosing appropriate color palettes** for different types of data.

### The Three Palette Types

Every color palette falls into one of three categories, each suited to different data types:

#### 1. Sequential Palettes (Quantitative, Ordered Data)

**Use for:**
- Continuous numerical data
- Ordered categories (mild → moderate → severe)
- Any data with a natural progression

**Characteristics:**
- **Single hue** varying in lightness/saturation
- **Clear directionality** (low → high)
- **Perceptually ordered** (darker = more)

**Examples:**

```
Good Sequential Palettes:
✓ Light Blue → Dark Blue
✓ White → Red
✓ Light Gray → Black
✓ Viridis (yellow → blue, perceptually uniform)
```

**Bad Sequential Palettes:**

```
✗ Rainbow (red → orange → yellow → green → blue → purple)
  → No inherent ordering
  → Perceptual non-uniformity

✗ Random hues (blue → pink → orange)
  → Implies categorical data, not continuous
```

**Code Example (Python):**

```
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data (heatmap)
np.random.seed(42)
data = np.random.randn(10, 10).cumsum(axis=1)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Good sequential palettes
palettes = [
    ('Blues', 'Good: Single Hue (Blue)'),
    ('YlOrRd', 'Good: Yellow-Orange-Red'),
    ('viridis', 'Good: Viridis (Perceptually Uniform)'),
    ('jet', 'BAD: Rainbow (No Order)'),
    ('Set3', 'BAD: Qualitative (Wrong Type)'),
    ('RdYlGn', 'ACCEPTABLE: Diverging (If Zero is Meaningful)')
]

for ax, (cmap, title) in zip(axes.flat, palettes):
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    ax.set_title(title, fontsize=11, fontweight='bold',
                color='green' if 'Good' in title else 'red')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('Sequential Palettes: Good vs Bad', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sequential_palettes.png', dpi=300, bbox_inches='tight')
plt.close()
```

**Code Example (R):**

```
library(ggplot2)
library(viridis)
library(RColorBrewer)
library(patchwork)

# Generate sample data
set.seed(42)
data_matrix <- matrix(cumsum(rnorm(100)), nrow=10, ncol=10)
data_long <- expand.grid(x=1:10, y=1:10)
data_long$value <- as.vector(data_matrix)

# Helper function to create heatmap
create_heatmap <- function(data, palette, title, quality) {
  ggplot(data, aes(x=x, y=y, fill=value)) +
    geom_tile() +
    scale_fill_gradientn(colors=palette) +
    labs(title=title) +
    theme_void() +
    theme(plot.title=element_text(hjust=0.5, face='bold', size=10,
                                   color=ifelse(quality=='good', 'darkgreen', 'red')),
          legend.position='bottom',
          legend.key.width=unit(1, 'cm'),
          legend.key.height=unit(0.3, 'cm'))
}

# Create plots

# Good sequential palettes
p1 <- create_heatmap(data_long, brewer.pal(9, 'Blues'),
                     'Good: Single Hue (Blue)', 'good')
p2 <- create_heatmap(data_long, brewer.pal(9, 'YlOrRd'),
                     'Good: Yellow-Orange-Red', 'good')
p3 <- create_heatmap(data_long, viridis(100),
                     'Good: Viridis (Perceptually Uniform)', 'good')

# Bad sequential palettes
p4 <- create_heatmap(data_long, rainbow(100),
                     'BAD: Rainbow (No Order)', 'bad')
p5 <- create_heatmap(data_long, brewer.pal(8, 'Set3'),
                     'BAD: Qualitative (Wrong Type)', 'bad')
p6 <- create_heatmap(data_long, brewer.pal(11, 'RdYlGn'),
                     'ACCEPTABLE: Diverging (If Zero Meaningful)', 'acceptable')

# Combine
combined <- (p1 | p2 | p3) / (p4 | p5 | p6) +
  plot_annotation(
    title = 'Sequential Palettes: Good vs Bad',
    theme = theme(plot.title = element_text(hjust = 0.5, size = 14, face = 'bold'))
  )

ggsave('sequential_palettes.png', combined, width = 15, height = 8, dpi = 300)
```

---

#### 2. Diverging Palettes (Data with Meaningful Midpoint)

**Use for:**
- Data with a **critical central value** (zero, neutral, baseline)
- Deviations in two directions (positive/negative, above/below average)
- Comparisons showing increase vs. decrease

**Characteristics:**
- **Two contrasting hues** at extremes
- **Neutral color** (white, gray, beige) at center
- **Symmetric intensity** increasing toward both ends

**Examples:**
```
Good Diverging Palettes:
✓ Blue ← White → Red (cold/hot, negative/positive)
✓ Green ← Beige → Purple (gene downregulation/upregulation)
✓ RdBu (Red-Blue, colorblind-safe)
```

**When NOT to Use:**
```
✗ Data without a meaningful center point
  Example: Temperature in Kelvin (0K is absolute, not "neutral")

✗ Data that's inherently one-directional
  Example: Age (0-100, no negative ages)
```

**Real-World Example:**
```
Gene expression fold-change:
- Values: -5 to +5
- Center: 0 (no change)
- Negative: Downregulated (blue)
- Positive: Upregulated (red)
→ Perfect for diverging palette
```

**Code Example (Python):**

```
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Generate diverging data (centered at zero)
np.random.seed(42)
data_diverging = np.random.randn(10, 10) * 2  # Mean=0, symmetric

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Good diverging palettes
palettes_div = [
    ('RdBu_r', 'Good: Red-Blue (Symmetric)'),
    ('PiYG', 'Good: Pink-Green (Colorblind-Safe)'),
    ('coolwarm', 'Good: Cool-Warm')
]

for ax, (cmap, title) in zip(axes.flat, palettes_div):
    # Set symmetric limits around zero
    vmax = np.abs(data_diverging).max()
    im = ax.imshow(data_diverging, cmap=cmap, aspect='auto',
                   vmin=-vmax, vmax=vmax)  # Critical: symmetric limits
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')

    # Colorbar with zero highlighted
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.ax.axhline(0, color='black', linewidth=2)  # Highlight zero

plt.suptitle('Diverging Palettes: Critical Zero Point',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('diverging_palettes.png', dpi=300, bbox_inches='tight')
plt.close()
```

**Code Example (R):**

```
library(ggplot2)
library(RColorBrewer)
library(patchwork)

# Generate diverging data
set.seed(42)
data_div <- matrix(rnorm(100) * 2, nrow=10, ncol=10)
data_long_div <- expand.grid(x=1:10, y=1:10)
data_long_div$value <- as.vector(data_div)

# Helper function for diverging heatmap
create_div_heatmap <- function(data, palette_name, title) {
  # Get symmetric limits
  max_abs <- max(abs(data$value))

  ggplot(data, aes(x=x, y=y, fill=value)) +
    geom_tile() +
    scale_fill_gradientn(colors=brewer.pal(11, palette_name),
                         limits=c(-max_abs, max_abs)) +  # Symmetric limits
    labs(title=title) +
    theme_void() +
    theme(plot.title=element_text(hjust=0.5, face='bold', size=10),
          legend.position='bottom',
          legend.key.width=unit(1.5, 'cm'),
          legend.key.height=unit(0.3, 'cm')) +
    # Highlight zero in legend
    geom_hline(yintercept=0, color='black', size=1)
}

# Create plots
p1 <- create_div_heatmap(data_long_div, 'RdBu', 'Good: Red-Blue (Symmetric)')
p2 <- create_div_heatmap(data_long_div, 'PiYG', 'Good: Pink-Green (Colorblind-Safe)')
p3 <- create_div_heatmap(data_long_div, 'BrBG', 'Good: Brown-Blue-Green')

combined <- p1 | p2 | p3
combined <- combined + plot_annotation(
  title = 'Diverging Palettes: Critical Zero Point',
  theme = theme(plot.title = element_text(hjust = 0.5, size = 14, face = 'bold'))
)

ggsave('diverging_palettes.png', combined, width = 15, height = 5, dpi = 300)
```

---

#### 3. Qualitative Palettes (Categorical Data)

**Use for:**
- **Unordered categories** (species, treatments, locations)
- **Nominal data** (no inherent ordering)
- **Group comparisons** (control vs. experimental groups)

**Characteristics:**
- **Distinct hues** (maximally different colors)
- **Similar lightness** (no implied hierarchy)
- **Visually balanced** (no one color dominates)

**Rules:**

```
✓ Limit to 6-8 categories maximum
  → Beyond this, colors become too similar

✓ Use colorblind-safe combinations
  → Avoid red-green only distinctions

✓ Consider adding shape/line style redundancy
  → Ensures accessibility
```

**Good Qualitative Palettes:**

```
✓ ColorBrewer 'Set2' (8 colors, colorblind-friendly)
✓ ColorBrewer 'Dark2' (8 colors, good contrast)
✓ Okabe-Ito palette (8 colors, designed for colorblindness)
✓ Custom palettes from your field's conventions
```

**Bad Qualitative Palettes:**

```
✗ Rainbow (implies ordering that doesn't exist)
✗ Sequential palette (Blues) for categories (implies hierarchy)
✗ Too many similar colors (light blue, medium blue, cyan, teal...)
```

**Code Example (Python):**

```
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Sample categorical data
np.random.seed(42)
categories = ['Group A', 'Group B', 'Group C', 'Group D', 'Group E']
x = np.repeat(np.arange(50), len(categories))
y = np.tile(np.arange(len(categories)), 50) + np.random.randn(len(categories)*50)*0.3
category = np.tile(categories, 50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Good qualitative palette (Okabe-Ito, colorblind-safe)
okabe_ito = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2',
             '#D55E00', '#CC79A7', '#000000']

axes[0, 0].scatter(x, y, c=[okabe_ito[i%8] for i, _ in enumerate(category)],
                   alpha=0.6, s=50)
axes[0, 0].set_title('GOOD: Okabe-Ito Palette\n(Colorblind-Safe, Distinct Hues)',
                     fontsize=11, fontweight='bold', color='green')
axes[0, 0].set_xlabel('X Variable')
axes[0, 0].set_ylabel('Y Variable')

# Good: ColorBrewer Set2
set2_colors = sns.color_palette('Set2', len(categories))
for i, cat in enumerate(categories):
    mask = np.array(category) == cat
    axes[0, 1].scatter(x[mask], y[mask], color=set2_colors[i],
                       label=cat, alpha=0.6, s=50)
axes[0, 1].set_title('GOOD: ColorBrewer Set2\n(Balanced, Distinct)',
                     fontsize=11, fontweight='bold', color='green')
axes[0, 1].legend(loc='upper right', frameon=True)
axes[0, 1].set_xlabel('X Variable')
axes[0, 1].set_ylabel('Y Variable')

# Bad: Rainbow (implies false ordering)
rainbow_colors = plt.cm.rainbow(np.linspace(0, 1, len(categories)))
for i, cat in enumerate(categories):
    mask = np.array(category) == cat
    axes[1, 0].scatter(x[mask], y[mask], color=rainbow_colors[i],
                       label=cat, alpha=0.6, s=50)
axes[1, 0].set_title('BAD: Rainbow\n(Implies Ordering, Not Colorblind-Safe)',
                     fontsize=11, fontweight='bold', color='red')
axes[1, 0].legend(loc='upper right', frameon=True)
axes[1, 0].set_xlabel('X Variable')
axes[1, 0].set_ylabel('Y Variable')

# Bad: Too similar colors (all blues)
blues_colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(categories)))
for i, cat in enumerate(categories):
    mask = np.array(category) == cat
    axes[1, 1].scatter(x[mask], y[mask], color=blues_colors[i],
                       label=cat, alpha=0.6, s=50)
axes[1, 1].set_title('BAD: Sequential for Categorical\n(Similar Colors, Implies Hierarchy)',
                     fontsize=11, fontweight='bold', color='red')
axes[1, 1].legend(loc='upper right', frameon=True)
axes[1, 1].set_xlabel('X Variable')
axes[1, 1].set_ylabel('Y Variable')

plt.suptitle('Qualitative Palettes: Good vs Bad', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('qualitative_palettes.png', dpi=300, bbox_inches='tight')
plt.close()
```

**Code Example (R):**

```
library(ggplot2)
library(RColorBrewer)
library(patchwork)

# Sample categorical data
set.seed(42)
n_per_group <- 50
categories <- c('Group A', 'Group B', 'Group C', 'Group D', 'Group E')

data_cat <- data.frame(
  x = rep(1:n_per_group, each=length(categories)),
  y = rep(1:length(categories), n_per_group) + rnorm(n_per_group*length(categories), 0, 0.3),
  category = rep(categories, n_per_group)
)

# Okabe-Ito colorblind-safe palette
okabe_ito <- c('#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2',
               '#D55E00', '#CC79A7', '#000000')

# Good: Okabe-Ito
p1 <- ggplot(data_cat, aes(x=x, y=y, color=category)) +
  geom_point(alpha=0.6, size=2) +
  scale_color_manual(values=okabe_ito[1:length(categories)]) +
  labs(title='GOOD: Okabe-Ito Palette\n(Colorblind-Safe, Distinct Hues)',
       x='X Variable', y='Y Variable') +
  theme_classic() +
  theme(plot.title=element_text(hjust=0.5, face='bold', color='darkgreen', size=10),
        legend.position='right')

# Good: ColorBrewer Set2
p2 <- ggplot(data_cat, aes(x=x, y=y, color=category)) +
  geom_point(alpha=0.6, size=2) +
  scale_color_brewer(palette='Set2') +
  labs(title='GOOD: ColorBrewer Set2\n(Balanced, Distinct)',
       x='X Variable', y='Y Variable') +
  theme_classic() +
  theme(plot.title=element_text(hjust=0.5, face='bold', color='darkgreen', size=10),
        legend.position='right')

# Bad: Rainbow
p3 <- ggplot(data_cat, aes(x=x, y=y, color=category)) +
  geom_point(alpha=0.6, size=2) +
  scale_color_manual(values=rainbow(length(categories))) +
  labs(title='BAD: Rainbow\n(Implies Ordering, Not Colorblind-Safe)',
       x='X Variable', y='Y Variable') +
  theme_classic() +
  theme(plot.title=element_text(hjust=0.5, face='bold', color='red', size=10),
        legend.position='right')

# Bad: Sequential (Blues) for categorical
p4 <- ggplot(data_cat, aes(x=x, y=y, color=category)) +
  geom_point(alpha=0.6, size=2) +
  scale_color_brewer(palette='Blues') +
  labs(title='BAD: Sequential for Categorical\n(Similar Colors, Implies Hierarchy)',
       x='X Variable', y='Y Variable') +
  theme_classic() +
  theme(plot.title=element_text(hjust=0.5, face='bold', color='red', size=10),
        legend.position='right')

# Combine
combined <- (p1 | p2) / (p3 | p4) + plot_annotation(
  title = 'Qualitative Palettes: Good vs Bad',
  theme = theme(plot.title = element_text(hjust = 0.5, size = 14, face = 'bold'))
)

ggsave('qualitative_palettes.png', combined, width = 14, height = 10, dpi = 300)
```

---

### Palette Selection Decision Tree

```
START: What type of data do I have?

├─ QUANTITATIVE/CONTINUOUS
│   ├─ Has meaningful midpoint (zero, baseline)?
│   │   └─ YES → DIVERGING PALETTE
│   │       Examples: RdBu, PiYG, BrBG
│   │
│   └─ NO → SEQUENTIAL PALETTE
│       Examples: Blues, Viridis, YlOrRd
│
└─ CATEGORICAL/NOMINAL
    ├─ Ordered categories (mild/moderate/severe)?
    │   └─ YES → SEQUENTIAL PALETTE
    │       (Treat as ordinal data)
    │
    └─ NO → QUALITATIVE PALETTE
        Examples: Set2, Dark2, Okabe-Ito

        Special consideration:
        - If >8 categories → Consider splitting figure
        - Always check colorblind simulation
        - Add shape/line redundancy
```

---

### Exercise 2.2.1: Palette Type Classification

**Objective:** Practice identifying appropriate palette types

**Instructions:**

For each dataset below, identify:
1. Data type (quantitative continuous, ordinal, nominal)
2. Appropriate palette type (sequential, diverging, qualitative)
3. Specific palette recommendation
4. Why alternatives would be wrong

**Datasets:**

**A. Temperature anomaly** (deviation from 20th century average)
- Values: -2°C to +2°C
- Represents: Climate change warming/cooling

**B. Species distribution** across 5 habitat types
- Categories: Forest, Grassland, Desert, Wetland, Tundra
- No inherent ordering

**C. Disease progression** stages
- Categories: Healthy, At-risk, Early disease, Advanced disease
- Clear progression

**D. Population density** (people per km²)
- Values: 0 to 10,000
- All positive, no meaningful center

**E. Stock price change** (% change from yesterday)
- Values: -15% to +15%
- Zero = no change

**Your answers should look like:**
```
Dataset A:
- Data type: Quantitative continuous with meaningful zero
- Palette type: Diverging
- Recommendation: RdBu_r (blue=cooling, red=warming)
- Why not sequential: Would lose critical distinction between warming/cooling
- Why not qualitative: Data is continuous, not categorical
```

---


-----


## 2.3 Light vs. Dark Background

### The Context-Driven Choice

The choice between light and dark themes isn't about personal preference—it's about **medium, venue, and function**. Scientific publications have specific requirements, and understanding when each theme is appropriate is crucial for effective communication.

### Publication Standard: Light Backgrounds (Default)

**Why light themes dominate scientific literature:**

1. **Print legacy**: Scientific journals evolved from print media where:
   - Black ink on white paper is most economical
   - High contrast ensures legibility
   - Grayscale reproduction is reliable

2. **Reading comprehension**: Studies show:
   - Dark text on light backgrounds reduces eye strain for extended reading
   - Better for detailed analysis and data interpretation
   - Maintains performance across various lighting conditions

3. **Journal requirements**: Most high-impact journals explicitly require:
   - White or light gray backgrounds
   - Black text and axis lines
   - Conservative, professional appearance

**Standard Scientific Figure Format:**
```
✓ White background
✓ Black or dark gray text
✓ Colored data elements (but limited palette)
✓ Light gray gridlines (if used)
✓ Clean, minimal aesthetic
```

---

### When Dark Themes Are Appropriate

**Acceptable Use Cases:**

**1. Presentations and Posters**
```
Why it works:
- Projected in dark/dim rooms
- High contrast improves visibility from distance
- More dramatic visual impact
- Reduces screen glare

Requirements:
- Still maintain high contrast
- Use lighter text colors
- Test in actual presentation conditions
```

**2. Supplementary Digital Materials**
```
Why it works:
- Interactive web-based figures
- Video presentations
- Screen-based analysis tools

Requirements:
- Provide light theme alternative
- Ensure all elements remain visible
- Test on multiple devices
```

**3. Specific Data Visualization Contexts**
```
Astronomical imaging: Stars on black sky (natural representation)
Fluorescence microscopy: Bright signals on dark background (matches actual imaging)
Network visualizations: Sometimes clearer on dark backgrounds

But even here: Consider converting to light theme for publication
```

---

### The Color Restraint Principle for Publications

**Critical Rule: Use color sparingly and purposefully**

In scientific publications, every color must have a **functional reason**, not merely aesthetic appeal. This principle stems from:

1. **Cognitive load**: Too many colors overwhelm readers
2. **Reproducibility**: Simpler color schemes are easier to recreate
3. **Accessibility**: Fewer colors reduce colorblind accessibility issues
4. **Professional standards**: Conservative color use signals scientific rigor
5. **Print costs**: Historically, color pages cost more (legacy influence)

---

### The "3-Color Rule" for Scientific Figures

**Guideline: Aim for 3 colors maximum per figure (excluding grayscale)**

**Why this works:**

**Example: Treatment Comparison Figure**

```
✓ GOOD (3 colors):
  - Control group: Gray (#808080)
  - Treatment A: Blue (#2E86AB)
  - Treatment B: Red (#A23B72)
  - Background: White
  - Text/axes: Black
  - Gridlines: Light gray (#D3D3D3)

→ Clear, distinct, professional
→ Easy to distinguish in legend
→ Works in grayscale (shades differ)
```

**Bad Example: Color Overload**

```
✗ BAD (8+ colors):
  - 8 different treatment groups in rainbow colors
  - Multiple colored gridlines
  - Colored background
  - Multicolored title

→ Visually chaotic
→ Hard to match legend to data
→ Loses impact of individual colors
```

---

### Logical Color Schemes: Semantic Consistency

**Colors should carry semantic meaning consistently across your entire manuscript**

**Principle: Same concept = Same color throughout all figures**

**Example Consistency Rules:**

**Temperature-related data:**

```
✓ Consistent across all figures:
  - Cold/low temperature: Blue (#3498DB)
  - Hot/high temperature: Red (#E74C3C)
  - Neutral: Gray (#95A5A6)

✗ Inconsistent (confusing):
  - Figure 1: Cold=Blue, Hot=Red
  - Figure 2: Cold=Green, Hot=Orange
  - Figure 3: Cold=Purple, Hot=Yellow
→ Reader must relearn color meaning each time
```

**Biological example:**

```
Consistent color scheme for cell types across all figures:
  - Neurons: Purple (#9B59B6)
  - Astrocytes: Green (#27AE60)
  - Microglia: Orange (#E67E22)

Applied to:
  - Figure 1: Immunohistochemistry images
  - Figure 2: Flow cytometry plots
  - Figure 3: Gene expression heatmaps
  - Figure 4: Quantification bar charts

→ Immediate recognition, reduced cognitive load
```

---

### Field-Specific Color Conventions

**Many scientific fields have established color conventions—follow them unless you have strong justification**

**Molecular Biology:**

```
- DNA: Blue
- RNA: Red
- Protein: Green or purple
- Upregulated genes: Red
- Downregulated genes: Blue/green
```

**Neuroscience:**

```
- Excitatory neurons: Red
- Inhibitory neurons: Blue
- Dendrites: Green
- Axons: Red or blue
```

**Climate Science:**

```
- Warming: Red/orange
- Cooling: Blue
- Land: Brown/green
- Water: Blue
```

**Medical Imaging:**

```
- PET scans: Rainbow (hot) scale
- MRI: Grayscale
- Functional activation: Hot colors on grayscale
```

**Respecting conventions:**
```
✓ Advantages:
  - Immediate interpretation by experts
  - Consistent with literature
  - Reduces confusion

✗ Breaking conventions:
  - Only if you have strong perceptual/accessibility reason
  - Must explain explicitly in caption
  - Risk confusing your audience
```

---

### Code Examples: Light Theme with Minimal Color

**Python Example: Publication-Ready Light Theme**

```
import matplotlib.pyplot as plt
import numpy as np

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2,
    'grid.color': '#D3D3D3',
    'grid.linewidth': 0.8,
    'grid.alpha': 0.5
})

# Sample data (treatment comparison)
np.random.seed(42)
time = np.arange(0, 24, 1)
control = 100 + np.cumsum(np.random.randn(len(time)) * 2)
treatment_a = 100 + np.cumsum(np.random.randn(len(time)) * 2 + 0.5)
treatment_b = 100 + np.cumsum(np.random.randn(len(time)) * 2 + 1.0)

# Define consistent color palette (3 colors only)
COLOR_CONTROL = '#808080'      # Gray
COLOR_TREATMENT_A = '#2E86AB'  # Blue
COLOR_TREATMENT_B = '#A23B72'  # Purple-red

fig, ax = plt.subplots(figsize=(7, 4.5))

# Plot data with ONLY 3 colors
ax.plot(time, control, color=COLOR_CONTROL, linewidth=2.5,
        label='Control', marker='o', markersize=4, markevery=3)
ax.plot(time, treatment_a, color=COLOR_TREATMENT_A, linewidth=2.5,
        label='Treatment A', marker='s', markersize=4, markevery=3)
ax.plot(time, treatment_b, color=COLOR_TREATMENT_B, linewidth=2.5,
        label='Treatment B', marker='^', markersize=4, markevery=3)

# Minimal, functional styling
ax.set_xlabel('Time (hours)', fontweight='bold')
ax.set_ylabel('Cell Viability (%)', fontweight='bold')
ax.set_title('Effect of Treatments on Cell Viability', fontweight='bold', pad=15)

# Legend: simple, unobtrusive
ax.legend(loc='upper left', frameon=True, facecolor='white',
          edgecolor='#333333', framealpha=1.0)

# Remove top and right spines (cleaner look)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Subtle gridlines (not distracting)
ax.grid(True, alpha=0.3, linewidth=0.7, linestyle='--')

plt.tight_layout()
plt.savefig('light_theme_minimal_color.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()

print("Figure saved with:")
print(f"  - Background: White")
print(f"  - Colors used: 3 (Gray, Blue, Purple-red)")
print(f"  - Text: Black")
print(f"  - Grid: Light gray, subtle")
```

---

**R Example: Publication-Ready Light Theme**

```
library(ggplot2)
library(dplyr)

# Sample data
set.seed(42)
time <- 0:23
data <- data.frame(
  time = rep(time, 3),
  viability = c(
    100 + cumsum(rnorm(length(time), 0, 2)),           # Control
    100 + cumsum(rnorm(length(time), 0.5, 2)),         # Treatment A
    100 + cumsum(rnorm(length(time), 1.0, 2))          # Treatment B
  ),
  group = rep(c('Control', 'Treatment A', 'Treatment B'), each=length(time))
)

# Consistent color palette (3 colors only)
COLOR_CONTROL <- '#808080'      # Gray
COLOR_TREATMENT_A <- '#2E86AB'  # Blue
COLOR_TREATMENT_B <- '#A23B72'  # Purple-red

colors_palette <- c('Control' = COLOR_CONTROL,
                    'Treatment A' = COLOR_TREATMENT_A,
                    'Treatment B' = COLOR_TREATMENT_B)

# Create publication-ready plot
p <- ggplot(data, aes(x = time, y = viability, color = group, shape = group)) +
  geom_line(size = 1.2) +
  geom_point(size = 2.5, data = data %>% filter(time %% 3 == 0)) +  # Every 3rd point

  # Apply color palette
  scale_color_manual(values = colors_palette) +
  scale_shape_manual(values = c(16, 15, 17)) +  # Circle, square, triangle

  # Labels
  labs(
    x = 'Time (hours)',
    y = 'Cell Viability (%)',
    title = 'Effect of Treatments on Cell Viability',
    color = NULL,
    shape = NULL
  ) +

  # Theme: clean, light background
  theme_classic(base_size = 11, base_family = 'Arial') +
  theme(
    # White background (publication standard)
    plot.background = element_rect(fill = 'white', color = NA),
    panel.background = element_rect(fill = 'white', color = NA),

    # Subtle grid
    panel.grid.major = element_line(color = '#D3D3D3', size = 0.3, linetype = 'dashed'),
    panel.grid.minor = element_blank(),

    # Bold axis labels
    axis.title = element_text(face = 'bold', size = 11),
    axis.text = element_text(size = 9, color = '#333333'),

    # Title
    plot.title = element_text(face = 'bold', size = 12, hjust = 0, margin = margin(b = 10)),

    # Legend: simple, unobtrusive
    legend.position = c(0.15, 0.85),
    legend.background = element_rect(fill = 'white', color = '#333333', size = 0.5),
    legend.key = element_rect(fill = 'white'),
    legend.text = element_text(size = 9),
    legend.margin = margin(t = 5, r = 5, b = 5, l = 5),

    # Remove top/right borders
    axis.line = element_line(color = '#333333', size = 0.7),
    panel.border = element_blank()
  )

# Save
ggsave('light_theme_minimal_color.png', p, width = 7, height = 4.5,
       dpi = 300, bg = 'white')

cat("Figure saved with:\n")
cat("  - Background: White\n")
cat("  - Colors used: 3 (Gray, Blue, Purple-red)\n")
cat("  - Text: Black\n")
cat("  - Grid: Light gray, subtle\n")
```

---

### Dark Theme: When and How (Presentations Only)

**Python Example: Dark Theme for Presentation**

```
import matplotlib.pyplot as plt
import numpy as np

# Dark theme settings (for presentations ONLY)
plt.style.use('dark_background')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,  # Larger for visibility
    'figure.facecolor': '#1E1E1E',
    'axes.facecolor': '#1E1E1E',
    'axes.edgecolor': '#CCCCCC',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'grid.color': '#444444',
    'grid.alpha': 0.5
})

# Same data as before
np.random.seed(42)
time = np.arange(0, 24, 1)
control = 100 + np.cumsum(np.random.randn(len(time)) * 2)
treatment_a = 100 + np.cumsum(np.random.randn(len(time)) * 2 + 0.5)
treatment_b = 100 + np.cumsum(np.random.randn(len(time)) * 2 + 1.0)

# LIGHTER colors for dark background (higher saturation/brightness)
COLOR_CONTROL_DARK = '#B0B0B0'      # Light gray
COLOR_TREATMENT_A_DARK = '#5DADE2'  # Light blue
COLOR_TREATMENT_B_DARK = '#EC7063'  # Light red

fig, ax = plt.subplots(figsize=(10, 6))  # Larger for presentations

ax.plot(time, control, color=COLOR_CONTROL_DARK, linewidth=3,
        label='Control', marker='o', markersize=6, markevery=3)
ax.plot(time, treatment_a, color=COLOR_TREATMENT_A_DARK, linewidth=3,
        label='Treatment A', marker='s', markersize=6, markevery=3)
ax.plot(time, treatment_b, color=COLOR_TREATMENT_B_DARK, linewidth=3,
        label='Treatment B', marker='^', markersize=6, markevery=3)

ax.set_xlabel('Time (hours)', fontweight='bold', fontsize=14)
ax.set_ylabel('Cell Viability (%)', fontweight='bold', fontsize=14)
ax.set_title('Effect of Treatments on Cell Viability\n(Presentation Version)',
             fontweight='bold', fontsize=16, color='white', pad=20)

ax.legend(loc='upper left', fontsize=12, frameon=True,
          facecolor='#2C2C2C', edgecolor='#CCCCCC')

ax.grid(True, alpha=0.3, linewidth=0.8)

plt.tight_layout()
plt.savefig('dark_theme_presentation.png', dpi=150,
            bbox_inches='tight', facecolor='#1E1E1E')
plt.close()

print("Dark theme figure (PRESENTATION ONLY) saved")
print("Note: Convert to light theme for publication submission")
```

---

**R Example: Dark Theme for Presentation**

```
library(ggplot2)

# Same data
set.seed(42)
time <- 0:23
data <- data.frame(
  time = rep(time, 3),
  viability = c(
    100 + cumsum(rnorm(length(time), 0, 2)),
    100 + cumsum(rnorm(length(time), 0.5, 2)),
    100 + cumsum(rnorm(length(time), 1.0, 2))
  ),
  group = rep(c('Control', 'Treatment A', 'Treatment B'), each=length(time))
)

# Lighter colors for dark background
COLOR_CONTROL_DARK <- '#B0B0B0'      # Light gray
COLOR_TREATMENT_A_DARK <- '#5DADE2'  # Light blue
COLOR_TREATMENT_B_DARK <- '#EC7063'  # Light red

colors_dark <- c('Control' = COLOR_CONTROL_DARK,
                 'Treatment A' = COLOR_TREATMENT_A_DARK,
                 'Treatment B' = COLOR_TREATMENT_B_DARK)

# Dark theme plot
p_dark <- ggplot(data, aes(x = time, y = viability, color = group, shape = group)) +
  geom_line(size = 1.5) +
  geom_point(size = 3.5, data = data %>% filter(time %% 3 == 0)) +

  scale_color_manual(values = colors_dark) +
  scale_shape_manual(values = c(16, 15, 17)) +

  labs(
    x = 'Time (hours)',
    y = 'Cell Viability (%)',
    title = 'Effect of Treatments on Cell Viability\n(Presentation Version)',
    color = NULL,
    shape = NULL
  ) +

  # Dark theme
  theme_minimal(base_size = 14, base_family = 'Arial') +
  theme(
    # Dark background
    plot.background = element_rect(fill = '#1E1E1E', color = NA),
    panel.background = element_rect(fill = '#1E1E1E', color = NA),

    # Light text
    text = element_text(color = 'white'),
    axis.text = element_text(color = 'white', size = 12),
    axis.title = element_text(color = 'white', face = 'bold', size = 14),
    plot.title = element_text(color = 'white', face = 'bold', size = 16, hjust = 0.5),

    # Subtle grid
    panel.grid.major = element_line(color = '#444444', size = 0.4),
    panel.grid.minor = element_blank(),

    # Legend
    legend.position = c(0.15, 0.85),
    legend.background = element_rect(fill = '#2C2C2C', color = '#CCCCCC', size = 0.5),
    legend.text = element_text(color = 'white', size = 12),
    legend.key = element_rect(fill = '#2C2C2C')
  )

ggsave('dark_theme_presentation.png', p_dark, width = 10, height = 6,
       dpi = 150, bg = '#1E1E1E')

cat("Dark theme figure (PRESENTATION ONLY) saved\n")
cat("Note: Convert to light theme for publication submission\n")
```

---

### Publication Submission Checklist: Color & Theme

**Before submitting your manuscript, verify:**

- [ ] **Background is white or light gray** (not dark)
- [ ] **Maximum 3-4 distinct colors** per figure (excluding grayscale)
- [ ] **Color consistency** across all figures (same concept = same color)
- [ ] **Semantic logic** (colors match intuitive meanings)
- [ ] **Field conventions respected** (unless explicitly justified)
- [ ] **Text is black** or very dark gray (#333333)
- [ ] **Gridlines are subtle** (light gray, thin, low opacity)
- [ ] **No decorative colors** (every color has functional purpose)
- [ ] **Works in grayscale** (test by converting to black & white)
- [ ] **Colorblind accessible** (test with simulation tools)

---

### Exercise 2.3.1: Color Restraint Practice

**Objective:** Practice creating publication-ready figures with minimal, logical color use

**Task:**

1. **Find a figure** from your own work or literature that uses >5 colors

2. **Redesign it** following these constraints:
   - Maximum 3 distinct colors (plus grayscale)
   - Light background (white)
   - Consistent color logic (explain your color choices)
   - No decorative color

3. **Document your choices:**
   
   Original figure: [description, number of colors used]

   Redesign:
   - Color 1: [Hex code] - Used for: [data/concept] - Reason: [explain]
   - Color 2: [Hex code] - Used for: [data/concept] - Reason: [explain]
   - Color 3: [Hex code] - Used for: [data/concept] - Reason: [explain]
   - Grayscale: Used for: [supporting elements]

   Logic: [Explain the semantic consistency of your color choices]

   Improvements: [What became clearer with fewer colors]
   


4. **Test accessibility:**
   - Convert to grayscale: Do distinctions remain clear?
   - Use colorblind simulator: Is information preserved?

---

**End of Section 2.3**

**Key Takeaways:**
- **Default to light themes** for all publications
- **Dark themes** only for presentations/posters
- **3-color maximum** (excluding grayscale) for clarity
- **Every color must have semantic meaning**
- **Consistency across manuscript** is mandatory
- **Respect field conventions** unless you have strong justification
---

## 2.4 Light vs. Deep Colors: Saturation and Intensity

### Understanding Color Intensity in Scientific Figures

In scientific visualization, the choice between **light (desaturated)** and **deep (saturated)** colors isn't about aesthetics—it's about **visual hierarchy**, **emphasis**, and **cognitive processing**. This section addresses when to use vivid, saturated colors versus pale, muted tones.

### The Saturation Spectrum

**Color saturation** refers to the intensity or purity of a color:

- **High saturation (deep colors)**: Vivid, pure, intense (e.g., pure red #FF0000)
- **Low saturation (light colors)**: Pale, muted, washed-out (e.g., light pink #FFB3BA)
- **Zero saturation**: Grayscale only

**In HSV/HSB color model:**
```
Saturation = 0%: Pure gray (no color)
Saturation = 50%: Muted, pastel-like
Saturation = 100%: Maximum vividness
```

---

### The Visual Hierarchy Principle: Deep Colors for Focus

**Rule: Use deep, saturated colors sparingly for PRIMARY focus only**

**Why this matters:**
1. **Pre-attentive attention**: Saturated colors grab attention immediately
2. **Visual weight**: Deep colors appear "heavier" and more important
3. **Cognitive load**: Too many saturated colors cause visual fatigue
4. **Professional standards**: Scientific figures should be clear, not garish

**The Hierarchy Strategy:**

```
LEVEL 1 (Primary focus): Deep, saturated colors
└─ The key finding, most important data
└─ Examples: Significant results, highlighted treatment group

LEVEL 2 (Supporting context): Medium saturation
└─ Important but secondary data
└─ Examples: Reference groups, all experimental conditions

LEVEL 3 (Background/infrastructure): Light, desaturated colors
└─ Supporting elements that shouldn't distract
└─ Examples: Gridlines, reference regions, non-significant data
```

---

### Practical Application Examples

#### Example 1: Highlighting Significant Results

**Scenario:** Clinical trial with 8 treatment groups, only 2 show significant improvement

**Bad Approach (Equal saturation):**

```
✗ All 8 groups in equally bright, saturated colors
  → No visual priority
  → Reader must read legend and statistics to find key results
  → Cognitive overload from too many bright colors
```

**Good Approach (Saturation hierarchy):**

```
✓ 2 significant groups: Deep, saturated colors (e.g., #E63946 red, #2E86AB blue)
✓ 6 non-significant groups: Light, desaturated gray (#CCCCCC)
  → Immediate visual focus on significant findings
  → Clean, uncluttered appearance
  → Professional and publication-ready
```

**Code Example (Python):**

```
import matplotlib.pyplot as plt
import numpy as np

# Sample data: 8 treatment groups
np.random.seed(42)
groups = [f'Group {i+1}' for i in range(8)]
values = [15, 12, 28, 10, 14, 25, 11, 13]  # Groups 3 and 6 are significant
errors = [2, 2.5, 3, 2, 2.5, 3.5, 2, 2.5]

# Identify significant groups
significant = [2, 5]  # Indices of Group 3 and Group 6

# GOOD: Deep colors for significant, light for non-significant
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# BAD approach: All equally saturated
colors_bad = ['#E63946', '#F77F00', '#FCBF49', '#06A77D',
              '#4ECDC4', '#2E86AB', '#8338EC', '#A23B72']

axes[0].bar(groups, values, color=colors_bad, edgecolor='black', linewidth=1.2)
axes[0].errorbar(groups, values, yerr=errors, fmt='none', ecolor='black', capsize=5)
axes[0].set_ylabel('Response (arbitrary units)', fontsize=11, fontweight='bold')
axes[0].set_title('BAD: All Colors Equally Saturated\n(No visual priority)',
                  fontsize=12, fontweight='bold', color='red')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# GOOD approach: Deep colors only for significant
colors_good = ['#D3D3D3'] * 8  # All light gray by default
colors_good[2] = '#E63946'     # Group 3: Deep red
colors_good[5] = '#2E86AB'     # Group 6: Deep blue

axes[1].bar(groups, values, color=colors_good, edgecolor='black', linewidth=1.2)
axes[1].errorbar(groups, values, yerr=errors, fmt='none', ecolor='black', capsize=5)
axes[1].set_ylabel('Response (arbitrary units)', fontsize=11, fontweight='bold')
axes[1].set_title('GOOD: Deep Colors for Significant Only\n(Clear visual hierarchy)',
                  fontsize=12, fontweight='bold', color='green')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Add significance markers
for idx in significant:
    axes[1].text(idx, values[idx] + errors[idx] + 2, '***',
                ha='center', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('saturation_hierarchy.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R):**

```
library(ggplot2)
library(dplyr)

# Sample data
groups <- paste0('Group ', 1:8)
values <- c(15, 12, 28, 10, 14, 25, 11, 13)
errors <- c(2, 2.5, 3, 2, 2.5, 3.5, 2, 2.5)
significant <- c(3, 6)  # Group 3 and 6 are significant

data <- data.frame(
  group = factor(groups, levels = groups),
  value = values,
  error = errors,
  is_significant = 1:8 %in% significant
)

# BAD: All equally saturated
colors_bad <- c('#E63946', '#F77F00', '#FCBF49', '#06A77D',
                '#4ECDC4', '#2E86AB', '#8338EC', '#A23B72')

p_bad <- ggplot(data, aes(x = group, y = value)) +
  geom_bar(stat = 'identity', aes(fill = group), color = 'black', size = 0.8) +
  geom_errorbar(aes(ymin = value - error, ymax = value + error),
                width = 0.3, size = 0.7) +
  scale_fill_manual(values = colors_bad) +
  labs(y = 'Response (arbitrary units)',
       title = 'BAD: All Colors Equally Saturated\n(No visual priority)') +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'red', size = 12),
    axis.title.x = element_blank(),
    axis.title.y = element_text(face = 'bold'),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = 'none',
    panel.grid.major.y = element_line(color = 'gray90')
  )

# GOOD: Deep colors only for significant
data$color <- ifelse(data$is_significant,
                     ifelse(data$group == 'Group 3', '#E63946', '#2E86AB'),
                     '#D3D3D3')

p_good <- ggplot(data, aes(x = group, y = value)) +
  geom_bar(stat = 'identity', aes(fill = color), color = 'black', size = 0.8) +
  geom_errorbar(aes(ymin = value - error, ymax = value + error),
                width = 0.3, size = 0.7) +
  scale_fill_identity() +
  labs(y = 'Response (arbitrary units)',
       title = 'GOOD: Deep Colors for Significant Only\n(Clear visual hierarchy)') +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'darkgreen', size = 12),
    axis.title.x = element_blank(),
    axis.title.y = element_text(face = 'bold'),
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid.major.y = element_line(color = 'gray90')
  ) +
  # Add significance markers
  geom_text(data = data %>% filter(is_significant),
            aes(label = '***', y = value + error + 2),
            size = 6, fontface = 'bold')

# Combine plots
library(patchwork)
combined <- p_bad | p_good
ggsave('saturation_hierarchy.png', combined, width = 14, height = 5,
       dpi = 300, bg = 'white')
```

---

#### Example 2: Correlation Strength Visualization

**Scenario:** Showing multiple correlations, want to emphasize strongest relationships

**Principle:** **Deep colors = Strong/significant, Light colors = Weak/non-significant**

**Code Example (Python):**

```
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Correlation matrix data
np.random.seed(42)
variables = ['Gene A', 'Gene B', 'Gene C', 'Gene D', 'Gene E']
n_vars = len(variables)

# Generate correlation matrix with some strong correlations
corr_matrix = np.random.rand(n_vars, n_vars) * 0.5
corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Symmetrize
np.fill_diagonal(corr_matrix, 1.0)

# Add some strong correlations
corr_matrix[0, 2] = corr_matrix[2, 0] = 0.85  # Strong positive
corr_matrix[1, 4] = corr_matrix[4, 1] = -0.78  # Strong negative
corr_matrix[3, 4] = corr_matrix[4, 3] = 0.72  # Moderate-strong

# Significance (p-values) - for demonstration
p_values = np.random.rand(n_vars, n_vars) * 0.1
p_values[0, 2] = p_values[2, 0] = 0.001  # Highly significant
p_values[1, 4] = p_values[4, 1] = 0.002  # Highly significant
p_values[3, 4] = p_values[4, 3] = 0.01   # Significant
np.fill_diagonal(p_values, 0)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# BAD: Uniform saturation regardless of significance
im1 = axes[0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
axes[0].set_xticks(range(n_vars))
axes[0].set_yticks(range(n_vars))
axes[0].set_xticklabels(variables, rotation=45, ha='right')
axes[0].set_yticklabels(variables)
axes[0].set_title('BAD: All Correlations Equal Saturation\n(Significant and non-significant look same)',
                  fontsize=12, fontweight='bold', color='red')

# Add correlation values
for i in range(n_vars):
    for j in range(n_vars):
        axes[0].text(j, i, f'{corr_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=9,
                    color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

plt.colorbar(im1, ax=axes[0], label='Correlation coefficient')

# GOOD: Deep colors for significant, desaturated for non-significant
# Create custom colormap with saturation based on significance
corr_colored = np.zeros((n_vars, n_vars, 3))

for i in range(n_vars):
    for j in range(n_vars):
        r = corr_matrix[i, j]
        p = p_values[i, j]

        if p < 0.05:  # Significant: deep colors
            if r > 0:
                # Deep red for significant positive
                saturation = min(abs(r), 1.0)
                corr_colored[i, j] = [0.9, 0.2, 0.2]  # Deep red
            else:
                # Deep blue for significant negative
                saturation = min(abs(r), 1.0)
                corr_colored[i, j] = [0.2, 0.3, 0.7]  # Deep blue
        else:  # Non-significant: light colors
            if r > 0:
                # Light pink for non-significant positive
                corr_colored[i, j] = [1.0, 0.85, 0.85]  # Light pink
            else:
                # Light blue for non-significant negative
                corr_colored[i, j] = [0.85, 0.9, 1.0]  # Light blue

axes[1].imshow(corr_colored, aspect='auto')
axes[1].set_xticks(range(n_vars))
axes[1].set_yticks(range(n_vars))
axes[1].set_xticklabels(variables, rotation=45, ha='right')
axes[1].set_yticklabels(variables)
axes[1].set_title('GOOD: Deep Colors for Significant Only\n(p < 0.05 with deep colors, others desaturated)',
                  fontsize=12, fontweight='bold', color='green')

# Add correlation values with significance markers
for i in range(n_vars):
    for j in range(n_vars):
        sig_marker = '***' if p_values[i, j] < 0.01 else ('*' if p_values[i, j] < 0.05 else '')
        axes[1].text(j, i, f'{corr_matrix[i, j]:.2f}\n{sig_marker}',
                    ha='center', va='center', fontsize=8,
                    color='white' if p_values[i, j] < 0.05 and abs(corr_matrix[i, j]) > 0.5 else 'black',
                    fontweight='bold' if sig_marker else 'normal')

plt.tight_layout()
plt.savefig('correlation_saturation_logic.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R):**

```r
library(ggplot2)
library(reshape2)
library(dplyr)

# Generate correlation data
set.seed(42)
variables <- c('Gene A', 'Gene B', 'Gene C', 'Gene D', 'Gene E')
n_vars <- length(variables)

# Correlation matrix
corr_matrix <- matrix(runif(n_vars^2, -0.5, 0.5), n_vars, n_vars)
corr_matrix <- (corr_matrix + t(corr_matrix)) / 2  # Symmetrize
diag(corr_matrix) <- 1.0

# Add strong correlations
corr_matrix[1, 3] <- corr_matrix[3, 1] <- 0.85   # Strong positive
corr_matrix[2, 5] <- corr_matrix[5, 2] <- -0.78  # Strong negative
corr_matrix[4, 5] <- corr_matrix[5, 4] <- 0.72   # Moderate-strong

# P-values
p_values <- matrix(runif(n_vars^2, 0, 0.1), n_vars, n_vars)
p_values[1, 3] <- p_values[3, 1] <- 0.001
p_values[2, 5] <- p_values[5, 2] <- 0.002
p_values[4, 5] <- p_values[5, 4] <- 0.01
diag(p_values) <- 0

# Convert to long format
rownames(corr_matrix) <- colnames(corr_matrix) <- variables
rownames(p_values) <- colnames(p_values) <- variables

corr_long <- melt(corr_matrix, varnames = c('Var1', 'Var2'), value.name = 'correlation')
p_long <- melt(p_values, varnames = c('Var1', 'Var2'), value.name = 'p_value')

data_combined <- corr_long %>%
  left_join(p_long, by = c('Var1', 'Var2')) %>%
  mutate(
    significant = p_value < 0.05,
    sig_marker = case_when(
      p_value < 0.01 ~ '***',
      p_value < 0.05 ~ '*',
      TRUE ~ ''
    ),
    # Color logic: deep colors for significant, light for non-significant
    color_category = case_when(
      significant & correlation > 0 ~ 'Significant Positive',
      significant & correlation < 0 ~ 'Significant Negative',
      !significant & correlation > 0 ~ 'Non-sig Positive',
      !significant & correlation < 0 ~ 'Non-sig Negative',
      TRUE ~ 'Neutral'
    )
  )

# Define colors
color_mapping <- c(
  'Significant Positive' = '#E63946',    # Deep red
  'Significant Negative' = '#2E86AB',    # Deep blue
  'Non-sig Positive' = '#FFD6D6',        # Light pink
  'Non-sig Negative' = '#D6E5F2',        # Light blue
  'Neutral' = '#FFFFFF'
)

# GOOD plot: Saturation based on significance
p_good <- ggplot(data_combined, aes(x = Var2, y = Var1, fill = color_category)) +
  geom_tile(color = 'white', size = 1) +
  geom_text(aes(label = paste0(sprintf('%.2f', correlation), '\n', sig_marker)),
            size = 3.5, fontface = ifelse(data_combined$significant, 'bold', 'plain')) +
  scale_fill_manual(values = color_mapping, name = 'Correlation Type') +
  labs(title = 'GOOD: Deep Colors for Significant Correlations\n(p < 0.05 with deep colors, others desaturated)',
       x = NULL, y = NULL) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'darkgreen', size = 12),
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    panel.grid = element_blank(),
    legend.position = 'bottom'
  ) +
  coord_fixed()

ggsave('correlation_saturation_logic.png', p_good, width = 8, height = 7,
       dpi = 300, bg = 'white')
```

---

### Guidelines for Saturation Use

**DO use deep, saturated colors for:**
- **Statistically significant results** (p < 0.05)
- **Primary experimental condition** (treatment vs. all controls)
- **Key finding** you want to emphasize
- **Strong effects** (large effect sizes, high correlations)
- **Highlighting outliers** or exceptional cases

**DO use light, desaturated colors for:**
- **Non-significant results** (p ≥ 0.05)
- **Reference/control groups** (when not the focus)
- **Background data** (context, not main finding)
- **Weak effects** (low correlations, small differences)
- **Supporting information** (supplementary analyses)

**Color Saturation Decision Matrix:**

| Data Type | Primary/Significant | Secondary/Non-significant |
|-----------|-------------------|------------------------|
| **Positive correlation** | Deep red (#E63946) | Light pink (#FFD6D6) |
| **Negative correlation** | Deep blue (#2E86AB) | Light blue (#D6E5F2) |
| **Treatment group** | Deep color (#9B59B6) | Light gray (#D3D3D3) |
| **Control group** | Medium gray (#808080) | Light gray (#ECECEC) |
| **Upregulated genes** | Deep red (#C0392B) | Light red (#F5B7B1) |
| **Downregulated genes** | Deep blue (#2874A6) | Light blue (#AED6F1) |

---

### The "3-Color + Saturation" Strategy

**Most effective publication approach: 3 base colors, varied saturation**

**Example: Gene Expression Study**

```
Base Palette (3 colors):
1. Red family: For upregulation
2. Blue family: For downregulation
3. Gray: For no change/control

Saturation levels:
- Deep (S=100%): Significant changes (p < 0.05, |fold-change| > 2)
- Medium (S=60%): Moderate changes (p < 0.05, |fold-change| 1-2)
- Light (S=30%): Non-significant changes (p ≥ 0.05)
```

**Code Example (Python) - Gene Expression Volcano Plot:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulate gene expression data
np.random.seed(42)
n_genes = 1000

# Log2 fold change
log2fc = np.random.randn(n_genes) * 2

# -log10 p-values
neg_log10_p = -np.log10(np.random.uniform(0.0001, 0.5, n_genes))

# Add some significant genes
sig_up_idx = np.random.choice(n_genes, 50, replace=False)
sig_down_idx = np.random.choice(np.setdiff1d(range(n_genes), sig_up_idx), 50, replace=False)

log2fc[sig_up_idx] = np.random.uniform(2, 4, 50)
log2fc[sig_down_idx] = np.random.uniform(-4, -2, 50)
neg_log10_p[sig_up_idx] = np.random.uniform(3, 6, 50)
neg_log10_p[sig_down_idx] = np.random.uniform(3, 6, 50)

# Categorize genes by significance and magnitude
def categorize_gene(fc, p_val):
    significant = p_val > -np.log10(0.05)  # p < 0.05
    strong_change = abs(fc) > 2

    if significant and strong_change:
        if fc > 0:
            return 'sig_up_strong', '#C0392B', 100  # Deep red
        else:
            return 'sig_down_strong', '#2874A6', 100  # Deep blue
    elif significant:
        if fc > 0:
            return 'sig_up_mod', '#E74C3C', 60  # Medium red
        else:
            return 'sig_down_mod', '#3498DB', 60  # Medium blue
    else:
        return 'not_sig', '#D5D8DC', 30  # Light gray

categories = [categorize_gene(fc, p) for fc, p in zip(log2fc, neg_log10_p)]
cat_names, colors, sizes = zip(*categories)

fig, ax = plt.subplots(figsize=(8, 6))

# Plot with saturation hierarchy
ax.scatter(log2fc, neg_log10_p, c=colors, s=15, alpha=0.6, edgecolors='none')

# Add thresholds
ax.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=1,
           label='p = 0.05', alpha=0.5)
ax.axvline(-2, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(2, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Labels and styling
ax.set_xlabel('Log₂ Fold Change', fontsize=12, fontweight='bold')
ax.set_ylabel('-Log₁₀ P-value', fontsize=12, fontweight='bold')
ax.set_title('Volcano Plot: Saturation Shows Significance & Magnitude',
             fontsize=13, fontweight='bold')

# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#C0392B', label='Sig. upregulated (|FC| > 2)'),
    Patch(facecolor='#E74C3C', label='Sig. upregulated (|FC| < 2)'),
    Patch(facecolor='#2874A6', label='Sig. downregulated (|FC| > 2)'),
    Patch(facecolor='#3498DB', label='Sig. downregulated (|FC| < 2)'),
    Patch(facecolor='#D5D8DC', label='Not significant')
]
ax.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=9)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('volcano_saturation_logic.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R):**

```r
library(ggplot2)
library(dplyr)

# Simulate gene expression data
set.seed(42)
n_genes <- 1000

data <- data.frame(
  log2fc = rnorm(n_genes, 0, 2),
  neg_log10_p = -log10(runif(n_genes, 0.0001, 0.5))
)

# Add significant genes
sig_up_idx <- sample(n_genes, 50)
sig_down_idx <- sample(setdiff(1:n_genes, sig_up_idx), 50)

data$log2fc[sig_up_idx] <- runif(50, 2, 4)
data$log2fc[sig_down_idx] <- runif(50, -4, -2)
data$neg_log10_p[sig_up_idx] <- runif(50, 3, 6)
data$neg_log10_p[sig_down_idx] <- runif(50, 3, 6)

# Categorize
data <- data %>%
  mutate(
    significant = neg_log10_p > -log10(0.05),
    strong_change = abs(log2fc) > 2,
    category = case_when(
      significant & strong_change & log2fc > 0 ~ 'Sig. Up (strong)',
      significant & strong_change & log2fc < 0 ~ 'Sig. Down (strong)',
      significant & log2fc > 0 ~ 'Sig. Up (moderate)',
      significant & log2fc < 0 ~ 'Sig. Down (moderate)',
      TRUE ~ 'Not significant'
    ),
    color = case_when(
      category == 'Sig. Up (strong)' ~ '#C0392B',      # Deep red
      category == 'Sig. Up (moderate)' ~ '#E74C3C',    # Medium red
      category == 'Sig. Down (strong)' ~ '#2874A6',    # Deep blue
      category == 'Sig. Down (moderate)' ~ '#3498DB',  # Medium blue
      TRUE ~ '#D5D8DC'                                  # Light gray
    )
  )

# Set factor levels for legend order
data$category <- factor(data$category, levels = c(
  'Sig. Up (strong)', 'Sig. Up (moderate)',
  'Sig. Down (strong)', 'Sig. Down (moderate)',
  'Not significant'
))

# Plot
p <- ggplot(data, aes(x = log2fc, y = neg_log10_p, color = category)) +
  geom_point(size = 1.5, alpha = 0.6) +

  # Thresholds
  geom_hline(yintercept = -log10(0.05), linetype = 'dashed',
             color = 'black', size = 0.7, alpha = 0.5) +
  geom_vline(xintercept = c(-2, 2), linetype = 'dashed',
             color = 'gray50', size = 0.7, alpha = 0.5) +

  # Colors
  scale_color_manual(values = c(
    'Sig. Up (strong)' = '#C0392B',
    'Sig. Up (moderate)' = '#E74C3C',
    'Sig. Down (strong)' = '#2874A6',
    'Sig. Down (moderate)' = '#3498DB',
    'Not significant' = '#D5D8DC'
  )) +

  # Labels
  labs(
    x = 'Log₂ Fold Change',
    y = '-Log₁₀ P-value',
    title = 'Volcano Plot: Saturation Shows Significance & Magnitude',
    color = 'Gene Category'
  ) +

  # Theme
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', size = 13),
    axis.title = element_text(face = 'bold'),
    legend.position = c(0.15, 0.85),
    legend.background = element_rect(fill = 'white', color = 'black', size = 0.5),
    panel.grid.major = element_line(color = 'gray90', size = 0.3)
  )

ggsave('volcano_saturation_logic.png', p, width = 8, height = 6,
       dpi = 300, bg = 'white')
```

---

### Publication Checklist: Color Saturation

**Before finalizing figures, verify:**

- [ ] **Deep colors reserved for significant results** (p < 0.05 or key findings)
- [ ] **Light colors for non-significant** or background data
- [ ] **Saturation hierarchy matches data importance**
- [ ] **Maximum 3 base colors** (not counting saturation variants)
- [ ] **Logical semantic meaning** (e.g., red=up, blue=down; maintained across all figures)
- [ ] **Works in grayscale** (test by converting)
- [ ] **Colorblind accessible** (test with simulator)

-----


## 2.5 Continuous vs. Discrete Color Scales

### Understanding the Distinction

The choice between continuous and discrete color scales is **not aesthetic—it's determined by your data type**. Using the wrong scale type can fundamentally misrepresent your data and mislead readers.

---

### Continuous Color Scales (Sequential/Diverging Gradients)

**Use for: Truly continuous numerical data**

**Data types:**
- Temperature measurements (15.3°C, 15.4°C, 15.5°C...)
- Gene expression levels (FPKM: 0.1, 2.3, 45.8, 123.4...)
- Concentration gradients (0.01 μM to 100 μM)
- Correlation coefficients (-1.0 to +1.0)
- Probability values (0.0 to 1.0)
- Any measurement on a continuous scale

**Key characteristic:** Values can take **any point along a spectrum**, including intermediate values.

**Visual representation:** Smooth gradient with no discrete boundaries

---

### When Continuous Scales Make Sense

#### Example 1: Correlation Matrices

**Logical Application:**
```
Correlation coefficient r ranges from -1.0 to +1.0
- Any value in between is possible: -0.87, -0.23, 0.0, 0.45, 0.91
- Smooth gradient shows strength naturally
- Diverging palette: Blue (negative) ← White (zero) → Red (positive)
```

**Why it's logical:**
- **Proportional interpretation**: Darker red = stronger positive correlation
- **Continuous nature**: Correlation is genuinely continuous
- **Intuitive reading**: Color intensity matches correlation strength

**Code Example (Python):**

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Generate correlation matrix
np.random.seed(42)
n_vars = 8
variables = [f'Var{i+1}' for i in range(n_vars)]

# Create realistic correlation structure
base_corr = np.random.randn(n_vars, n_vars) * 0.3
corr_matrix = base_corr @ base_corr.T
corr_matrix = corr_matrix / np.outer(np.sqrt(np.diag(corr_matrix)),
                                      np.sqrt(np.diag(corr_matrix)))
np.fill_diagonal(corr_matrix, 1.0)

# Continuous scale with logical meaning
fig, ax = plt.subplots(figsize=(8, 7))

# Use diverging colormap: negative=blue, zero=white, positive=red
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

# Styling
ax.set_xticks(range(n_vars))
ax.set_yticks(range(n_vars))
ax.set_xticklabels(variables, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(variables, fontsize=10)

# Add correlation values
for i in range(n_vars):
    for j in range(n_vars):
        text_color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
        ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                ha='center', va='center', fontsize=9,
                color=text_color, fontweight='bold' if abs(corr_matrix[i, j]) > 0.7 else 'normal')

# Colorbar with clear interpretation
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Correlation Coefficient (r)', fontsize=11, fontweight='bold')
cbar.ax.axhline(0, color='black', linewidth=2, linestyle='-')  # Highlight zero

ax.set_title('CONTINUOUS SCALE: Correlation Matrix\n(Smooth gradient shows continuous relationship)',
             fontsize=12, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('continuous_correlation.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R):**

```r
library(ggplot2)
library(reshape2)
library(dplyr)

# Generate correlation matrix
set.seed(42)
n_vars <- 8
variables <- paste0('Var', 1:n_vars)

base_corr <- matrix(rnorm(n_vars^2, 0, 0.3), n_vars, n_vars)
corr_matrix <- base_corr %*% t(base_corr)
corr_matrix <- corr_matrix / outer(sqrt(diag(corr_matrix)), sqrt(diag(corr_matrix)))
diag(corr_matrix) <- 1.0

# Convert to long format
rownames(corr_matrix) <- colnames(corr_matrix) <- variables
corr_long <- melt(corr_matrix, varnames = c('Var1', 'Var2'), value.name = 'correlation')

# Plot with continuous scale
p <- ggplot(corr_long, aes(x = Var2, y = Var1, fill = correlation)) +
  geom_tile(color = 'white', size = 0.5) +

  # Continuous diverging scale
  scale_fill_gradient2(
    low = '#2166AC',      # Blue (negative)
    mid = 'white',        # White (zero)
    high = '#B2182B',     # Red (positive)
    midpoint = 0,
    limits = c(-1, 1),
    name = 'Correlation\nCoefficient (r)'
  ) +

  # Add text labels
  geom_text(aes(label = sprintf('%.2f', correlation)),
            size = 3,
            color = ifelse(abs(corr_long$correlation) > 0.5, 'white', 'black'),
            fontface = ifelse(abs(corr_long$correlation) > 0.7, 'bold', 'plain')) +

  labs(title = 'CONTINUOUS SCALE: Correlation Matrix\n(Smooth gradient shows continuous relationship)') +

  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', size = 12),
    axis.title = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    panel.grid = element_blank(),
    legend.position = 'right'
  ) +
  coord_fixed()

ggsave('continuous_correlation.png', p, width = 9, height = 7,
       dpi = 300, bg = 'white')
```

---

#### Example 2: Gene Expression Heatmaps

**Logical Application:**
```
Expression level (FPKM/TPM) is genuinely continuous
- Values: 0.1, 1.3, 5.7, 23.4, 156.8...
- Often log-transformed for visualization (log2 or log10)
- Sequential gradient shows intensity naturally
```

**Why continuous scale is appropriate:**
- Expression can take any positive value
- Smooth transitions reflect biological reality
- Color intensity proportional to expression level

**Important: Add significance layer with saturation (from 2.3)**

**Code Example (Python):**

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Simulate gene expression data
np.random.seed(42)
n_genes = 30
n_samples = 8

# Log2-transformed expression values (continuous)
expression = np.random.lognormal(mean=2, sigma=1.5, size=(n_genes, n_samples))
log2_expr = np.log2(expression + 1)

# Simulate p-values for differential expression
p_values = np.random.uniform(0, 0.15, n_genes)
significant = p_values < 0.05

# Gene and sample names
genes = [f'Gene{i+1}' for i in range(n_genes)]
samples = [f'S{i+1}' for i in range(n_samples)]

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Panel A: Standard continuous heatmap (all genes equal visual weight)
ax1 = axes[0]
im1 = ax1.imshow(log2_expr, cmap='YlOrRd', aspect='auto')
ax1.set_xticks(range(n_samples))
ax1.set_yticks(range(n_genes))
ax1.set_xticklabels(samples, fontsize=8)
ax1.set_yticklabels(genes, fontsize=7)
ax1.set_xlabel('Samples', fontsize=10, fontweight='bold')
ax1.set_ylabel('Genes', fontsize=10, fontweight='bold')
ax1.set_title('CONTINUOUS: Standard Heatmap\n(All genes equal visual weight)',
              fontsize=11, fontweight='bold')

cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046)
cbar1.set_label('Log₂ Expression', fontsize=10, fontweight='bold')

# Panel B: Enhanced with significance (deep colors for significant only)
ax2 = axes[1]

# Create custom colormap: deep colors for significant, light for non-significant
expression_colored = np.zeros((n_genes, n_samples, 3))

# Normalize expression for color mapping
expr_norm = (log2_expr - log2_expr.min()) / (log2_expr.max() - log2_expr.min())

for i in range(n_genes):
    for j in range(n_samples):
        intensity = expr_norm[i, j]

        if significant[i]:  # Significant: deep red gradient
            # Deep red scale
            expression_colored[i, j] = [
                0.2 + intensity * 0.6,  # R: 0.2 to 0.8
                0.1 * (1 - intensity),  # G: decrease with intensity
                0.1 * (1 - intensity)   # B: decrease with intensity
            ]
        else:  # Non-significant: light pink gradient
            # Light pink scale
            expression_colored[i, j] = [
                0.95,                    # R: always high (pink)
                0.7 + intensity * 0.25,  # G: light
                0.7 + intensity * 0.25   # B: light
            ]

ax2.imshow(expression_colored, aspect='auto')
ax2.set_xticks(range(n_samples))
ax2.set_yticks(range(n_genes))
ax2.set_xticklabels(samples, fontsize=8)
ax2.set_yticklabels(genes, fontsize=7)
ax2.set_xlabel('Samples', fontsize=10, fontweight='bold')
ax2.set_ylabel('Genes', fontsize=10, fontweight='bold')
ax2.set_title('CONTINUOUS + SIGNIFICANCE:\n(Deep colors = p<0.05, Light = non-sig)',
              fontsize=11, fontweight='bold', color='green')

# Add significance markers
for i in range(n_genes):
    if significant[i]:
        ax2.text(-1.5, i, '***', fontsize=10, fontweight='bold',
                ha='center', va='center', color='red')

plt.tight_layout()
plt.savefig('continuous_expression_heatmap.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R):**

```r
library(ggplot2)
library(reshape2)
library(dplyr)
library(patchwork)

# Simulate gene expression data
set.seed(42)
n_genes <- 30
n_samples <- 8

expression <- matrix(rlnorm(n_genes * n_samples, meanlog=2, sdlog=1.5),
                     nrow=n_genes, ncol=n_samples)
log2_expr <- log2(expression + 1)

# P-values
p_values <- runif(n_genes, 0, 0.15)
significant <- p_values < 0.05

# Names
genes <- paste0('Gene', 1:n_genes)
samples <- paste0('S', 1:n_samples)

# Convert to long format
colnames(log2_expr) <- samples
rownames(log2_expr) <- genes

expr_long <- log2_expr %>%
  as.data.frame() %>%
  tibble::rownames_to_column('Gene') %>%
  melt(id.vars = 'Gene', variable.name = 'Sample', value.name = 'Expression') %>%
  mutate(Gene = factor(Gene, levels = genes))

# Add significance
expr_long <- expr_long %>%
  left_join(
    data.frame(Gene = genes, p_value = p_values, significant = significant),
    by = 'Gene'
  )

# Panel A: Standard continuous heatmap
p1 <- ggplot(expr_long, aes(x = Sample, y = Gene, fill = Expression)) +
  geom_tile(color = 'white', size = 0.5) +
  scale_fill_gradientn(
    colors = c('#FFFFCC', '#FFEDA0', '#FED976', '#FEB24C', '#FD8D3C', '#FC4E2A', '#E31A1C', '#B10026'),
    name = 'Log₂\nExpression'
  ) +
  labs(title = 'CONTINUOUS: Standard Heatmap\n(All genes equal visual weight)') +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', size = 11),
    axis.title = element_text(face = 'bold', size = 10),
    axis.text.y = element_text(size = 7),
    axis.text.x = element_text(size = 8),
    panel.grid = element_blank()
  )

# Panel B: With significance (deep vs light colors)
expr_long <- expr_long %>%
  mutate(
    # Scale expression 0-1
    expr_norm = (Expression - min(Expression)) / (max(Expression) - min(Expression)),
    # Color based on significance
    color = case_when(
      significant ~ rgb(0.2 + expr_norm * 0.6, 0.1 * (1 - expr_norm), 0.1 * (1 - expr_norm)),
      TRUE ~ rgb(0.95, 0.7 + expr_norm * 0.25, 0.7 + expr_norm * 0.25)
    )
  )

p2 <- ggplot(expr_long, aes(x = Sample, y = Gene)) +
  geom_tile(aes(fill = color), color = 'white', size = 0.5) +
  scale_fill_identity() +
  labs(title = 'CONTINUOUS + SIGNIFICANCE\n(Deep colors = p<0.05, Light = non-sig)') +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', size = 11, color = 'darkgreen'),
    axis.title = element_text(face = 'bold', size = 10),
    axis.text.y = element_text(size = 7),
    axis.text.x = element_text(size = 8),
    panel.grid = element_blank()
  )

# Add significance markers
sig_genes <- expr_long %>%
  filter(significant) %>%
  distinct(Gene)

p2 <- p2 +
  geom_text(data = sig_genes, aes(x = 0, y = Gene, label = '***'),
            size = 3.5, fontface = 'bold', color = 'red', hjust = 1)

# Combine
combined <- p1 | p2
ggsave('continuous_expression_heatmap.png', combined, width = 14, height = 8,
       dpi = 300, bg = 'white')
```

---

### Discrete Color Scales (Categorical Palettes)

**Use for: Truly categorical data with distinct, unordered groups**

**Data types:**
- Species names (Homo sapiens, Mus musculus, Drosophila...)
- Treatment groups (Control, Drug A, Drug B, Placebo)
- Cell types (Neurons, Astrocytes, Microglia, Endothelial)
- Geographic regions (North, South, East, West)
- Genotypes (Wild-type, Heterozygous, Homozygous mutant)

**Key characteristic:** Values are **distinct categories** with no inherent ordering or intermediate states.

**Visual representation:** Distinct, non-gradient colors

---

### When Discrete Scales Make Sense

#### Example 1: Cell Type Identification

**Logical Application:**
```
Cell types are categorical:
- A cell is a neuron OR an astrocyte, not "half-neuron, half-astrocyte"
- No intermediate states or smooth transitions
- Each type needs a distinct, recognizable color
```

**Why discrete scale is logical:**
- Categories are mutually exclusive
- No quantitative relationship between categories
- Distinct colors prevent false impression of ordering

**Code Example (Python):**

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulate spatial cell type data (e.g., from microscopy or spatial transcriptomics)
np.random.seed(42)
n_cells = 500

# Spatial coordinates
x = np.random.rand(n_cells) * 100
y = np.random.rand(n_cells) * 100

# Cell types (categorical - mutually exclusive)
cell_types = np.random.choice(['Neuron', 'Astrocyte', 'Microglia', 'Oligodendrocyte', 'Endothelial'],
                              size=n_cells, p=[0.35, 0.25, 0.15, 0.15, 0.10])

# DISCRETE colors for categorical data (Okabe-Ito colorblind-safe palette)
color_map = {
    'Neuron': '#E69F00',         # Orange
    'Astrocyte': '#56B4E9',      # Sky blue
    'Microglia': '#009E73',      # Green
    'Oligodendrocyte': '#F0E442', # Yellow
    'Endothelial': '#CC79A7'     # Pink
}

colors = [color_map[ct] for ct in cell_types]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# BAD: Continuous colormap for categorical data
ax1 = axes[0]
cell_type_numeric = np.array([list(color_map.keys()).index(ct) for ct in cell_types])
scatter1 = ax1.scatter(x, y, c=cell_type_numeric, cmap='viridis', s=50, alpha=0.7)
ax1.set_xlabel('X Position (μm)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Y Position (μm)', fontsize=11, fontweight='bold')
ax1.set_title('BAD: Continuous Scale for Categories\n(Implies false ordering/gradient)',
              fontsize=12, fontweight='bold', color='red')
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('Cell Type (?)', fontsize=10)
ax1.set_aspect('equal')

# GOOD: Discrete colors for categorical data
ax2 = axes[1]
for cell_type, color in color_map.items():
    mask = cell_types == cell_type
    ax2.scatter(x[mask], y[mask], c=color, s=50, alpha=0.7,
               label=cell_type, edgecolors='black', linewidths=0.5)

ax2.set_xlabel('X Position (μm)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Y Position (μm)', fontsize=11, fontweight='bold')
ax2.set_title('GOOD: Discrete Colors for Categories\n(Each type distinct and recognizable)',
              fontsize=12, fontweight='bold', color='green')
ax2.legend(loc='upper right', frameon=True, fontsize=9, title='Cell Type',
          title_fontsize=10, edgecolor='black')
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('discrete_cell_types.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R):**

```r
library(ggplot2)
library(dplyr)
library(patchwork)

# Simulate spatial cell type data
set.seed(42)
n_cells <- 500

data <- data.frame(
  x = runif(n_cells, 0, 100),
  y = runif(n_cells, 0, 100),
  cell_type = sample(c('Neuron', 'Astrocyte', 'Microglia', 'Oligodendrocyte', 'Endothelial'),
                     size = n_cells, replace = TRUE,
                     prob = c(0.35, 0.25, 0.15, 0.15, 0.10))
)

# Discrete colors (Okabe-Ito palette)
color_map <- c(
  'Neuron' = '#E69F00',
  'Astrocyte' = '#56B4E9',
  'Microglia' = '#009E73',
  'Oligodendrocyte' = '#F0E442',
  'Endothelial' = '#CC79A7'
)

# BAD: Treating categorical as continuous
data$cell_type_numeric <- as.numeric(factor(data$cell_type))

p1 <- ggplot(data, aes(x = x, y = y, color = cell_type_numeric)) +
  geom_point(size = 2.5, alpha = 0.7) +
  scale_color_viridis_c(name = 'Cell Type (?)') +
  labs(x = 'X Position (μm)', y = 'Y Position (μm)',
       title = 'BAD: Continuous Scale for Categories\n(Implies false ordering/gradient)') +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'red', size = 12),
    axis.title = element_text(face = 'bold'),
    aspect.ratio = 1
  )

# GOOD: Discrete colors
p2 <- ggplot(data, aes(x = x, y = y, color = cell_type)) +
  geom_point(size = 2.5, alpha = 0.7) +
  scale_color_manual(values = color_map, name = 'Cell Type') +
  labs(x = 'X Position (μm)', y = 'Y Position (μm)',
       title = 'GOOD: Discrete Colors for Categories\n(Each type distinct and recognizable)') +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'darkgreen', size = 12),
    axis.title = element_text(face = 'bold'),
    legend.position = c(0.85, 0.85),
    legend.background = element_rect(fill = 'white', color = 'black', size = 0.5),
    aspect.ratio = 1
  )

# Combine
combined <- p1 | p2
ggsave('discrete_cell_types.png', combined, width = 14, height = 6,
       dpi = 300, bg = 'white')
```

---

### Common Mistakes: Mixing Continuous and Discrete

**Mistake 1: Using continuous scale for categories**

```
❌ BAD EXAMPLE:
Treatment groups (A, B, C, D) colored with rainbow gradient
→ Implies ordering: "B is between A and C"
→ Misleading when no such relationship exists

✓ FIX:
Use distinct colors from qualitative palette (Set2, Dark2, Okabe-Ito)
```

**Mistake 2: Using discrete colors for continuous data**

```
❌ BAD EXAMPLE:
Temperature data (15°C, 16°C, 17°C...) with 3 discrete colors (blue, yellow, red)
→ Creates false boundaries: "Below 20°=blue, 20-25°=yellow, above 25°=red"
→ Loses precision of continuous measurement

✓ FIX:
Use continuous gradient (sequential colormap: light blue → dark blue)
```

---

### Decision Matrix: Continuous vs. Discrete

| Question | Answer | Scale Type | Example |
|----------|--------|------------|---------|
| Can data take intermediate values? | Yes | **Continuous** | Temperature: 25.7°C |
| Are categories mutually exclusive? | Yes | **Discrete** | Species: Human OR Mouse |
| Is there natural ordering? | Yes (quantitative) | **Continuous** | Concentration: 0-100 μM |
| Is there natural ordering? | Yes (categorical) | **Discrete Sequential** | Disease: Mild/Moderate/Severe |
| Is zero/midpoint meaningful? | Yes | **Continuous Diverging** | Fold change: -2x to +2x |
| Do you need exact value reading? | Yes | **Continuous** | Gene expression levels |
| Do you need category identification? | Yes | **Discrete** | Treatment groups |

---

### Exercise 2.5.1: Scale Type Classification

**Objective:** Practice identifying appropriate scale types and justifying choices

**Instructions:**

For each dataset, determine:
1. Is data continuous or categorical?
2. If continuous: sequential or diverging?
3. If categorical: qualitative or ordinal?
4. What colormap would you use?
5. What would happen if you used the wrong scale type?

**Datasets:**

**A. Soil pH measurements** across a field
- Values: 4.2, 5.8, 6.1, 6.9, 7.3, 8.1
- Range: 0-14

**B. Voting preference** in an election
- Categories: Candidate A, Candidate B, Candidate C, Undecided

**C. Protein abundance** change after treatment
- Values: -3.2x, -1.5x, +0.2x, +2.1x, +4.8x fold-change
- Reference: 1x (no change)

**D. Animal species** observed in a habitat
- Categories: Deer, Rabbit, Fox, Squirrel, Bird

**E. Pain scale** reported by patients
- Categories: None, Mild, Moderate, Severe, Extreme

**Format your answer like this:**

```
Dataset A: Soil pH
- Data type: Continuous numerical
- Scale type: Sequential continuous
- Colormap: Single-hue gradient (e.g., Yellow-Green-Blue for pH)
- Justification: pH is genuinely continuous; any intermediate value possible
- Wrong scale consequences: If discrete → false boundaries (e.g., "acidic" vs "neutral")
  where transition is actually gradual

Dataset E: Pain scale
- Data type: Ordinal categorical
- Scale type: Discrete sequential (ordered categories)
- Colormap: Light → Dark single hue (e.g., light yellow → deep red)
- Justification: Clear ordering but categories are discrete (no "2.5" pain level)
- Wrong scale consequences: If continuous gradient → implies precision that doesn't exist
```

---

**End of Section 2.5**

**Key Takeaways:**
- **Continuous scales** for data that can take any value in a range
- **Discrete scales** for mutually exclusive categories
- **Never mix types**: Using continuous for categorical (or vice versa) misrepresents data
- **Significance matters**: Layer saturation hierarchy onto continuous scales
- **Semantic logic**: Color choices must match data meaning (correlation: blue-white-red, categories: distinct hues)



## 2.6 Color in Context: Scientific Taste and Discipline Conventions

### The Unwritten Rules of Scientific Color Use

While color theory provides the technical foundation, successful scientific figures must also navigate **field-specific conventions** and **aesthetic expectations** that signal professionalism and credibility. This isn't about arbitrary preferences—these conventions have evolved to facilitate rapid interpretation within specialized communities.

---

### What is "Scientific Taste" in Color?

**Scientific taste** refers to the collectively accepted color practices that:

1. **Signal competence**: Proper use shows you understand your field
2. **Facilitate communication**: Readers interpret colors according to learned conventions
3. **Convey rigor**: Conservative, purposeful color use suggests careful analysis
4. **Enable comparison**: Consistent schemes across literature allow mental cross-referencing

**Not scientific taste:**
- Personal aesthetic preferences
- Trendy design styles
- Decorative color choices
- Maximum visual impact regardless of clarity

---

### Core Principles of Scientific Color Taste

#### Principle 1: Conservation (Restraint Over Abundance)

**The Rule:** Use the **minimum number of colors** necessary to convey information.

**Why it signals quality:**
```
Too many colors suggests:
- Lack of focus (unclear message)
- Inexperience (not understanding conventions)
- Trying to impress rather than inform

Restrained color use suggests:
- Clear thinking (one message, one figure)
- Confidence (don't need flashiness)
- Respect for reader's cognitive load
```

**Quantitative Guidelines:**

| Figure Type | Recommended Max Colors | Rationale |
|-------------|----------------------|-----------|
| **Single plot** | 3-4 distinct hues | Beyond this, legend becomes confusing |
| **Multi-panel (related)** | 3-4 across all panels | Consistency aids interpretation |
| **Multi-panel (diverse)** | 2-3 per panel | Each panel can have its own scheme if data types differ |
| **Heatmap** | 1 gradient (sequential or diverging) | Continuous data needs smooth scale |

**Code Example (Python):**

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
time = np.arange(0, 24, 2)

# Simulate experimental data
n_groups = 3  # Restrained: only 3 groups shown

data = {
    'Control': 100 + np.cumsum(np.random.randn(len(time)) * 3),
    'Treatment A': 100 + np.cumsum(np.random.randn(len(time)) * 3 + 0.8),
    'Treatment B': 100 + np.cumsum(np.random.randn(len(time)) * 3 + 1.5)
}

# GOOD: Restrained palette (3 colors max, grayscale for non-essential)
COLORS_GOOD = {
    'Control': '#7F8C8D',      # Gray (reference)
    'Treatment A': '#3498DB',  # Blue
    'Treatment B': '#E74C3C'   # Red
}

# BAD: Excessive colors (trying to show too much at once)
data_bad = {**data, **{
    'Treatment C': 100 + np.cumsum(np.random.randn(len(time)) * 3 + 0.5),
    'Treatment D': 100 + np.cumsum(np.random.randn(len(time)) * 3 + 0.3),
    'Treatment E': 100 + np.cumsum(np.random.randn(len(time)) * 3 + 1.0),
    'Treatment F': 100 + np.cumsum(np.random.randn(len(time)) * 3 + 0.7)
}}

COLORS_BAD = {
    'Control': '#7F8C8D',
    'Treatment A': '#3498DB',
    'Treatment B': '#E74C3C',
    'Treatment C': '#F39C12',
    'Treatment D': '#8E44AD',
    'Treatment E': '#16A085',
    'Treatment F': '#D35400'
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# BAD: Too many groups/colors
ax1 = axes[0]
for group, values in data_bad.items():
    ax1.plot(time, values, marker='o', linewidth=2, markersize=5,
            color=COLORS_BAD[group], label=group)

ax1.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Response (%)', fontsize=11, fontweight='bold')
ax1.set_title('BAD: Too Many Colors\n(Cognitive overload, unclear message)',
             fontsize=12, fontweight='bold', color='red')
ax1.legend(loc='upper left', fontsize=8, ncol=2, frameon=True)
ax1.grid(alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# GOOD: Restrained (3 colors)
ax2 = axes[1]
for group, values in data.items():
    ax2.plot(time, values, marker='o', linewidth=2.5, markersize=6,
            color=COLORS_GOOD[group], label=group)

ax2.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Response (%)', fontsize=11, fontweight='bold')
ax2.set_title('GOOD: Restrained Color Use\n(Clear, focused message)',
             fontsize=12, fontweight='bold', color='green')
ax2.legend(loc='upper left', fontsize=9, frameon=True)
ax2.grid(alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('color_restraint_principle.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R):**

```r
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)

set.seed(42)
time <- seq(0, 24, by = 2)

# GOOD: 3 groups only
data_good <- data.frame(
  time = rep(time, 3),
  response = c(
    100 + cumsum(rnorm(length(time), 0, 3)),
    100 + cumsum(rnorm(length(time), 0.8, 3)),
    100 + cumsum(rnorm(length(time), 1.5, 3))
  ),
  group = rep(c('Control', 'Treatment A', 'Treatment B'), each = length(time))
)

COLORS_GOOD <- c(
  'Control' = '#7F8C8D',
  'Treatment A' = '#3498DB',
  'Treatment B' = '#E74C3C'
)

# BAD: 7 groups (too many)
data_bad <- data.frame(
  time = rep(time, 7),
  response = c(
    100 + cumsum(rnorm(length(time), 0, 3)),
    100 + cumsum(rnorm(length(time), 0.8, 3)),
    100 + cumsum(rnorm(length(time), 1.5, 3)),
    100 + cumsum(rnorm(length(time), 0.5, 3)),
    100 + cumsum(rnorm(length(time), 0.3, 3)),
    100 + cumsum(rnorm(length(time), 1.0, 3)),
    100 + cumsum(rnorm(length(time), 0.7, 3))
  ),
  group = rep(c('Control', 'Treatment A', 'Treatment B', 'Treatment C',
                'Treatment D', 'Treatment E', 'Treatment F'), each = length(time))
)

COLORS_BAD <- c(
  'Control' = '#7F8C8D',
  'Treatment A' = '#3498DB',
  'Treatment B' = '#E74C3C',
  'Treatment C' = '#F39C12',
  'Treatment D' = '#8E44AD',
  'Treatment E' = '#16A085',
  'Treatment F' = '#D35400'
)

# BAD plot
p1 <- ggplot(data_bad, aes(x = time, y = response, color = group)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  scale_color_manual(values = COLORS_BAD) +
  labs(x = 'Time (hours)', y = 'Response (%)',
       title = 'BAD: Too Many Colors\n(Cognitive overload, unclear message)',
       color = NULL) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'red', size = 12),
    axis.title = element_text(face = 'bold'),
    legend.position = c(0.15, 0.85),
    legend.background = element_rect(fill = 'white', color = 'black', size = 0.5),
    panel.grid.major = element_line(color = 'gray90', size = 0.3)
  )

# GOOD plot
p2 <- ggplot(data_good, aes(x = time, y = response, color = group)) +
  geom_line(size = 1.5) +
  geom_point(size = 4) +
  scale_color_manual(values = COLORS_GOOD) +
  labs(x = 'Time (hours)', y = 'Response (%)',
       title = 'GOOD: Restrained Color Use\n(Clear, focused message)',
       color = NULL) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'darkgreen', size = 12),
    axis.title = element_text(face = 'bold'),
    legend.position = c(0.15, 0.85),
    legend.background = element_rect(fill = 'white', color = 'black', size = 0.5),
    panel.grid.major = element_line(color = 'gray90', size = 0.3)
  )

combined <- p1 | p2
ggsave('color_restraint_principle.png', combined, width = 14, height = 5,
       dpi = 300, bg = 'white')
```

---

#### Principle 2: Semantic Consistency (Color = Meaning)

**The Rule:** Establish color-concept associations and maintain them **throughout your entire manuscript**.

**Why it matters:**

```
Reader mental model:
Figure 1: "Blue = Treatment A"
[Reads text, looks at Figure 2]
Figure 2: "Blue = Treatment A" (same)
→ Instant recognition, no re-learning

Inconsistent approach:
Figure 1: "Blue = Treatment A"
Figure 2: "Red = Treatment A" (changed!)
→ Confusion, constant cross-referencing, error-prone
```

**Semantic Logic Examples:**

**Temperature-related:**
```
✓ Blue = Cold/Low
✓ Red = Hot/High
✗ Never reverse this (culturally ingrained)
```

**Biological expression:**
```
✓ Red = Upregulated/Increased
✓ Blue/Green = Downregulated/Decreased
✓ Gray = No change/Control
```

**Clinical outcomes:**
```
✓ Green = Healthy/Improved
✓ Red = Disease/Worsened
✓ Gray = Baseline
```

**Statistical significance:**
```
✓ Deep/saturated = Significant (p<0.05)
✓ Light/desaturated = Non-significant
✓ Consistent across all figures
```

**Code Example (Python) - Manuscript-Wide Consistency:**

```python
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# DEFINE COLORS ONCE - USE THROUGHOUT MANUSCRIPT
# ============================================

# Global color scheme for entire paper
MANUSCRIPT_COLORS = {
    # Treatment groups (consistent across ALL figures)
    'Control': '#95A5A6',        # Gray
    'Drug_A': '#3498DB',         # Blue
    'Drug_B': '#E74C3C',         # Red

    # Expression levels
    'Upregulated': '#C0392B',    # Deep red
    'Downregulated': '#2874A6',  # Deep blue
    'Unchanged': '#D5D8DC',      # Light gray

    # Significance
    'Significant': '#E67E22',    # Orange (highlights)
    'NonSignificant': '#ECF0F1'  # Very light gray
}

# ============================================
# FIGURE 1: Time Course
# ============================================

np.random.seed(42)
time = np.arange(0, 24, 2)

data_fig1 = {
    'Control': 100 + np.cumsum(np.random.randn(len(time)) * 2),
    'Drug_A': 100 + np.cumsum(np.random.randn(len(time)) * 2 + 0.5),
    'Drug_B': 100 + np.cumsum(np.random.randn(len(time)) * 2 + 1.0)
}

fig1, ax1 = plt.subplots(figsize=(7, 5))

for group, values in data_fig1.items():
    ax1.plot(time, values, marker='o', linewidth=2.5, markersize=6,
            color=MANUSCRIPT_COLORS[group], label=group)

ax1.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Cell Viability (%)', fontsize=11, fontweight='bold')
ax1.set_title('Figure 1: Time Course Analysis\n(Colors: Control=Gray, Drug A=Blue, Drug B=Red)',
             fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10, frameon=True)
ax1.grid(alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('manuscript_fig1_consistent_colors.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()

# ============================================
# FIGURE 2: Endpoint Comparison (SAME COLORS)
# ============================================

fig2, ax2 = plt.subplots(figsize=(6, 5))

groups = ['Control', 'Drug_A', 'Drug_B']
endpoint_values = [85, 92, 78]
errors = [5, 4, 6]

# Use SAME colors from MANUSCRIPT_COLORS
bar_colors = [MANUSCRIPT_COLORS[g] for g in groups]

bars = ax2.bar(groups, endpoint_values, color=bar_colors,
              edgecolor='black', linewidth=1.5, width=0.6)
ax2.errorbar(groups, endpoint_values, yerr=errors, fmt='none',
            ecolor='black', capsize=8, linewidth=2)

ax2.set_ylabel('Final Viability (%)', fontsize=11, fontweight='bold')
ax2.set_title('Figure 2: Endpoint Comparison\n(SAME colors as Figure 1 for consistency)',
             fontsize=12, fontweight='bold')
ax2.set_ylim(0, 110)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('manuscript_fig2_consistent_colors.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Colors consistent across Figures 1 and 2")
print("✓ Control always Gray, Drug_A always Blue, Drug_B always Red")
```

**Code Example (R) - Manuscript-Wide Consistency:**

```r
library(ggplot2)
library(dplyr)
library(patchwork)

# ============================================
# DEFINE COLORS ONCE - USE THROUGHOUT MANUSCRIPT
# ============================================

MANUSCRIPT_COLORS <- list(
  # Treatment groups
  Control = '#95A5A6',
  Drug_A = '#3498DB',
  Drug_B = '#E74C3C',

  # Expression
  Upregulated = '#C0392B',
  Downregulated = '#2874A6',
  Unchanged = '#D5D8DC',

  # Significance
  Significant = '#E67E22',
  NonSignificant = '#ECF0F1'
)

# ============================================
# FIGURE 1: Time Course
# ============================================

set.seed(42)
time <- seq(0, 24, by = 2)

data_fig1 <- data.frame(
  time = rep(time, 3),
  viability = c(
    100 + cumsum(rnorm(length(time), 0, 2)),
    100 + cumsum(rnorm(length(time), 0.5, 2)),
    100 + cumsum(rnorm(length(time), 1.0, 2))
  ),
  group = rep(c('Control', 'Drug_A', 'Drug_B'), each = length(time))
)

p1 <- ggplot(data_fig1, aes(x = time, y = viability, color = group)) +
  geom_line(size = 1.5) +
  geom_point(size = 4) +
  scale_color_manual(values = unlist(MANUSCRIPT_COLORS[c('Control', 'Drug_A', 'Drug_B')])) +
  labs(x = 'Time (hours)', y = 'Cell Viability (%)',
       title = 'Figure 1: Time Course Analysis\n(Colors: Control=Gray, Drug A=Blue, Drug B=Red)',
       color = NULL) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', size = 12),
    axis.title = element_text(face = 'bold'),
    legend.position = c(0.15, 0.85),
    legend.background = element_rect(fill = 'white', color = 'black', size = 0.5),
    panel.grid.major = element_line(color = 'gray90', size = 0.3)
  )

ggsave('manuscript_fig1_consistent_colors.png', p1, width = 7, height = 5,
       dpi = 300, bg = 'white')

# ============================================
# FIGURE 2: Endpoint Comparison (SAME COLORS)
# ============================================

data_fig2 <- data.frame(
  group = factor(c('Control', 'Drug_A', 'Drug_B'),
                levels = c('Control', 'Drug_A', 'Drug_B')),
  viability = c(85, 92, 78),
  error = c(5, 4, 6)
)

p2 <- ggplot(data_fig2, aes(x = group, y = viability, fill = group)) +
  geom_bar(stat = 'identity', color = 'black', size = 1, width = 0.6) +
  geom_errorbar(aes(ymin = viability - error, ymax = viability + error),
               width = 0.25, size = 1) +
  scale_fill_manual(values = unlist(MANUSCRIPT_COLORS[c('Control', 'Drug_A', 'Drug_B')])) +
  labs(y = 'Final Viability (%)',
       title = 'Figure 2: Endpoint Comparison\n(SAME colors as Figure 1 for consistency)') +
  ylim(0, 110) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', size = 12),
    axis.title.x = element_blank(),
    axis.title.y = element_text(face = 'bold'),
    legend.position = 'none',
    panel.grid.major.y = element_line(color = 'gray90', size = 0.3)
  )

ggsave('manuscript_fig2_consistent_colors.png', p2, width = 6, height = 5,
       dpi = 300, bg = 'white')

cat("✓ Colors consistent across Figures 1 and 2\n")
cat("✓ Control always Gray, Drug_A always Blue, Drug_B always Red\n")
```

---

### Field-Specific Color Conventions

Different scientific disciplines have evolved distinct color conventions. **Knowing and following these signals domain expertise.**

#### Molecular Biology & Genomics

**Standard Conventions:**

```
DNA/RNA/Protein:
✓ DNA: Blue
✓ RNA: Red
✓ Protein: Green or Purple

Gene Expression:
✓ Upregulated: Red spectrum
✓ Downregulated: Blue/Green spectrum
✓ No change: Gray/White

Western Blots:
✓ Grayscale (mimics X-ray film)
✓ Or: Single color gradient (e.g., green for fluorescence)
```

**Why these conventions:**
- Historical: DNA gels visualized with UV (blue), RNA with different dyes
- Red/green originally from microarray technology (Cy3/Cy5 dyes)
- Community agreement for rapid cross-study interpretation

#### Neuroscience

**Standard Conventions:**

```
Cell Types:
✓ Neurons: Often red or orange
✓ Astrocytes: Green or cyan
✓ Microglia: Blue or purple

Activity/Calcium Imaging:
✓ Low activity: Blue/Purple
✓ High activity: Yellow/Red (fire scale)
```

#### Climate Science

**Standard Conventions:**

```
Temperature Anomalies:
✓ Cooling: Blue
✓ Neutral: White/Beige
✓ Warming: Red/Orange
(Never reverse this!)

Precipitation:
✓ Drought: Brown/Yellow
✓ Normal: Green
✓ Excess: Blue

Elevation:
✓ Low: Green/Blue
✓ Mid: Yellow/Brown
✓ High: White (snow-capped)
```

**Code Example - Climate Convention (Python):**

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulate temperature anomaly data
np.random.seed(42)
years = np.arange(1900, 2024)
temp_anomaly = np.cumsum(np.random.randn(len(years)) * 0.1) - 0.5

fig, ax = plt.subplots(figsize=(10, 5))

# Use climate science convention: blue=cooling, red=warming
colors = ['#2166AC' if t < 0 else '#B2182B' for t in temp_anomaly]

ax.bar(years, temp_anomaly, color=colors, width=1, edgecolor='none')
ax.axhline(0, color='black', linewidth=2, linestyle='-')

ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Temperature Anomaly (°C)', fontsize=12, fontweight='bold')
ax.set_title('Climate Science Convention: Blue=Cooling, Red=Warming\n(Never reverse this!)',
             fontsize=13, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2166AC', label='Below average (cooling)'),
    Patch(facecolor='#B2182B', label='Above average (warming)')
]
ax.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig('climate_convention_colors.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example - Climate Convention (R):**

```r
library(ggplot2)

set.seed(42)
years <- 1900:2023
temp_anomaly <- cumsum(rnorm(length(years), 0, 0.1)) - 0.5

data <- data.frame(
  year = years,
  anomaly = temp_anomaly,
  direction = ifelse(temp_anomaly < 0, 'Below average (cooling)', 'Above average (warming)')
)

p <- ggplot(data, aes(x = year, y = anomaly, fill = direction)) +
  geom_bar(stat = 'identity', width = 1) +
  geom_hline(yintercept = 0, color = 'black', size = 1.5) +
  scale_fill_manual(values = c('Below average (cooling)' = '#2166AC',
                                'Above average (warming)' = '#B2182B')) +
  labs(x = 'Year', y = 'Temperature Anomaly (°C)',
       title = 'Climate Science Convention: Blue=Cooling, Red=Warming\n(Never reverse this!)',
       fill = NULL) +
  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', size = 13),
    axis.title = element_text(face = 'bold'),
    legend.position = c(0.2, 0.9),
    legend.background = element_rect(fill = 'white', color = 'black', size = 0.5),
    panel.grid.major.y = element_line(color = 'gray90', size = 0.3)
  )

ggsave('climate_convention_colors.png', p, width = 10, height = 5,
       dpi = 300, bg = 'white')
```

---

### When to Break Conventions (Rarely)

**Acceptable reasons to deviate:**

1. **Accessibility requirement**: Convention uses red-green (colorblind problematic)
   - Solution: Use blue-orange instead, note change in caption

2. **Technical constraint**: Equipment outputs specific colors
   - Example: Microscope fluorophores are fixed
   - Solution: State actual fluorophore colors in methods, use convention in schematic

3. **Conflicting conventions**: Interdisciplinary work where fields disagree
   - Solution: Choose one, state explicitly in caption, be consistent

**Unacceptable reasons:**

- "I like purple better than blue"
- "This looks cooler/more modern"
- "To make my figure stand out"

**How to break conventions properly:**

```
✓ State explicitly in figure caption:
  "Note: We use blue for upregulation and red for downregulation
   (reverse of typical convention) to match the color scheme
   established in our previous work (Smith et al., 2022)"

✓ Provide justification in methods section

✓ Ensure internal consistency remains absolute
```

---

### Exercise 2.6.1: Convention Recognition and Application

**Objective:** Identify field conventions and apply them correctly

**Part A: Convention Identification**

For your field, research and document:

1. **Standard color schemes** for common data types (check recent high-impact papers)
2. **Forbidden combinations** (colors that would confuse readers)
3. **Historical reasons** for these conventions (if discoverable)

**Format:**

```
Field: [Your discipline]

Convention 1:
- Data type: [e.g., Cell viability]
- Standard colors: [e.g., Green=live, Red=dead]
- Source: [e.g., Flow cytometry dye convention]
- Strength: [Strong/Moderate/Weak] (how universal is this?)

Forbidden combination:
- Never use: [e.g., Red for control group in my field]
- Reason: [e.g., Red always means "treatment" or "diseased"]
```

**Part B: Cross-Field Comparison**

Find a paper from a different discipline and analyze:

1. What color conventions differ from yours?
2. Which are universal (e.g., hot=red, cold=blue)?
3. If you had to present your data to that community, what would you change?

---

### Summary: Color Taste Checklist

**Before finalizing your manuscript's figures:**

- [ ] **Maximum 3-4 distinct hues** per figure (excluding grayscale)
- [ ] **Semantic consistency** across all figures (same concept = same color)
- [ ] **Field conventions respected** (or explicitly justified if broken)
- [ ] **Color assignments documented** (create a style guide for co-authors)
- [ ] **No decorative color** (every color has functional meaning)
- [ ] **Works in context** (looks professional compared to recent papers in target journal)

---

**End of Section 2.6**


## 2.7 Accessibility and Integrity in Color

### The Ethical Imperative

Color accessibility isn't a "nice-to-have" feature—it's an **ethical requirement** and increasingly a **publication mandate**. Approximately **8% of males and 0.5% of females** have some form of color vision deficiency (CVD), meaning 1 in 12 men and 1 in 200 women in your audience may struggle with your figures if you don't design for accessibility.

**Why this matters for scientific communication:**

1. **Inclusivity**: Exclude 8% of potential readers by default is unacceptable
2. **Information integrity**: Critical findings should be accessible to all
3. **Journal requirements**: Many journals now mandate colorblind-safe figures
4. **Reproducibility**: Accessible figures are clearer for everyone, not just CVD readers
5. **Professional standards**: Signals rigor and attention to detail

---

### Types of Color Vision Deficiency

Understanding the types helps you design effectively:

#### 1. Deuteranopia (Green-Blind, ~5% of males)

**What's affected:**
- Cannot distinguish between red and green
- Perceive them as variations of yellow/brown

**Problematic combinations:**
```
✗ Red vs. Green (appears as yellow vs. yellow)
✗ Red-green traffic light analogy (both look similar)
```

**Safe alternatives:**
```
✓ Blue vs. Orange
✓ Blue vs. Red (distinguishable by brightness difference)
✓ Purple vs. Yellow
```

#### 2. Protanopia (Red-Blind, ~2.5% of males)

**What's affected:**
- Cannot distinguish between red and green
- Red appears darker/brownish
- Similar confusion to deuteranopia but with brightness shifts

**Problematic combinations:**
```
✗ Red vs. Green
✗ Red vs. Orange (both appear similar)
```

#### 3. Tritanopia (Blue-Blind, <1% of population, rare)

**What's affected:**
- Cannot distinguish between blue and yellow
- Blue appears greenish, yellow appears pinkish

**Problematic combinations:**
```
✗ Blue vs. Yellow
✗ Blue vs. Green
```

**Note:** Most colorblind-safe palettes focus on deuteranopia/protanopia (most common).

---

### Designing for Color Accessibility

#### Strategy 1: Use Colorblind-Safe Palettes

**Pre-vetted palettes that work for all CVD types:**

**Okabe-Ito Palette (Recommended):**

```
Designed specifically for universal accessibility
8 colors, distinguishable by all CVD types

Colors:
#E69F00  Orange
#56B4E9  Sky Blue
#009E73  Bluish Green
#F0E442  Yellow
#0072B2  Blue
#D55E00  Vermilion (red-orange)
#CC79A7  Reddish Purple
#000000  Black
```

**ColorBrewer Palettes:**

```
Many ColorBrewer schemes are CVD-friendly
- "Set2" (qualitative, 8 colors)
- "Dark2" (qualitative, 8 colors)
- "RdYlBu" (diverging, if used correctly)

Check: colorbrewer2.org has CVD-safe filter

```

**Code Example (Python) - Okabe-Ito Implementation:**

```python
import matplotlib.pyplot as plt
import numpy as np

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

# Simulate grouped data
np.random.seed(42)
groups = ['Control', 'Treatment A', 'Treatment B', 'Treatment C']
values = [25, 32, 28, 35]
errors = [3, 4, 3.5, 4.5]

# Assign Okabe-Ito colors
group_colors = [
    OKABE_ITO['black'],          # Control: Black
    OKABE_ITO['sky_blue'],       # Treatment A: Sky Blue
    OKABE_ITO['vermilion'],      # Treatment B: Vermilion
    OKABE_ITO['bluish_green']    # Treatment C: Bluish Green
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Standard view
ax1 = axes[0]
bars1 = ax1.bar(groups, values, color=group_colors, edgecolor='black', linewidth=1.5)
ax1.errorbar(groups, values, yerr=errors, fmt='none', ecolor='black',
            capsize=8, linewidth=2)
ax1.set_ylabel('Response (arbitrary units)', fontsize=11, fontweight='bold')
ax1.set_title('Okabe-Ito Palette: Standard View\n(Colorblind-safe)',
             fontsize=12, fontweight='bold', color='green')
ax1.set_ylim(0, 45)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', alpha=0.3)

# Simulated deuteranopia view (approximation)
# Convert to grayscale perception weights for deuteranopia
def simulate_deuteranopia(hex_color):
    """Rough approximation of deuteranopia perception"""
    # Convert hex to RGB
    rgb = tuple(int(hex_color[i:i+2], 16)/255 for i in (1, 3, 5))
    # Deuteranopia: red and green channels affected
    # Simplified transformation (not exact)
    gray = 0.3 * rgb[0] + 0.4 * rgb[1] + 0.3 * rgb[2]
    adjusted = (rgb[2], gray, rgb[1])  # Shift perception
    return tuple(min(1, max(0, c)) for c in adjusted)

cvd_colors = [simulate_deuteranopia(c) for c in group_colors]

ax2 = axes[1]
bars2 = ax2.bar(groups, values, color=cvd_colors, edgecolor='black', linewidth=1.5)
ax2.errorbar(groups, values, yerr=errors, fmt='none', ecolor='black',
            capsize=8, linewidth=2)
ax2.set_ylabel('Response (arbitrary units)', fontsize=11, fontweight='bold')
ax2.set_title('Simulated Deuteranopia View\n(Colors still distinguishable)',
             fontsize=12, fontweight='bold', color='green')
ax2.set_ylim(0, 45)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('okabe_ito_accessibility.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Okabe-Ito palette used")
print("✓ Distinguishable by all color vision types")
```

**Code Example (R) - Okabe-Ito Implementation:**

```r
library(ggplot2)
library(patchwork)

# Okabe-Ito colorblind-safe palette
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

# Data
data <- data.frame(
  group = factor(c('Control', 'Treatment A', 'Treatment B', 'Treatment C'),
                levels = c('Control', 'Treatment A', 'Treatment B', 'Treatment C')),
  value = c(25, 32, 28, 35),
  error = c(3, 4, 3.5, 4.5)
)

# Assign colors
group_colors <- c(
  'Control' = OKABE_ITO['black'],
  'Treatment A' = OKABE_ITO['sky_blue'],
  'Treatment B' = OKABE_ITO['vermilion'],
  'Treatment C' = OKABE_ITO['bluish_green']
)

# Plot
p <- ggplot(data, aes(x = group, y = value, fill = group)) +
  geom_bar(stat = 'identity', color = 'black', size = 1, width = 0.7) +
  geom_errorbar(aes(ymin = value - error, ymax = value + error),
               width = 0.3, size = 1) +
  scale_fill_manual(values = group_colors) +
  labs(y = 'Response (arbitrary units)',
       title = 'Okabe-Ito Palette: Colorblind-Safe\n(Distinguishable by all color vision types)') +
  ylim(0, 45) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'darkgreen', size = 12),
    axis.title.x = element_blank(),
    axis.title.y = element_text(face = 'bold'),
    legend.position = 'none',
    panel.grid.major.y = element_line(color = 'gray90', size = 0.3)
  )

ggsave('okabe_ito_accessibility.png', p, width = 7, height = 5,
       dpi = 300, bg = 'white')

cat("✓ Okabe-Ito palette used\n")
cat("✓ Distinguishable by all color vision types\n")
```

---

#### Strategy 2: Redundant Encoding (Color + Something Else)

**The most robust approach: Never rely on color alone**

**Redundant channels:**
- **Color + Shape** (circles vs. squares vs. triangles)
- **Color + Line style** (solid vs. dashed vs. dotted)
- **Color + Pattern/Texture** (for bar charts: solid, striped, dotted)
- **Color + Position** (separate panels/facets)
- **Color + Text labels** (direct labeling on plot)

**Code Example (Python) - Redundant Encoding:**


```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.linspace(0, 10, 50)

# Three treatment groups
data = {
    'Control': (x, 5 + 0.5*x + np.random.randn(50)*0.5),
    'Treatment A': (x, 5 + 1.2*x + np.random.randn(50)*0.5),
    'Treatment B': (x, 5 + 0.8*x + np.random.randn(50)*0.5)
}

# Okabe-Ito colors
colors = {
    'Control': '#000000',      # Black
    'Treatment A': '#56B4E9',  # Sky Blue
    'Treatment B': '#D55E00'   # Vermilion
}

# Different markers (redundant encoding with shape)
markers = {
    'Control': 'o',        # Circle
    'Treatment A': 's',    # Square
    'Treatment B': '^'     # Triangle
}

# Different line styles (additional redundancy)
linestyles = {
    'Control': '-',         # Solid
    'Treatment A': '--',    # Dashed
    'Treatment B': '-.'     # Dash-dot
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Color only (vulnerable to CVD)
ax1 = axes[0]
for group, (x_data, y_data) in data.items():
    ax1.plot(x_data, y_data, color=colors[group], linewidth=2.5, label=group)

ax1.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Response', fontsize=11, fontweight='bold')
ax1.set_title('COLOR ONLY\n(Vulnerable if colors indistinguishable)',
             fontsize=12, fontweight='bold', color='red')
ax1.legend(loc='upper left', frameon=True, fontsize=10)
ax1.grid(alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Color + Shape + Line style (robust, redundant)
ax2 = axes[1]
for group, (x_data, y_data) in data.items():
    ax2.plot(x_data, y_data,
            color=colors[group],
            linestyle=linestyles[group],
            linewidth=2.5,
            marker=markers[group],
            markersize=6,
            markevery=5,  # Show marker every 5th point
            label=group)

ax2.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Response', fontsize=11, fontweight='bold')
ax2.set_title('COLOR + SHAPE + LINE STYLE\n(Accessible: multiple redundant cues)',
             fontsize=12, fontweight='bold', color='green')
ax2.legend(loc='upper left', frameon=True, fontsize=10)
ax2.grid(alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('redundant_encoding_accessibility.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Redundant encoding: Color + Shape + Line style")
print("✓ Information preserved even if color is lost")
```

**Code Example (R) - Redundant Encoding:**

```r
library(ggplot2)
library(dplyr)
library(patchwork)

set.seed(42)
x <- seq(0, 10, length.out = 50)

# Data
data <- data.frame(
  x = rep(x, 3),
  y = c(
    5 + 0.5*x + rnorm(50, 0, 0.5),
    5 + 1.2*x + rnorm(50, 0, 0.5),
    5 + 0.8*x + rnorm(50, 0, 0.5)
  ),
  group = rep(c('Control', 'Treatment A', 'Treatment B'), each = 50)
)

# Okabe-Ito colors
colors <- c(
  'Control' = '#000000',
  'Treatment A' = '#56B4E9',
  'Treatment B' = '#D55E00'
)

# Color only
p1 <- ggplot(data, aes(x = x, y = y, color = group)) +
  geom_line(size = 1.5) +
  scale_color_manual(values = colors) +
  labs(x = 'Time (hours)', y = 'Response',
       title = 'COLOR ONLY\n(Vulnerable if colors indistinguishable)',
       color = NULL) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'red', size = 12),
    axis.title = element_text(face = 'bold'),
    legend.position = c(0.2, 0.85),
    legend.background = element_rect(fill = 'white', color = 'black', size = 0.5),
    panel.grid.major = element_line(color = 'gray90', size = 0.3)
  )

# Color + Shape + Line style (redundant)
p2 <- ggplot(data, aes(x = x, y = y, color = group, linetype = group, shape = group)) +
  geom_line(size = 1.5) +
  geom_point(data = data %>% filter(row_number() %% 5 == 0), size = 3) +
  scale_color_manual(values = colors) +
  scale_linetype_manual(values = c('Control' = 'solid',
                                    'Treatment A' = 'dashed',
                                    'Treatment B' = 'dotdash')) +
  scale_shape_manual(values = c('Control' = 16, 'Treatment A' = 15, 'Treatment B' = 17)) +
  labs(x = 'Time (hours)', y = 'Response',
       title = 'COLOR + SHAPE + LINE STYLE\n(Accessible: multiple redundant cues)',
       color = NULL, linetype = NULL, shape = NULL) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'darkgreen', size = 12),
    axis.title = element_text(face = 'bold'),
    legend.position = c(0.2, 0.85),
    legend.background = element_rect(fill = 'white', color = 'black', size = 0.5),
    panel.grid.major = element_line(color = 'gray90', size = 0.3)
  )

# Combine
combined <- p1 | p2
ggsave('redundant_encoding_accessibility.png', combined, width = 14, height = 5,
       dpi = 300, bg = 'white')

cat("✓ Redundant encoding: Color + Shape + Line style\n")
cat("✓ Information preserved even if color is lost\n")
```

---

#### Strategy 3: Test Your Figures

**Don't guess—simulate how your figures appear to CVD readers**

**Tools for testing:**

**1. Color Oracle (Free, Desktop App)**

```
- Windows, Mac, Linux
- Real-time simulation overlay
- Shows Deuteranopia, Protanopia, Tritanopia views
- Download: colororacle.org
```

**2. Coblis (Web-based)**

```
- Upload image, see CVD simulations
- coblis.blogspot.com
```

**3. R Package: colorblindcheck**

```
library(colorblindcheck)

# Test your palette
palette_check(colors, plot = TRUE)
```

**4. Python: colorspacious**

```
from colorspacious import cspace_converter

# Test color pairs for distinguishability
```

**Testing workflow:**

```
1. Create figure with your colors
2. Export as PNG/PDF
3. Open in Color Oracle → Toggle CVD simulation
4. Check: Can you still distinguish all elements?
5. If not: Revise colors or add redundant encoding
6. Repeat until all CVD types are accessible
```

---

### Color Integrity: Ethical Use of Color

Beyond accessibility, **color integrity** means using color honestly—not to mislead or manipulate interpretation.

#### Integrity Principle 1: No Manipulation of Color Scales

**Common manipulations (unethical):**

**Misleading Technique 1: Non-linear color scales**

```
❌ DISHONEST:
Heatmap where:
- 0-50: Gentle blue gradient (subtle)
- 50-51: Sudden jump to deep red (dramatic)
→ Exaggerates small difference near threshold

✓ HONEST:
Linear or perceptually uniform scale
- Each color step = equal data step
```

**Misleading Technique 2: Asymmetric diverging scales**

```
❌ DISHONEST:
Fold-change heatmap:
- Downregulation: -10x to 0 (wide blue range)
- Upregulation: 0 to +2x (narrow red range)
→ Visually minimizes upregulation

✓ HONEST:
Symmetric scale: -10x to +10x
OR log-transform to make symmetric: log2(-10 to +10)
```

**Code Example (Python) - Honest vs. Dishonest Scales:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulate gene expression data with range -3 to +3 log2 fold change
np.random.seed(42)
data = np.random.randn(10, 10) * 1.5  # Symmetric around zero

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# DISHONEST: Asymmetric scale (manipulated to emphasize positive)
ax1 = axes[0]
im1 = ax1.imshow(data, cmap='RdBu_r', vmin=-3, vmax=1)  # Asymmetric!
ax1.set_title('DISHONEST: Asymmetric Scale\n(vmin=-3, vmax=1: exaggerates positives)',
             fontsize=12, fontweight='bold', color='red')
ax1.set_xticks([])
ax1.set_yticks([])
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Log2 Fold Change', fontsize=10, fontweight='bold')
cbar1.ax.axhline(0, color='black', linewidth=2)

# HONEST: Symmetric scale
ax2 = axes[1]
vmax = np.abs(data).max()  # Symmetric limit
im2 = ax2.imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax)  # Symmetric
ax2.set_title('HONEST: Symmetric Scale\n(Equal range for up/down regulation)',
             fontsize=12, fontweight='bold', color='green')
ax2.set_xticks([])
ax2.set_yticks([])
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Log2 Fold Change', fontsize=10, fontweight='bold')
cbar2.ax.axhline(0, color='black', linewidth=2)

plt.tight_layout()
plt.savefig('color_scale_integrity.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R) - Honest vs. Dishonest Scales:**

```r
library(ggplot2)
library(reshape2)
library(patchwork)

# Simulate data
set.seed(42)
data_matrix <- matrix(rnorm(100, 0, 1.5), 10, 10)
data_long <- melt(data_matrix)
names(data_long) <- c('Row', 'Col', 'FoldChange')

# DISHONEST: Asymmetric
p1 <- ggplot(data_long, aes(x = Col, y = Row, fill = FoldChange)) +
  geom_tile() +
  scale_fill_gradient2(low = '#2166AC', mid = 'white', high = '#B2182B',
                       midpoint = 0,
                       limits = c(-3, 1),  # Asymmetric!
                       name = 'Log2\nFold Change') +
  labs(title = 'DISHONEST: Asymmetric Scale\n(vmin=-3, vmax=1: exaggerates positives)') +
  theme_void() +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'red', size = 12),
    legend.position = 'right'
  )

# HONEST: Symmetric
vmax <- max(abs(data_long$FoldChange))

p2 <- ggplot(data_long, aes(x = Col, y = Row, fill = FoldChange)) +
  geom_tile() +
  scale_fill_gradient2(low = '#2166AC', mid = 'white', high = '#B2182B',
                       midpoint = 0,
                       limits = c(-vmax, vmax),  # Symmetric
                       name = 'Log2\nFold Change') +
  labs(title = 'HONEST: Symmetric Scale\n(Equal range for up/down regulation)') +
  theme_void() +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'darkgreen', size = 12),
    legend.position = 'right'
  )

combined <- p1 | p2
ggsave('color_scale_integrity.png', combined, width = 14, height = 6,
       dpi = 300, bg = 'white')
```

---

#### Integrity Principle 2: Consistent Processing Across Groups

**The rule:** If you adjust color/contrast/brightness for one image, apply **identical** adjustments to ALL comparison images.

**Common violation in microscopy:**

```
❌ DISHONEST:
Control image: Default exposure
Treated image: Increased brightness + contrast → looks more impressive

This is IMAGE MANIPULATION and grounds for retraction

✓ HONEST:
Same exposure settings for all images
Same post-processing (if any) applied uniformly
Document all adjustments in methods
```

**Journal requirements (e.g., Nature, Science, Cell):**

- No manipulation that obscures, eliminates, or misrepresents information
- Adjustments (brightness, contrast) must be linear and applied to entire image
- Must apply identically to controls and experimental groups
- Must disclose all adjustments in methods section
- Must provide original, unprocessed images if requested

---

#### Integrity Principle 3: Report Colormap Choices

**Transparency requirement:** State your colormap explicitly, especially if non-standard.

**In figure caption:**

```
"Heatmap colors represent correlation coefficients (blue = negative,
white = zero, red = positive) using the RdBu diverging colormap."
```

**In methods section:**

```
"All heatmaps were generated using the 'viridis' perceptually uniform
colormap to ensure accessibility for colorblind readers."
```

---

### Complete Accessibility Checklist

**Before submitting your manuscript:**

- [ ] **Colorblind-safe palette** used (Okabe-Ito, ColorBrewer CVD-safe, or tested)
- [ ] **Redundant encoding** employed (color + shape/line/texture for categories)
- [ ] **Tested with CVD simulator** (Color Oracle, Coblis, or similar)
- [ ] **Works in grayscale** (print figure in black & white—still interpretable?)
- [ ] **Symmetric scales** for diverging data (no manipulation)
- [ ] **Consistent processing** across comparison groups (microscopy, gels, etc.)
- [ ] **Colormap documented** in caption or methods
- [ ] **High contrast** (sufficient for low-vision readers)
- [ ] **No color-only legends** (combine with labels or patterns)
- [ ] **Field conventions respected** (unless accessibility requires deviation—state explicitly)

---

### Exercise 2.7.1: Accessibility Audit

**Objective:** Evaluate and improve accessibility of existing figures

**Part A: CVD Testing**

1. Find 3 figures from your recent work or literature
2. Run each through Color Oracle (or Coblis)
3. For each CVD type (Deuteranopia, Protanopia, Tritanopia), document:
   - Which elements become indistinguishable?
   - Is critical information lost?
   - How would you fix it?

**Format:**

```
Figure: [Description]
Original colors: [List]

Deuteranopia simulation:
- Problem: "Red and green bars look identical"
- Information lost: "Cannot distinguish Treatment A vs B"
- Fix: "Use blue vs. orange instead" OR "Add patterns to bars"

Accessibility score: [1-5, 5=perfect]
Recommended changes: [List specific actions]
```

**Part B: Redundant Encoding Practice**

Take one figure that relies solely on color distinction and redesign it with redundant encoding:

1. **Original:** Color only
2. **Redesign:** Color + Shape/Line/Pattern
3. **Test:** Works in grayscale?
4. **Compare:** Side-by-side screenshots

---

### Summary: Accessibility & Integrity Principles

**Accessibility:**
- Use colorblind-safe palettes (Okabe-Ito recommended)
- Employ redundant encoding (never color alone)
- Test with CVD simulators before submission
- Ensure grayscale compatibility

**Integrity:**
- No manipulated color scales (symmetric for diverging data)
- Consistent processing across comparison groups
- Transparent reporting of colormap choices
- Adherence to journal guidelines for image processing

**These are not optional—they are ethical requirements for responsible scientific communication.**

---

**The Language of Color**

**Key Takeaways:**
- **Color theory** provides technical foundation (RGB, CIELAB, perceptual uniformity)
- **Palette types** match data types (sequential, diverging, qualitative)
- **Saturation hierarchy** directs attention (deep=significant, light=background)
- **Restraint and consistency** signal professionalism (3-color max, semantic meaning)
- **Field conventions** facilitate interpretation (know and follow or justify deviations)
- **Accessibility** is mandatory (CVD-safe palettes, redundant encoding, testing)
- **Integrity** is ethical (no manipulation, symmetric scales, transparency)



## 2.8 Logical Color Mapping: Special Cases and Edge Scenarios

### The Critical Principle: Color Logic Must Match Data Reality

A common error in scientific figures is applying color schemes that create **false distinctions** or **hide meaningful patterns**. This section addresses edge cases where standard color choices fail logical tests.

---

### Scenario 1: All Values Are Significant

**The Problem:** When **every data point meets your significance threshold**, using a two-color saturation scheme (deep=significant, light=non-significant) becomes meaningless and misleading.

#### Case Study: Genome-Wide Significance

```
RNA-seq experiment:
- 500 genes tested
- All 500 show p < 0.05 (all significant)
- Fold changes range from +1.2x to +8.5x

❌ WRONG approach:
Use deep red for all genes (because all are "significant")
→ Loses information about magnitude differences
→ Visually monotonous, no hierarchy

✓ CORRECT approach:
Use continuous gradient based on effect size (fold change)
- Light red: Small but significant changes (+1.2x to +2x)
- Medium red: Moderate changes (+2x to +4x)
- Deep red: Large changes (+4x to +8.5x)
→ Color now represents biological importance, not statistical threshold
```

**Logical Reasoning:**

When the binary distinction (significant vs. non-significant) provides **no information** (all are significant), you must shift to encoding the **continuous quantitative variable** (effect size, correlation strength, etc.).

**Code Example (Python):**

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulate gene expression: ALL significant, varying magnitudes
np.random.seed(42)
n_genes = 50
n_samples = 8

# All genes have significant fold changes (1.2x to 8.5x)
fold_changes = np.random.uniform(1.2, 8.5, (n_genes, n_samples))
log2_fc = np.log2(fold_changes)

# All p-values < 0.05 (all significant)
p_values = np.random.uniform(0.0001, 0.049, n_genes)

genes = [f'Gene{i+1}' for i in range(n_genes)]
samples = [f'S{i+1}' for i in range(n_samples)]

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# WRONG: Binary coloring (all one color because all significant)
ax1 = axes[0]
# All get same deep red (useless)
im1 = ax1.imshow(log2_fc, cmap='Reds', vmin=0, vmax=3.1, aspect='auto')
ax1.set_title('❌ WRONG: All Significant → All Same Deep Color\n(Loses magnitude information)',
              fontsize=12, fontweight='bold', color='red')
ax1.set_xlabel('Samples', fontsize=10, fontweight='bold')
ax1.set_ylabel('Genes', fontsize=10, fontweight='bold')
ax1.set_xticks(range(n_samples))
ax1.set_xticklabels(samples, fontsize=8)
ax1.set_yticks(range(0, n_genes, 10))
ax1.set_yticklabels([genes[i] for i in range(0, n_genes, 10)], fontsize=7)

# Add misleading note
ax1.text(0.5, 1.05, 'All genes p < 0.05 → All deep red (uninformative)',
         transform=ax1.transAxes, ha='center', fontsize=9,
         style='italic', color='red', weight='bold')

cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046)
cbar1.set_label('Log₂ Fold Change\n(binary: all "significant")', fontsize=9)

# CORRECT: Continuous gradient based on magnitude
ax2 = axes[1]
im2 = ax2.imshow(log2_fc, cmap='YlOrRd', vmin=np.log2(1.2), vmax=np.log2(8.5), aspect='auto')
ax2.set_title('✓ CORRECT: Continuous Gradient by Magnitude\n(Shows biological importance)',
              fontsize=12, fontweight='bold', color='green')
ax2.set_xlabel('Samples', fontsize=10, fontweight='bold')
ax2.set_ylabel('Genes', fontsize=10, fontweight='bold')
ax2.set_xticks(range(n_samples))
ax2.set_xticklabels(samples, fontsize=8)
ax2.set_yticks(range(0, n_genes, 10))
ax2.set_yticklabels([genes[i] for i in range(0, n_genes, 10)], fontsize=7)

# Add explanatory note
ax2.text(0.5, 1.05, 'All genes p < 0.05 → Color = effect size (informative)',
         transform=ax2.transAxes, ha='center', fontsize=9,
         style='italic', color='green', weight='bold')

cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046)
cbar2.set_label('Log₂ Fold Change\n(continuous magnitude)', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('all_significant_color_logic.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()

print("✓ When all values significant: encode magnitude, not binary threshold")
```

**Code Example (R):**

```r
library(ggplot2)
library(reshape2)
library(patchwork)

# Simulate data: ALL significant
set.seed(42)
n_genes <- 50
n_samples <- 8

fold_changes <- matrix(runif(n_genes * n_samples, 1.2, 8.5), n_genes, n_samples)
log2_fc <- log2(fold_changes)

# All p < 0.05
p_values <- runif(n_genes, 0.0001, 0.049)

genes <- paste0('Gene', 1:n_genes)
samples <- paste0('S', 1:n_samples)

colnames(log2_fc) <- samples
rownames(log2_fc) <- genes

# Convert to long format
data_long <- melt(log2_fc, varnames = c('Gene', 'Sample'), value.name = 'Log2FC')

# WRONG: All same deep color (uninformative)
p1 <- ggplot(data_long, aes(x = Sample, y = Gene, fill = Log2FC)) +
  geom_tile() +
  scale_fill_gradient(low = '#FEE5D9', high = '#A50F15',  # Fixed deep red for "all significant"
                      limits = c(0, 3.1),
                      name = 'Log₂ FC\n(binary)') +
  labs(title = '❌ WRONG: All Significant → All Same Deep Color\n(Loses magnitude information)',
       x = 'Samples', y = 'Genes') +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'red', size = 11),
    axis.title = element_text(face = 'bold', size = 10),
    axis.text.y = element_text(size = 6),
    axis.text.x = element_text(size = 8),
    panel.grid = element_blank(),
    legend.position = 'right'
  ) +
  annotate('text', x = 4.5, y = 52, label = 'All genes p < 0.05 → All deep red (uninformative)',
           size = 3, fontface = 'italic', color = 'red')

# CORRECT: Continuous gradient
p2 <- ggplot(data_long, aes(x = Sample, y = Gene, fill = Log2FC)) +
  geom_tile() +
  scale_fill_gradientn(colors = c('#FFFFCC', '#FFEDA0', '#FED976', '#FEB24C',
                                   '#FD8D3C', '#FC4E2A', '#E31A1C', '#BD0026'),
                       limits = c(log2(1.2), log2(8.5)),
                       name = 'Log₂ FC\n(magnitude)') +
  labs(title = '✓ CORRECT: Continuous Gradient by Magnitude\n(Shows biological importance)',
       x = 'Samples', y = 'Genes') +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'darkgreen', size = 11),
    axis.title = element_text(face = 'bold', size = 10),
    axis.text.y = element_text(size = 6),
    axis.text.x = element_text(size = 8),
    panel.grid = element_blank(),
    legend.position = 'right'
  ) +
  annotate('text', x = 4.5, y = 52, label = 'All genes p < 0.05 → Color = effect size (informative)',
           size = 3, fontface = 'italic', color = 'darkgreen')

# Combine
combined <- p1 / p2
ggsave('all_significant_color_logic.png', combined, width = 10, height = 12,
       dpi = 300, bg = 'white')

cat("✓ When all values significant: encode magnitude, not binary threshold\n")
```

---

### Scenario 2: Missing Data in Heatmaps

**The Problem:** Missing values (NA, not measured, failed QC) require a **distinct visual treatment** that doesn't interfere with your continuous colormap.

#### The Logical Rules:

1. **Never use a color from your gradient** for missing data
   - If gradient is blue-white-red, don't use gray (looks like "low value")
   - Missing ≠ Zero ≠ Low value

2. **Use a visually distinct, non-data color**
   - **Best: Crosshatch/pattern** (universally understood as "no data")
   - **Good: Distinct color outside gradient** (e.g., black, white with border)
   - **Acceptable: Very light gray with clear border** (if pattern not possible)

3. **Label explicitly in legend**
   - "Missing/Not measured" with example patch

#### Common Mistakes:

```
❌ MISTAKE 1: Using gradient color for NA
Gradient: Light blue → Dark blue
Missing data: Light blue or white
→ Looks like "low value" not "absent"

❌ MISTAKE 2: Using zero for NA
Substituting NA with 0 in data
→ Falsely implies measurement of zero

❌ MISTAKE 3: No visual distinction
Leaving cells blank (white) when background is white
→ Invisible, looks like data omission error
```

**Code Example (Python) - Missing Data:**

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Simulate expression data with missing values
np.random.seed(42)
n_genes = 20
n_samples = 10

data = np.random.randn(n_genes, n_samples) * 2 + 5  # Mean ~5

# Introduce random missing values (20%)
missing_mask = np.random.rand(n_genes, n_samples) < 0.2
data_with_na = data.copy()
data_with_na[missing_mask] = np.nan

genes = [f'Gene{i+1}' for i in range(n_genes)]
samples = [f'S{i+1}' for i in range(n_samples)]

fig, axes = plt.subplots(1, 3, figsize=(16, 7))

# BAD 1: Missing data shown in gradient color (looks like low value)
ax1 = axes[0]
data_bad1 = np.nan_to_num(data_with_na, nan=0)  # Replace NA with 0
im1 = ax1.imshow(data_bad1, cmap='YlOrRd', vmin=0, vmax=10, aspect='auto')
ax1.set_title('❌ BAD: NA → 0 (Looks like real low value)',
              fontsize=11, fontweight='bold', color='red')
ax1.set_xlabel('Samples', fontsize=9, fontweight='bold')
ax1.set_ylabel('Genes', fontsize=9, fontweight='bold')
ax1.set_xticks(range(n_samples))
ax1.set_xticklabels(samples, fontsize=7)
ax1.set_yticks(range(0, n_genes, 5))
ax1.set_yticklabels([genes[i] for i in range(0, n_genes, 5)], fontsize=7)
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046)
cbar1.set_label('Expression', fontsize=9)

# BAD 2: Missing data in white (invisible on white background)
ax2 = axes[1]
im2 = ax2.imshow(data_with_na, cmap='YlOrRd', vmin=0, vmax=10, aspect='auto')
ax2.set_title('❌ BAD: NA → White (Invisible/ambiguous)',
              fontsize=11, fontweight='bold', color='red')
ax2.set_xlabel('Samples', fontsize=9, fontweight='bold')
ax2.set_ylabel('Genes', fontsize=9, fontweight='bold')
ax2.set_xticks(range(n_samples))
ax2.set_xticklabels(samples, fontsize=7)
ax2.set_yticks(range(0, n_genes, 5))
ax2.set_yticklabels([genes[i] for i in range(0, n_genes, 5)], fontsize=7)
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046)
cbar2.set_label('Expression', fontsize=9)

# GOOD: Missing data with distinct visual (gray + border)
ax3 = axes[2]
# Plot data normally
im3 = ax3.imshow(data_with_na, cmap='YlOrRd', vmin=0, vmax=10, aspect='auto')

# Overlay missing data with distinct color + pattern
for i in range(n_genes):
    for j in range(n_samples):
        if missing_mask[i, j]:
            # Draw rectangle with distinct color (gray) and thick black border
            rect = Rectangle((j-0.5, i-0.5), 1, 1,
                            facecolor='#CCCCCC',
                            edgecolor='black',
                            linewidth=2,
                            hatch='///')  # Crosshatch pattern
            ax3.add_patch(rect)

ax3.set_title('✓ GOOD: NA → Distinct Color + Pattern',
              fontsize=11, fontweight='bold', color='green')
ax3.set_xlabel('Samples', fontsize=9, fontweight='bold')
ax3.set_ylabel('Genes', fontsize=9, fontweight='bold')
ax3.set_xticks(range(n_samples))
ax3.set_xticklabels(samples, fontsize=7)
ax3.set_yticks(range(0, n_genes, 5))
ax3.set_yticklabels([genes[i] for i in range(0, n_genes, 5)], fontsize=7)

# Custom colorbar with NA indicator
cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046)
cbar3.set_label('Expression', fontsize=9, fontweight='bold')

# Add legend for missing data
missing_patch = mpatches.Patch(facecolor='#CCCCCC', edgecolor='black',
                               hatch='///', linewidth=2, label='Missing/NA')
ax3.legend(handles=[missing_patch], loc='upper right', fontsize=8, frameon=True)

plt.tight_layout()
plt.savefig('missing_data_color_logic.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Missing data requires distinct visual treatment")
print("✓ Never use gradient colors for NA values")
```

**Code Example (R) - Missing Data:**

```r
library(ggplot2)
library(reshape2)
library(dplyr)
library(patchwork)

# Simulate data with missing values
set.seed(42)
n_genes <- 20
n_samples <- 10

data_matrix <- matrix(rnorm(n_genes * n_samples, 5, 2), n_genes, n_samples)

# Introduce missing values
missing_mask <- matrix(runif(n_genes * n_samples) < 0.2, n_genes, n_samples)
data_with_na <- data_matrix
data_with_na[missing_mask] <- NA

genes <- paste0('Gene', 1:n_genes)
samples <- paste0('S', 1:n_samples)

colnames(data_with_na) <- samples
rownames(data_with_na) <- genes

# Convert to long format
data_long <- melt(data_with_na, varnames = c('Gene', 'Sample'), value.name = 'Expression')
data_long$is_missing <- is.na(data_long$Expression)

# BAD 1: Replace NA with 0 (misleading)
data_bad1 <- data_long
data_bad1$Expression[is.na(data_bad1$Expression)] <- 0

p1 <- ggplot(data_bad1, aes(x = Sample, y = Gene, fill = Expression)) +
  geom_tile(color = 'white', size = 0.5) +
  scale_fill_gradientn(colors = c('#FFFFCC', '#FED976', '#FD8D3C', '#E31A1C'),
                       limits = c(0, 10),
                       name = 'Expression') +
  labs(title = '❌ BAD: NA → 0 (Looks like real low value)') +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'red', size = 11),
    axis.text.y = element_text(size = 7),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    panel.grid = element_blank()
  )

# BAD 2: Missing data invisible (white on white)
p2 <- ggplot(data_long, aes(x = Sample, y = Gene, fill = Expression)) +
  geom_tile(color = 'white', size = 0.5) +
  scale_fill_gradientn(colors = c('#FFFFCC', '#FED976', '#FD8D3C', '#E31A1C'),
                       limits = c(0, 10),
                       name = 'Expression',
                       na.value = 'white') +  # NA = white (invisible)
  labs(title = '❌ BAD: NA → White (Invisible/ambiguous)') +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'red', size = 11),
    axis.text.y = element_text(size = 7),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    panel.grid = element_blank()
  )

# GOOD: Distinct color + pattern for NA
p3 <- ggplot(data_long, aes(x = Sample, y = Gene, fill = Expression)) +
  geom_tile(color = 'white', size = 0.5) +
  scale_fill_gradientn(colors = c('#FFFFCC', '#FED976', '#FD8D3C', '#E31A1C'),
                       limits = c(0, 10),
                       name = 'Expression',
                       na.value = '#CCCCCC') +  # NA = distinct gray
  # Add border for missing cells
  geom_tile(data = data_long %>% filter(is_missing),
            aes(x = Sample, y = Gene),
            fill = '#CCCCCC', color = 'black', size = 1.5, alpha = 0.7) +
  labs(title = '✓ GOOD: NA → Distinct Color + Border',
       caption = 'Gray cells with black border = Missing/Not measured') +
  theme_minimal(base_size = 10) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'darkgreen', size = 11),
    plot.caption = element_text(hjust = 0.5, face = 'italic', size = 9),
    axis.text.y = element_text(size = 7),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    panel.grid = element_blank()
  )

# Combine
combined <- (p1 | p2) / p3
ggsave('missing_data_color_logic.png', combined, width = 14, height = 12,
       dpi = 300, bg = 'white')

cat("✓ Missing data requires distinct visual treatment\n")
cat("✓ Never use gradient colors for NA values\n")
```

---

### Scenario 3: Zero is Meaningful vs. Zero is Arbitrary

**The Logical Distinction:**

Some datasets have a **meaningful zero point** (requires diverging colormap), while others have **arbitrary zero** (requires sequential colormap).

#### Case A: Meaningful Zero (Use Diverging)

```
Examples where zero has special meaning:
- Fold change: 0 = no change (negative = decrease, positive = increase)
- Temperature anomaly: 0 = average (negative = below, positive = above)
- Financial profit/loss: 0 = break-even
- pH: 7 = neutral (< 7 acidic, > 7 basic)
- Correlation: 0 = no relationship

✓ CORRECT: Diverging colormap (Blue ← White (zero) → Red)
```

#### Case B: Arbitrary Zero (Use Sequential)

```
Examples where zero is just lower bound:
- Gene expression (FPKM): 0 = not expressed, but not "negative expression"
- Temperature in Kelvin: 0K is absolute zero, but no "negative temp"
- Count data: 0 = none detected, but no "negative counts"
- Distance: 0 = here, but no "negative distance"

✓ CORRECT: Sequential colormap (Light → Dark single hue)
❌ WRONG: Diverging colormap (implies negative values exist)
```

**Code Example (Python):**

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Case A: Fold change (meaningful zero)
fold_change_data = np.random.randn(10, 10) * 2  # Centered at 0

# Case B: Gene expression counts (arbitrary zero, all positive)
expression_data = np.abs(np.random.randn(10, 10) * 2) + 1  # All positive

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# CASE A - WRONG: Sequential for data with meaningful zero
ax1 = axes[0, 0]
im1 = ax1.imshow(fold_change_data, cmap='Reds', vmin=-4, vmax=4, aspect='auto')
ax1.set_title('❌ WRONG: Sequential for Fold Change\n(Zero is meaningful midpoint!)',
              fontsize=11, fontweight='bold', color='red')
plt.colorbar(im1, ax=ax1, label='Fold Change')
ax1.set_xticks([])
ax1.set_yticks([])

# CASE A - CORRECT: Diverging for data with meaningful zero
ax2 = axes[0, 1]
im2 = ax2.imshow(fold_change_data, cmap='RdBu_r', vmin=-4, vmax=4, aspect='auto')
ax2.set_title('✓ CORRECT: Diverging for Fold Change\n(White = no change)',
              fontsize=11, fontweight='bold', color='green')
cbar2 = plt.colorbar(im2, ax=ax2, label='Fold Change')
cbar2.ax.axhline(0, color='black', linewidth=2)
ax2.set_xticks([])
ax2.set_yticks([])

# CASE B - WRONG: Diverging for all-positive data
ax3 = axes[1, 0]
im3 = ax3.imshow(expression_data, cmap='RdBu_r', vmin=0, vmax=6, aspect='auto')
ax3.set_title('❌ WRONG: Diverging for Expression\n(Implies negative values exist)',
              fontsize=11, fontweight='bold', color='red')
plt.colorbar(im3, ax=ax3, label='Expression (FPKM)')
ax3.set_xticks([])
ax3.set_yticks([])

# CASE B - CORRECT: Sequential for all-positive data
ax4 = axes[1, 1]
im4 = ax4.imshow(expression_data, cmap='YlOrRd', vmin=0, vmax=6, aspect='auto')
ax4.set_title('✓ CORRECT: Sequential for Expression\n(Zero = lower bound)',
              fontsize=11, fontweight='bold', color='green')
plt.colorbar(im4, ax=ax4, label='Expression (FPKM)')
ax4.set_xticks([])
ax4.set_yticks([])

plt.tight_layout()
plt.savefig('meaningful_zero_logic.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Meaningful zero → Diverging colormap")
print("✓ Arbitrary zero → Sequential colormap")
```

**Code Example (R):**

```r
library(ggplot2)
library(reshape2)
library(patchwork)

set.seed(42)

# Case A: Fold change (meaningful zero)
fold_change <- matrix(rnorm(100, 0, 2), 10, 10)

# Case B: Expression (arbitrary zero, all positive)
expression <- matrix(abs(rnorm(100, 0, 2)) + 1, 10, 10)

# Convert to long
fc_long <- melt(fold_change)
names(fc_long) <- c('Row', 'Col', 'FoldChange')

expr_long <- melt(expression)
names(expr_long) <- c('Row', 'Col', 'Expression')

# CASE A - WRONG
p1 <- ggplot(fc_long, aes(x = Col, y = Row, fill = FoldChange)) +
  geom_tile() +
  scale_fill_gradient(low = 'white', high = 'red',
                      limits = c(-4, 4),
                      name = 'Fold\nChange') +
  labs(title = '❌ WRONG: Sequential for Fold Change\n(Zero is meaningful midpoint!)') +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5, face = 'bold', color = 'red', size = 10))

# CASE A - CORRECT
p2 <- ggplot(fc_long, aes(x = Col, y = Row, fill = FoldChange)) +
  geom_tile() +
  scale_fill_gradient2(low = '#2166AC', mid = 'white', high = '#B2182B',
                       midpoint = 0,
                       limits = c(-4, 4),
                       name = 'Fold\nChange') +
  labs(title = '✓ CORRECT: Diverging for Fold Change\n(White = no change)') +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5, face = 'bold', color = 'darkgreen', size = 10))

# CASE B - WRONG
p3 <- ggplot(expr_long, aes(x = Col, y = Row, fill = Expression)) +
  geom_tile() +
  scale_fill_gradient2(low = 'blue', mid = 'white', high = 'red',
                       midpoint = 3,
                       limits = c(0, 6),
                       name = 'Expression\n(FPKM)') +
  labs(title = '❌ WRONG: Diverging for Expression\n(Implies negative values exist)') +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5, face = 'bold', color = 'red', size = 10))

# CASE B - CORRECT
p4 <- ggplot(expr_long, aes(x = Col, y = Row, fill = Expression)) +
  geom_tile() +
  scale_fill_gradientn(colors = c('#FFFFCC', '#FED976', '#FD8D3C', '#E31A1C'),
                       limits = c(0, 6),
                       name = 'Expression\n(FPKM)') +
  labs(title = '✓ CORRECT: Sequential for Expression\n(Zero = lower bound)') +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5, face = 'bold', color = 'darkgreen', size = 10))

combined <- (p1 | p2) / (p3 | p4)
ggsave('meaningful_zero_logic.png', combined, width = 12, height = 10,
       dpi = 300, bg = 'white')

cat("✓ Meaningful zero → Diverging colormap\n")
cat("✓ Arbitrary zero → Sequential colormap\n")
```

---

### Summary: Logical Color Mapping Rules

**Rule 1: When significance is universal**

```
If ALL values meet threshold (all p < 0.05):
→ Encode magnitude/effect size with continuous gradient
→ Binary distinction provides no information
```

**Rule 2: Missing data treatment**

```
Missing values require:
→ Color OUTSIDE your gradient (not gray if gradient uses gray)
→ Visual distinction (pattern, border, or unique color)
→ Explicit legend entry: "Missing/Not measured"
```

**Rule 3: Zero point semantics**

```
Meaningful zero (fold change, anomaly):
→ Diverging colormap (blue ← white (zero) → red)

Arbitrary zero (counts, absolute scale):
→ Sequential colormap (light → dark single hue)
```

---

### Exercise 2.8.1: Logical Color Mapping Audit

**Objective:** Identify and fix logical color mapping errors

**Instructions:**

For each scenario, determine:
1. What is wrong with the current color scheme?
2. What does the data actually represent?
3. What colormap would be logically correct?
4. Why does the logical choice matter for interpretation?

**Scenarios:**

**A. RNA-seq heatmap**
- Data: All 1000 genes have p < 0.01 (all highly significant)
- Current coloring: Deep red for all (because all significant)
- Fold changes: Range from +1.5x to +12x


**B. Spatial temperature map**
- Data: Average January temperatures across cities (range: -15°C to +25°C)
- Current coloring: Diverging blue-white-red with white at 0°C
- Question: Is 0°C meaningful here?

**C. Correlation heatmap with missing data**
- Data: Pearson correlations (-1 to +1), 15% cells have insufficient data
- Current coloring: RdBu diverging, missing cells shown as light blue
- Question: What's the problem?

**D. Western blot quantification**
- Data: Protein band intensities (arbitrary units, 0-255)
- Current coloring: Diverging colormap centered at 128
- Question: Does 128 have special biological meaning?

**E. Clinical trial with mixed significance**
- Data: 20 endpoints measured, 3 show p<0.05, 17 show p>0.05
- Current coloring: All bars in shades of blue gradient
- Question: How should significance be encoded?

**Your answers should include:**

```
Scenario [Letter]:

Current approach:
[Describe what they're doing wrong]

Data reality:
[What the numbers actually represent]

Logical issue:
[Why current approach is misleading/uninformative]

Correct solution:
[Specific colormap + justification]

Example:
Scenario A: All significant genes

Current: Deep red for all (binary: significant=red)
Data reality: All p<0.01, fold changes vary 1.5x-12x
Logical issue: Binary coloring loses magnitude information
Correct solution: Continuous yellow-orange-red gradient encoding fold change
  → Light yellow: +1.5x, Deep red: +12x
  → Reader sees biological importance, not just statistical threshold
```

---

### Scenario 4: Cyclic Data (Time of Day, Angles, Compass Directions)

**The Special Case:** Some data is **cyclic**—the end wraps back to the beginning (e.g., 23:59 wraps to 00:00).

#### Logical Requirement: Cyclic Colormap

**Data types needing cyclic colormaps:**
- Time of day (24-hour clock)
- Day of year (Dec 31 → Jan 1)
- Compass directions (359° → 0°)
- Phase angles (2π → 0)
- Seasonal patterns

**Standard colormap FAILS:**

```
❌ Sequential colormap (light blue → dark blue):
  - 23:00 appears dark blue (high value)
  - 00:00 appears light blue (low value)
  → False discontinuity! These times are adjacent

✓ Cyclic colormap (e.g., 'twilight', 'hsv'):
  - Start and end colors are perceptually similar
  - Smooth transition across wraparound point
```

**Code Example (Python) - Cyclic Data:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulate circadian gene expression (24-hour cycle)
np.random.seed(42)
hours = np.arange(0, 24, 0.5)  # Every 30 min
expression = 5 + 3 * np.sin(2 * np.pi * (hours - 6) / 24) + np.random.randn(len(hours)) * 0.5

# Create circular heatmap data (genes x time)
n_genes = 15
circular_data = np.array([
    5 + 3 * np.sin(2 * np.pi * (hours - np.random.randint(0, 24)) / 24) +
    np.random.randn(len(hours)) * 0.5
    for _ in range(n_genes)
])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# WRONG: Sequential colormap (creates false discontinuity)
ax1 = axes[0]
im1 = ax1.imshow(circular_data, cmap='viridis', aspect='auto',
                 extent=[0, 24, 0, n_genes])
ax1.set_xlabel('Time of Day (hours)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Gene', fontsize=11, fontweight='bold')
ax1.set_title('❌ WRONG: Sequential for Cyclic Data\n(23:00 and 00:00 look unrelated)',
              fontsize=12, fontweight='bold', color='red')
ax1.axvline(0, color='white', linestyle='--', linewidth=2, alpha=0.7)
ax1.axvline(24, color='white', linestyle='--', linewidth=2, alpha=0.7)
ax1.text(12, n_genes + 0.5, 'False boundary at midnight',
         ha='center', fontsize=9, color='red', fontweight='bold')
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Expression', fontsize=10)

# CORRECT: Cyclic colormap (twilight - perceptually uniform cyclic)
ax2 = axes[1]
im2 = ax2.imshow(circular_data, cmap='twilight', aspect='auto',
                 extent=[0, 24, 0, n_genes])
ax2.set_xlabel('Time of Day (hours)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Gene', fontsize=11, fontweight='bold')
ax2.set_title('✓ CORRECT: Cyclic Colormap\n(Smooth across midnight)',
              fontsize=12, fontweight='bold', color='green')
ax2.axvline(0, color='white', linestyle='--', linewidth=2, alpha=0.7)
ax2.axvline(24, color='white', linestyle='--', linewidth=2, alpha=0.7)
ax2.text(12, n_genes + 0.5, 'Continuous across midnight',
         ha='center', fontsize=9, color='green', fontweight='bold')
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Expression', fontsize=10)

plt.tight_layout()
plt.savefig('cyclic_data_colormap.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Cyclic data requires cyclic colormap (twilight, hsv)")
print("✓ Start and end colors must be perceptually similar")
```

**Code Example (R) - Cyclic Data:**

```r
library(ggplot2)
library(reshape2)
library(patchwork)

# Simulate circadian data
set.seed(42)
hours <- seq(0, 24, by = 0.5)
n_genes <- 15

circular_data <- sapply(1:n_genes, function(i) {
  phase <- runif(1, 0, 24)
  5 + 3 * sin(2 * pi * (hours - phase) / 24) + rnorm(length(hours), 0, 0.5)
})

# Convert to long format
data_long <- melt(circular_data)
names(data_long) <- c('Time', 'Gene', 'Expression')
data_long$Time <- hours[data_long$Time]

# WRONG: Sequential colormap
p1 <- ggplot(data_long, aes(x = Time, y = Gene, fill = Expression)) +
  geom_tile() +
  scale_fill_viridis_c(option = 'viridis', name = 'Expression') +
  geom_vline(xintercept = c(0, 24), color = 'white', linetype = 'dashed', size = 1) +
  labs(x = 'Time of Day (hours)', y = 'Gene',
       title = '❌ WRONG: Sequential for Cyclic Data\n(23:00 and 00:00 look unrelated)',
       caption = 'False boundary at midnight') +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'red', size = 12),
    plot.caption = element_text(hjust = 0.5, color = 'red', face = 'italic'),
    axis.title = element_text(face = 'bold'),
    panel.grid = element_blank()
  )

# CORRECT: Create cyclic palette manually (R doesn't have built-in twilight)
# Using HSV color space for cyclic effect
cyclic_colors <- hsv(seq(0, 1, length.out = 100), s = 0.7, v = 0.8)

p2 <- ggplot(data_long, aes(x = Time, y = Gene, fill = Expression)) +
  geom_tile() +
  scale_fill_gradientn(colors = cyclic_colors, name = 'Expression') +
  geom_vline(xintercept = c(0, 24), color = 'white', linetype = 'dashed', size = 1) +
  labs(x = 'Time of Day (hours)', y = 'Gene',
       title = '✓ CORRECT: Cyclic Colormap\n(Smooth across midnight)',
       caption = 'Continuous across midnight') +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'darkgreen', size = 12),
    plot.caption = element_text(hjust = 0.5, color = 'darkgreen', face = 'italic'),
    axis.title = element_text(face = 'bold'),
    panel.grid = element_blank()
  )

combined <- p1 / p2
ggsave('cyclic_data_colormap.png', combined, width = 10, height = 10,
       dpi = 300, bg = 'white')

cat("✓ Cyclic data requires cyclic colormap\n")
cat("✓ Start and end colors must be perceptually similar\n")
```

---

### Scenario 5: Multiple Scales in One Figure

**The Problem:** When showing multiple datasets with **different value ranges** in the same figure, color scales must be handled carefully.

#### Case A: Shared Scale (When Appropriate)

**Use when:**
- Data types are directly comparable (same units, same meaning)
- Relative magnitudes matter
- Example: Same gene measured across conditions

```
✓ CORRECT: Shared color scale
Gene expression (FPKM) in 3 tissues
- All use same scale: 0-100 FPKM
- Direct visual comparison valid
```

#### Case B: Independent Scales (When Necessary)

**Use when:**
- Different data types (e.g., temperature vs. precipitation)
- Vastly different ranges would obscure one dataset
- Example: Gene A (range 1-10) vs. Gene B (range 100-1000)

```
✓ CORRECT: Independent scales, clearly labeled
- Gene A: 0-10 scale
- Gene B: 0-1000 scale
- State explicitly in caption or on colorbars
```

**Critical Rule:** **Never use same colormap for different scales without explicit labeling**

**Code Example (Python):**

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Two genes with very different expression ranges
gene_low = np.random.uniform(1, 10, (10, 10))  # Low expression
gene_high = np.random.uniform(100, 1000, (10, 10))  # High expression

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# MISLEADING: Same colormap, not labeled properly
ax1 = axes[0, 0]
im1 = ax1.imshow(gene_low, cmap='Reds', vmin=0, vmax=1000, aspect='auto')
ax1.set_title('Gene A (Low Range)', fontsize=11, fontweight='bold')
plt.colorbar(im1, ax=ax1, label='Expression')

ax2 = axes[0, 1]
im2 = ax2.imshow(gene_high, cmap='Reds', vmin=0, vmax=1000, aspect='auto')
ax2.set_title('Gene B (High Range)', fontsize=11, fontweight='bold')
plt.colorbar(im2, ax=ax2, label='Expression')

axes[0, 0].text(0.5, -0.15, '❌ MISLEADING: Same scale (Gene A looks uniformly low)',
                transform=axes[0, 0].transAxes, ha='center', fontsize=10,
                color='red', fontweight='bold')

# CORRECT: Independent scales, clearly labeled
ax3 = axes[1, 0]
im3 = ax3.imshow(gene_low, cmap='Reds', vmin=1, vmax=10, aspect='auto')
ax3.set_title('Gene A (Independent Scale)', fontsize=11, fontweight='bold')
cbar3 = plt.colorbar(im3, ax=ax3)
cbar3.set_label('Expression\n(1-10 FPKM)', fontsize=9, fontweight='bold')

ax4 = axes[1, 1]
im4 = ax4.imshow(gene_high, cmap='Reds', vmin=100, vmax=1000, aspect='auto')
ax4.set_title('Gene B (Independent Scale)', fontsize=11, fontweight='bold')
cbar4 = plt.colorbar(im4, ax=ax4)
cbar4.set_label('Expression\n(100-1000 FPKM)', fontsize=9, fontweight='bold')

axes[1, 0].text(0.5, -0.15, '✓ CORRECT: Independent scales, clearly labeled',
                transform=axes[1, 0].transAxes, ha='center', fontsize=10,
                color='green', fontweight='bold')

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig('multiple_scales_logic.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Multiple scales require explicit labeling")
print("✓ State ranges clearly in captions and colorbars")
```

**Code Example (R):**

```r
library(ggplot2)
library(reshape2)
library(patchwork)

set.seed(42)

# Two genes with different ranges
gene_low <- matrix(runif(100, 1, 10), 10, 10)
gene_high <- matrix(runif(100, 100, 1000), 10, 10)

# Convert to long
low_long <- melt(gene_low)
names(low_long) <- c('Row', 'Col', 'Expression')
low_long$Gene <- 'Gene A'

high_long <- melt(gene_high)
names(high_long) <- c('Row', 'Col', 'Expression')
high_long$Gene <- 'Gene B'

# MISLEADING: Same scale
p1_low <- ggplot(low_long, aes(x = Col, y = Row, fill = Expression)) +
  geom_tile() +
  scale_fill_gradient(low = 'white', high = 'red',
                      limits = c(0, 1000),
                      name = 'Expression') +
  labs(title = 'Gene A (Low Range)',
       subtitle = '❌ MISLEADING: Same scale') +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5, face = 'bold', size = 11),
        plot.subtitle = element_text(hjust = 0.5, color = 'red', size = 9))

p2_high <- ggplot(high_long, aes(x = Col, y = Row, fill = Expression)) +
  geom_tile() +
  scale_fill_gradient(low = 'white', high = 'red',
                      limits = c(0, 1000),
                      name = 'Expression') +
  labs(title = 'Gene B (High Range)',
       subtitle = '(Gene A looks uniformly low)') +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5, face = 'bold', size = 11),
        plot.subtitle = element_text(hjust = 0.5, color = 'red', size = 9, face = 'italic'))

# CORRECT: Independent scales
p3_low <- ggplot(low_long, aes(x = Col, y = Row, fill = Expression)) +
  geom_tile() +
  scale_fill_gradient(low = 'white', high = 'red',
                      limits = c(1, 10),
                      name = 'Expression\n(1-10 FPKM)') +
  labs(title = 'Gene A (Independent Scale)',
       subtitle = '✓ CORRECT: Scale optimized') +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5, face = 'bold', size = 11),
        plot.subtitle = element_text(hjust = 0.5, color = 'darkgreen', size = 9))

p4_high <- ggplot(high_long, aes(x = Col, y = Row, fill = Expression)) +
  geom_tile() +
  scale_fill_gradient(low = 'white', high = 'red',
                      limits = c(100, 1000),
                      name = 'Expression\n(100-1000 FPKM)') +
  labs(title = 'Gene B (Independent Scale)',
       subtitle = '(Clearly labeled)') +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5, face = 'bold', size = 11),
        plot.subtitle = element_text(hjust = 0.5, color = 'darkgreen', size = 9, face = 'italic'))

combined <- (p1_low | p2_high) / (p3_low | p4_high)
ggsave('multiple_scales_logic.png', combined, width = 12, height = 10,
       dpi = 300, bg = 'white')

cat("✓ Multiple scales require explicit labeling\n")
cat("✓ State ranges clearly in captions and colorbars\n")
```

---

### Complete Logical Color Mapping Checklist

**Before finalizing figures with complex color schemes:**

- [ ] **If all values meet threshold** (e.g., all significant): Encode magnitude, not binary
- [ ] **Missing data** uses color OUTSIDE gradient + clear visual distinction
- [ ] **Zero point checked**: Meaningful zero → Diverging; Arbitrary zero → Sequential
- [ ] **Cyclic data** (time, angles): Use cyclic colormap (twilight, hsv)
- [ ] **Multiple scales**: Each clearly labeled or use shared scale if appropriate
- [ ] **Asymmetric scales** avoided (unless data truly asymmetric)
- [ ] **Legend explicitly states** what each color represents
- [ ] **Figure caption documents** colormap choice and scale ranges

---

### Summary: Advanced Color Logic

**The Core Principle:**
Color mapping must reflect data structure, not override it

**Common Violations:**
1. Binary coloring when all values are in same category
2. Using gradient colors for missing data
3. Diverging colormap for data without meaningful center
4. Sequential colormap for cyclic data
5. Identical colormaps for vastly different scales without labeling

**The Solution:**
- Match colormap type to data structure
- Explicitly handle edge cases (NA, all significant, etc.)
- Label everything clearly
- Test interpretation: Can reader understand without reading caption?

---

### **2.8 Color Palette Design Principles**

**Core Principle:** One figure page should contain **3 colors ideally, maximum 5 colors** (excluding specialized plots like UMAPs where categorical distinctions require more).

### **Rule 1: The 3-5 Color Maximum**

**Scientific Principle:** Human working memory effectively tracks 3-5 distinct categories simultaneously. Beyond this, cognitive load increases exponentially.

**Application:**

```python
import matplotlib.pyplot as plt
import numpy as np

# CORRECT: 3-color palette (ideal)
IDEAL_PALETTE = {
    'WT': '#7F8C8D',        # Wild-type: neutral gray
    'KO': '#E74C3C',        # Knockout: red (problem/loss)
    'Rescue': '#3498DB'     # Rescue: blue (restoration)
}

# Example: Gene expression comparison
np.random.seed(42)
conditions = ['WT', 'KO', 'Rescue']
values = [100, 45, 92]  # Relative expression
errors = [8, 6, 9]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Good example (3 colors)
ax1 = axes[0]
colors_good = [IDEAL_PALETTE[c] for c in conditions]
bars1 = ax1.bar(conditions, values, yerr=errors, capsize=8,
                color=colors_good, edgecolor='black', linewidth=2, width=0.6)
ax1.set_ylabel('Relative Expression (%)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 120)
ax1.set_title('✓ GOOD: 3 Colors\n(Clear, memorable)',
              fontsize=13, fontweight='bold', color='green')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', alpha=0.3)

# Add interpretation annotations
ax1.annotate('', xy=(0, values[0]), xytext=(1, values[1]),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax1.text(0.5, (values[0]+values[1])/2 - 10, 'Loss of\nfunction',
        ha='center', fontsize=9, style='italic')

# Panel B: Too many colors example
ax2 = axes[1]
conditions_bad = ['WT', 'KO1', 'KO2', 'Rescue1', 'Rescue2',
                  'Treatment A', 'Treatment B', 'Combination']
values_bad = np.random.randint(40, 100, len(conditions_bad))
colors_bad = plt.cm.tab10(np.linspace(0, 1, len(conditions_bad)))

bars2 = ax2.bar(range(len(conditions_bad)), values_bad,
                color=colors_bad, edgecolor='black', linewidth=1, width=0.7)
ax2.set_xticks(range(len(conditions_bad)))
ax2.set_xticklabels(conditions_bad, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Relative Expression (%)', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 120)
ax2.set_title('❌ BAD: 8 Colors\n(Overwhelming, hard to distinguish)',
              fontsize=13, fontweight='bold', color='red')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('color_rule_3to5.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ 3-5 color rule demonstrated")
```

**When you MUST use >5 colors (e.g., UMAP clustering):**

```python
# Solution: Use numbers/labels instead of relying solely on color
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

np.random.seed(42)

# Simulate UMAP with 15 clusters
n_cells = 1000
umap1 = np.random.randn(n_cells) * 3
umap2 = np.random.randn(n_cells) * 3
clusters = np.random.randint(0, 15, n_cells)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel A: Color only (hard to distinguish)
ax1 = axes[0]
scatter1 = ax1.scatter(umap1, umap2, c=clusters, cmap='tab20',
                       s=20, alpha=0.6, edgecolors='none')
ax1.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
ax1.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
ax1.set_title('❌ 15 Clusters: Color Only\n(Hard to match legend)',
              fontsize=13, fontweight='bold', color='red')
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('Cluster ID', fontsize=11, fontweight='bold')

# Panel B: Color + numbered labels (clear)
ax2 = axes[1]
scatter2 = ax2.scatter(umap1, umap2, c=clusters, cmap='tab20',
                       s=20, alpha=0.6, edgecolors='none')

# Add cluster number labels at centroids
for cluster_id in range(15):
    mask = clusters == cluster_id
    if np.sum(mask) > 0:
        centroid_x = umap1[mask].mean()
        centroid_y = umap2[mask].mean()

        # White circle background for visibility
        circle = Circle((centroid_x, centroid_y), radius=0.5,
                       facecolor='white', edgecolor='black',
                       linewidth=2, zorder=10)
        ax2.add_patch(circle)

        # Cluster number
        ax2.text(centroid_x, centroid_y, str(cluster_id),
                ha='center', va='center', fontsize=10,
                fontweight='bold', zorder=11)

ax2.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
ax2.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
ax2.set_title('✓ BETTER: Color + Numbers\n(Easy to reference)',
              fontsize=13, fontweight='bold', color='green')

plt.tight_layout()
plt.savefig('many_clusters_solution.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
```

---

### **Rule 2: Color Harmony - Avoid Deep + Light Mixing**

**Principle:** Mixing very dark and very light colors creates harsh, unbalanced contrast that looks unprofessional.

**Solution:** Keep colors within similar lightness ranges, or balance proportions intentionally.

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
import numpy as np

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# BAD Example: Deep + Light mixing (unbalanced)
ax1 = axes[0]
bad_colors = ['#1A1A1A', '#EFEFEF', '#2C3E50', '#ECF0F1']  # Very dark + very light
bad_labels = ['Very Dark', 'Very Light', 'Dark', 'Light']

for i, (c, label) in enumerate(zip(bad_colors, bad_labels)):
    rect = mpatches.Rectangle((i*2.5, 0), 2, 1, facecolor=c,
                             edgecolor='black', linewidth=2)
    ax1.add_patch(rect)

    # Calculate lightness
    rgb = mcolors.hex2color(c)
    lightness = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]

    ax1.text(i*2.5 + 1, 0.5, f'{label}\nL={lightness:.2f}',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='white' if lightness < 0.5 else 'black')

ax1.set_xlim(0, 10)
ax1.set_ylim(0, 1)
ax1.set_title('❌ BAD: Deep + Light Mixing (Harsh Contrast, L range: 0.07-0.94)',
              fontsize=13, fontweight='bold', color='red', pad=15)
ax1.axis('off')

# GOOD Example: Balanced lightness
ax2 = axes[1]
good_colors = ['#7F8C8D', '#3498DB', '#E74C3C', '#27AE60']  # Balanced
good_labels = ['Gray', 'Blue', 'Red', 'Green']

for i, (c, label) in enumerate(zip(good_colors, good_labels)):
    rect = mpatches.Rectangle((i*2.5, 0), 2, 1, facecolor=c,
                             edgecolor='black', linewidth=2)
    ax2.add_patch(rect)

    rgb = mcolors.hex2color(c)
    lightness = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]

    ax2.text(i*2.5 + 1, 0.5, f'{label}\nL={lightness:.2f}',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='white')

ax2.set_xlim(0, 10)
ax2.set_ylim(0, 1)
ax2.set_title('✓ GOOD: Balanced Lightness (L range: 0.35-0.55)',
              fontsize=13, fontweight='bold', color='green', pad=15)
ax2.axis('off')

# ACCEPTABLE: Intentional gradient (if needed)
ax3 = axes[2]
gradient_colors = ['#D6EAF8', '#85C1E9', '#3498DB', '#21618C', '#1B4F72']  # Blue gradient
gradient_labels = ['Very Light', 'Light', 'Medium', 'Dark', 'Very Dark']

for i, (c, label) in enumerate(zip(gradient_colors, gradient_labels)):
    rect = mpatches.Rectangle((i*2, 0), 1.8, 1, facecolor=c,
                             edgecolor='black', linewidth=2)
    ax3.add_patch(rect)

    rgb = mcolors.hex2color(c)
    lightness = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]

    ax3.text(i*2 + 0.9, 0.5, f'{label}\nL={lightness:.2f}',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='white' if lightness < 0.5 else 'black')

ax3.set_xlim(0, 10)
ax3.set_ylim(0, 1)
ax3.set_title('✓ ACCEPTABLE: Intentional Gradient (Single Hue Family, Smooth Transition)',
              fontsize=13, fontweight='bold', color='green', pad=15)
ax3.axis('off')

plt.tight_layout()
plt.savefig('color_balance_lightness.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
```

---

### **Rule 3: Biological Logic in Color Assignment**

**Principle:** Colors should follow scientific conventions and intuitive associations.

**Examples:**

```python
# 1. WT, KO, Rescue: Color proximity reflects biological relationship
BIOLOGICAL_LOGIC = {
    'WT': '#7F8C8D',      # Wild-type: neutral gray (baseline)
    'Rescue': '#5D6D7E',  # Rescue: similar gray (close to WT) ← KEY POINT
    'KO': '#E74C3C'       # Knockout: red (different/problem)
}
# → WT and Rescue should have SIMILAR colors (both are functional states)

# 2. Disease severity: Darker = more severe (intuitive)
DISEASE_SEVERITY = {
    'Healthy': '#D5F4E6',      # Very light green
    'Mild': '#82E0AA',         # Light green
    'Moderate': '#F39C12',     # Orange (warning)
    'Severe': '#E67E22',       # Dark orange
    'Critical': '#C0392B'      # Dark red (danger)
}

# 3. Traffic light convention: Red=stop/problem, Green=go/healthy
INTUITIVE_ASSOCIATIONS = {
    'Control': '#27AE60',      # Green = healthy/baseline
    'Disease': '#E74C3C',      # Red = problem
    'Treatment': '#3498DB'     # Blue = intervention
}

# 4. Temperature: Blue=cold, Red=hot
TEMPERATURE_SCALE = {
    'cold': '#3498DB',
    'warm': '#F39C12',
    'hot': '#E74C3C'
}
```

**Code Example: WT-KO-Rescue Color Logic**

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# BAD: KO and Rescue have similar colors (confusing relationship)
ax1 = axes[0]
conditions_bad = ['WT', 'KO', 'Rescue']
values_bad = [100, 45, 95]
colors_bad = ['#7F8C8D', '#E74C3C', '#C0392B']  # KO and Rescue both reddish!

bars1 = ax1.bar(conditions_bad, values_bad, color=colors_bad,
               edgecolor='black', linewidth=2, width=0.6)
ax1.set_ylabel('Function (%)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 120)
ax1.set_title('❌ BAD: KO and Rescue Look Similar\n(Implies they are related states)',
              fontsize=13, fontweight='bold', color='red')
ax1.axhline(100, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax1.text(2.5, 102, 'WT baseline', fontsize=9, style='italic')

# GOOD: WT and Rescue similar (both functional), KO different
ax2 = axes[1]
conditions_good = ['WT', 'KO', 'Rescue']
values_good = [100, 45, 95]
colors_good = ['#7F8C8D', '#E74C3C', '#5D6D7E']  # WT and Rescue similar grays!

bars2 = ax2.bar(conditions_good, values_good, color=colors_good,
               edgecolor='black', linewidth=2, width=0.6)
ax2.set_ylabel('Function (%)', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 120)
ax2.set_title('✓ GOOD: WT and Rescue Similar Colors\n(Correctly shows functional relationship)',
              fontsize=13, fontweight='bold', color='green')
ax2.axhline(100, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Add annotations showing color logic
ax2.annotate('', xy=(0, 110), xytext=(2, 110),
            arrowprops=dict(arrowstyle='<->', color='green', lw=3))
ax2.text(1, 113, 'Similar colors =\nSimilar function', ha='center',
        fontsize=9, fontweight='bold', color='green',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

ax2.annotate('', xy=(1, -8), xytext=(1, -8),
            arrowprops=dict(arrowstyle='-', color='red', lw=0))
ax2.text(1, -12, 'Different color =\nLoss of function', ha='center',
        fontsize=9, fontweight='bold', color='red',
        bbox=dict(boxstyle='round', facecolor='#FFCCCC', alpha=0.7))

plt.tight_layout()
plt.savefig('wt_ko_rescue_color_logic.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
```

---

### **Rule 4: NA/Missing Data Color Strategy**

**Principle:** NA colors must be distinguishable and EXCLUDED from the main color scheme.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)

# Simulate data with missing values
data = {
    'Gene': [f'Gene{i}' for i in range(1, 26)],
    'Sample1': np.random.randn(25),
    'Sample2': np.random.randn(25),
    'Sample3': np.random.randn(25),
    'Sample4': np.random.randn(25)
}

df = pd.DataFrame(data).set_index('Gene')

# Introduce NA values
df.iloc[3:6, 1] = np.nan
df.iloc[10:12, 2] = np.nan
df.iloc[18, 3] = np.nan

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# BAD: NA color within main palette (confusing)
ax1 = axes[0]
data_for_plot = df.fillna(-999)  # Sentinel value
im1 = ax1.imshow(data_for_plot, cmap='RdBu_r', aspect='auto',
                 vmin=-2, vmax=2)
ax1.set_title('❌ BAD: NA Blends with Data\n(Can\'t distinguish missing values)',
              fontsize=13, fontweight='bold', color='red')
ax1.set_xlabel('Samples', fontsize=11, fontweight='bold')
ax1.set_ylabel('Genes', fontsize=11, fontweight='bold')
ax1.set_xticks(range(4))
ax1.set_xticklabels(['Sample1', 'Sample2', 'Sample3', 'Sample4'], rotation=45, ha='right')
plt.colorbar(im1, ax=ax1, label='Expression (Z-score)')

# GOOD: NA in distinct color (crosshatch or gray)
ax2 = axes[1]

# Create masked array
data_masked = np.ma.masked_where(np.isnan(df.values), df.values)

im2 = ax2.imshow(data_masked, cmap='RdBu_r', aspect='auto',
                 vmin=-2, vmax=2)

# Add gray rectangles for NA values
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        if np.isnan(df.values[i, j]):
            ax2.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                       facecolor='#D3D3D3',
                                       edgecolor='black',
                                       linewidth=1.5,
                                       hatch='///',
                                       fill=True))

ax2.set_title('✓ GOOD: NA in Distinct Color\n(Crosshatch pattern, separate from scale)',
              fontsize=13, fontweight='bold', color='green')
ax2.set_xlabel('Samples', fontsize=11, fontweight='bold')
ax2.set_ylabel('Genes', fontsize=11, fontweight='bold')
ax2.set_xticks(range(4))
ax2.set_xticklabels(['Sample1', 'Sample2', 'Sample3', 'Sample4'], rotation=45, ha='right')

cbar2 = plt.colorbar(im2, ax=ax2, label='Expression (Z-score)')

# Add NA legend
from matplotlib.patches import Patch
na_patch = Patch(facecolor='#D3D3D3', edgecolor='black',
                hatch='///', label='NA / Missing')
ax2.legend(handles=[na_patch], loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig('na_color_strategy.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
```

**R Equivalent:**

```r
library(ggplot2)
library(tidyr)
library(dplyr)

# Create data with NAs
set.seed(42)
data <- expand.grid(
  Gene = paste0('Gene', 1:25),
  Sample = paste0('Sample', 1:4)
) %>%
  mutate(Expression = rnorm(n()))

# Introduce NAs
data$Expression[30:35] <- NA
data$Expression[60:62] <- NA

# Plot with distinct NA handling
ggplot(data, aes(x = Sample, y = Gene, fill = Expression)) +
  geom_tile(color = 'white', size = 0.5) +

  # Main color scale (for non-NA values)
  scale_fill_gradient2(low = '#3498DB', mid = 'white', high = '#E74C3C',
                      midpoint = 0, na.value = '#D3D3D3',  # Gray for NA
                      name = 'Expression\n(Z-score)') +

  # Add pattern to NA tiles (requires ggpattern package)
  # library(ggpattern)
  # geom_tile_pattern(data = filter(data, is.na(Expression)),
  #                   pattern = 'crosshatch', pattern_density = 0.5,
  #                   fill = '#D3D3D3', color = 'black', size = 1) +

  labs(title = '✓ NA Values in Distinct Gray (Separate from Scale)',
       x = 'Samples', y = 'Genes') +

  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = 'bold', size = 13, hjust = 0.5, color = 'darkgreen'),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = 'right'
  )

ggsave('na_color_strategy_r.png', width = 10, height = 8, dpi = 300)
```

---

### **Rule 5: Sequential Colors - Single Hue Families**

**When to use:** Continuous data (e.g., expression levels, concentrations, counts)

**Principle:** Use gradations of a **single hue** to show magnitude, not rainbow colors.

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

np.random.seed(42)

# Simulate continuous data (e.g., gene expression)
data = np.random.rand(10, 10) * 100

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# BAD: Rainbow (not perceptually uniform, no clear magnitude)
ax1 = axes[0]
im1 = ax1.imshow(data, cmap='jet', aspect='auto')
ax1.set_title('❌ BAD: Rainbow\n(Unclear magnitude progression)',
              fontsize=13, fontweight='bold', color='red')
plt.colorbar(im1, ax=ax1, label='Expression')

# GOOD: Single hue gradient (clear magnitude)
ax2 = axes[1]
im2 = ax2.imshow(data, cmap='Blues', aspect='auto')
ax2.set_title('✓ GOOD: Single Hue (Blue)\n(Clear: Light → Dark = Low → High)',
              fontsize=13, fontweight='bold', color='green')
plt.colorbar(im2, ax=ax2, label='Expression')

# ALSO GOOD: Custom single-hue gradient
ax3 = axes[2]
colors_custom = ['#FFFFFF', '#EBF5FB', '#D6EAF8', '#AED6F1',
                '#85C1E9', '#5DADE2', '#3498DB', '#2E86C1',
                '#2874A6', '#21618C']
cmap_custom = LinearSegmentedColormap.from_list('custom_blues', colors_custom)
im3 = ax3.imshow(data, cmap=cmap_custom, aspect='auto')
ax3.set_title('✓ ALSO GOOD: Custom Blue Gradient\n(Smooth progression)',
              fontsize=13, fontweight='bold', color='green')
plt.colorbar(im3, ax=ax3, label='Expression')

plt.tight_layout()
plt.savefig('sequential_color_single_hue.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
```

## **2.9 Shape Selection and Usage (NEW SECTION)**

### **Rule: Shape Must Provide Clear Contrast**

**Principle:** Regular vs. irregular shapes, filled vs. open - use contrast to distinguish groups.

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Simulate data for different conditions
conditions = ['Control', 'Treatment A', 'Treatment B', 'Treatment C']
n_points = 30

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# BAD: Similar shapes (hard to distinguish)
ax1 = axes[0]
shapes_bad = ['o', 's', '^', 'v']  # All filled, similar sizes
for i, (cond, marker) in enumerate(zip(conditions, shapes_bad)):
    x = np.random.randn(n_points) + i*2
    y = np.random.randn(n_points)
    ax1.scatter(x, y, s=80, marker=marker, color='#3498DB',
               alpha=0.6, edgecolors='black', linewidths=1,
               label=cond)

ax1.set_xlabel('Variable X', fontsize=12, fontweight='bold')
ax1.set_ylabel('Variable Y', fontsize=12, fontweight='bold')
ax1.set_title('❌ BAD: Subtle Shape Differences\n(Hard to distinguish quickly)',
              fontsize=13, fontweight='bold', color='red')
ax1.legend(loc='upper left', frameon=True, fontsize=10)
ax1.grid(alpha=0.3)

# GOOD: Contrasting shapes (easy to distinguish)
ax2 = axes[1]
shapes_good = ['o', 's', '^', 'D']  # Mix of regular shapes
fills = [True, False, True, False]  # Alternate filled/open
colors_contrast = ['#3498DB', '#3498DB', '#E74C3C', '#E74C3C']

for i, (cond, marker, filled, color) in enumerate(zip(conditions, shapes_good, fills, colors_contrast)):
    x = np.random.randn(n_points) + i*2
    y = np.random.randn(n_points)

    if filled:
        ax2.scatter(x, y, s=100, marker=marker, color=color,
                   alpha=0.7, edgecolors='black', linewidths=1.5,
                   label=cond)
    else:
        ax2.scatter(x, y, s=100, marker=marker, facecolors='none',
                   edgecolors=color, linewidths=2,
                   label=cond)

ax2.set_xlabel('Variable X', fontsize=12, fontweight='bold')
ax2.set_ylabel('Variable Y', fontsize=12, fontweight='bold')
ax2.set_title('✓ GOOD: High Contrast Shapes\n(Filled/Open + Color + Shape)',
              fontsize=13, fontweight='bold', color='green')
ax2.legend(loc='upper left', frameon=True, fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('shape_contrast.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
```

**Shape Palette Recommendations:**

```python
# Accessible shape combinations (easily distinguishable)
SHAPE_PALETTE_GOOD = {
    'Group1': {'marker': 'o', 'fill': True, 'size': 80},    # Filled circle
    'Group2': {'marker': 's', 'fill': False, 'size': 80},   # Open square
    'Group3': {'marker': '^', 'fill': True, 'size': 100},   # Filled triangle
    'Group4': {'marker': 'D', 'fill': False, 'size': 70}    # Open diamond
}

# Bad: Too subtle differences
SHAPE_PALETTE_BAD = {
    'Group1': {'marker': 'o', 'fill': True, 'size': 80},    # Filled circle
    'Group2': {'marker': 'o', 'fill': True, 'size': 90},    # Slightly larger circle (!)
    'Group3': {'marker': 'o', 'fill': True, 'size': 70},    # Slightly smaller circle (!)
    'Group4': {'marker': 's', 'fill': True, 'size': 80}     # Square (finally different)
}
```
---

**Chapter 2 Complete Summary:**

You now understand:
- **Color theory** (RGB, HSV, CIELAB) and perceptual uniformity
- **Palette types** (sequential, diverging, qualitative) matched to data types
- **Saturation logic** (deep=important/significant, light=background)
- **Restraint** (3-color maximum) and **consistency** (same concept=same color)
- **Field conventions** and when/how to deviate
- **Accessibility** (colorblind-safe, redundant encoding, testing)
- **Integrity** (no manipulation, symmetric scales, transparency)
- **Logical edge cases** (all significant, missing data, meaningful zero, cyclic data, multiple scales)

---
