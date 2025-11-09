# Chapter 3: Typography, Annotation & Labels

## 3.1 The Functional Role of Text in Scientific Figures

### Text is Not Decoration—It's Data Infrastructure

In scientific figures, **every text element serves a specific functional purpose**. Unlike artistic or marketing graphics where text might be styled for aesthetic impact, scientific text must prioritize:

1. **Clarity**: Instant comprehension without ambiguity
2. **Hierarchy**: Guide reader to information in logical order
3. **Precision**: Accurate communication of units, values, and relationships
4. **Accessibility**: Readable across viewing conditions (screen, print, projection)
5. **Consistency**: Predictable patterns reduce cognitive load

**Text Elements in a Complete Figure:**

```
Essential text components:
├─ Title/Caption (what is being shown)
├─ Axis labels (what dimensions represent)
├─ Axis tick labels (scale values)
├─ Legend (what colors/symbols mean)
├─ Data labels (optional: direct value annotation)
├─ Statistical annotations (significance markers, p-values)
└─ Panel labels (A, B, C for multi-panel figures)
```

---

### The Hierarchy of Text Importance

**Level 1: Critical (Must Read)**
- Axis labels with units
- Legend identifying groups
- Panel labels (A, B, C)
→ **Bold, 10-12pt, high contrast**

**Level 2: Important (Should Read)**
- Axis tick values
- Figure title/caption number
- Statistical significance markers
→ **Regular weight, 9-10pt**

**Level 3: Supporting (May Read)**
- Secondary annotations
- Supplementary notes
→ **Regular weight, 8-9pt, possibly lighter gray**

**Code Example (Python) - Text Hierarchy:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data
np.random.seed(42)
categories = ['Control', 'Treatment A', 'Treatment B']
values = [25, 32, 28]
errors = [3, 4, 3.5]

fig, ax = plt.subplots(figsize=(8, 6))

# Data
bars = ax.bar(categories, values, color=['#7F8C8D', '#3498DB', '#E74C3C'],
              edgecolor='black', linewidth=1.5, width=0.6)
ax.errorbar(categories, values, yerr=errors, fmt='none',
            ecolor='black', capsize=8, linewidth=2)

# LEVEL 1: Critical - Axis labels (bold, large)
ax.set_xlabel('Treatment Group', fontsize=12, fontweight='bold', color='#000000')
ax.set_ylabel('Cell Viability (%)', fontsize=12, fontweight='bold', color='#000000')

# LEVEL 2: Important - Tick labels (regular, medium)
ax.tick_params(axis='both', labelsize=10, colors='#333333')

# Add statistical annotation (Level 2)
ax.text(1, 35, '***', ha='center', va='bottom',
        fontsize=14, fontweight='bold', color='#000000')
ax.plot([0, 1], [34, 34], 'k-', linewidth=1.5)

# LEVEL 3: Supporting - Explanatory note (smaller, lighter)
ax.text(0.02, 0.98, '*** p < 0.001', transform=ax.transAxes,
        fontsize=8, va='top', ha='left', color='#666666', style='italic')

# Title (Level 1)
ax.set_title('Effect of Treatment on Cell Viability',
             fontsize=13, fontweight='bold', pad=15)

ax.set_ylim(0, 45)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('text_hierarchy_example.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R) - Text Hierarchy:**

```r
library(ggplot2)

# Data
data <- data.frame(
  group = factor(c('Control', 'Treatment A', 'Treatment B'),
                 levels = c('Control', 'Treatment A', 'Treatment B')),
  value = c(25, 32, 28),
  error = c(3, 4, 3.5)
)

colors <- c('Control' = '#7F8C8D', 'Treatment A' = '#3498DB', 'Treatment B' = '#E74C3C')

p <- ggplot(data, aes(x = group, y = value, fill = group)) +
  geom_bar(stat = 'identity', color = 'black', size = 1, width = 0.6) +
  geom_errorbar(aes(ymin = value - error, ymax = value + error),
                width = 0.25, size = 1) +
  scale_fill_manual(values = colors) +

  # LEVEL 1: Critical - Axis labels (bold, large)
  labs(x = 'Treatment Group',
       y = 'Cell Viability (%)',
       title = 'Effect of Treatment on Cell Viability') +

  # Statistical annotation (Level 2)
  annotate('segment', x = 1, xend = 2, y = 34, yend = 34, size = 1) +
  annotate('text', x = 1.5, y = 35, label = '***',
           size = 5, fontface = 'bold') +

  # LEVEL 3: Supporting note
  annotate('text', x = 0.6, y = 43, label = '*** p < 0.001',
           size = 2.8, hjust = 0, fontface = 'italic', color = '#666666') +

  ylim(0, 45) +

  theme_classic(base_size = 11) +
  theme(
    # Level 1: Critical text
    axis.title = element_text(face = 'bold', size = 12, color = '#000000'),
    plot.title = element_text(hjust = 0.5, face = 'bold', size = 13),

    # Level 2: Important
    axis.text = element_text(size = 10, color = '#333333'),

    legend.position = 'none',
    panel.grid.major.y = element_line(color = 'gray90', size = 0.3)
  )

ggsave('text_hierarchy_example.png', p, width = 8, height = 6,
       dpi = 300, bg = 'white')
```

---

## 3.2 Font Selection for Scientific Figures

### The Golden Rule: Sans-Serif for Figures, Serif for Text

**Why sans-serif for figures:**
- Better legibility at small sizes
- Cleaner appearance when scaled/compressed
- Works better with data (numbers, symbols)
- Maintains clarity in digital and print

**Recommended Font Families:**

**Tier 1: Universally Available, Publication-Ready**
```
Arial: Clean, neutral, universally supported
Helvetica: Professional standard (Mac default, expensive license)
Calibri: Modern, readable (Microsoft Office default)
```

**Tier 2: Enhanced Readability**
```
Roboto: Google's open-source, excellent screen rendering
Open Sans: Friendly, highly legible, free
Source Sans Pro: Adobe's open-source, designed for UI
```

**Tier 3: Specialized Scientific**
```
CMU Sans (Computer Modern): Matches LaTeX documents
DejaVu Sans: Unicode support, open-source
Liberation Sans: Metric-compatible with Arial (open-source)
```

**Fonts to AVOID in Scientific Figures:**
```
❌ Comic Sans: Unprofessional
❌ Papyrus: Decorative, illegible at small sizes
❌ Brush Script: Artistic, not data-appropriate
❌ Elaborate serifs (Times, Garamond) in graphs: Cluttered at small sizes
```

**Code Example (Python) - Font Consistency:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Set global font parameters (do this ONCE at start of script)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],  # Fallback order
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14
})

# Sample figure
np.random.seed(42)
x = np.linspace(0, 10, 50)
y = 2*x + np.random.randn(50)*2

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(x, y, color='#3498DB', s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
ax.plot(x, 2*x, 'r--', linewidth=2, label='y = 2x')

ax.set_xlabel('Independent Variable (units)')
ax.set_ylabel('Dependent Variable (units)')
ax.set_title('Consistent Sans-Serif Typography')
ax.legend(loc='upper left', frameon=True)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('font_consistency.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Font family: Arial/Helvetica")
print("✓ Consistent sizing across all text elements")
```

**Code Example (R) - Font Consistency:**

```r
library(ggplot2)

# Sample data
set.seed(42)
data <- data.frame(
  x = seq(0, 10, length.out = 50),
  y = 2 * seq(0, 10, length.out = 50) + rnorm(50, 0, 2)
)

p <- ggplot(data, aes(x = x, y = y)) +
  geom_point(color = '#3498DB', size = 3, alpha = 0.7) +
  geom_abline(intercept = 0, slope = 2, color = 'red', linetype = 'dashed', size = 1) +

  labs(x = 'Independent Variable (units)',
       y = 'Dependent Variable (units)',
       title = 'Consistent Sans-Serif Typography') +

  # Specify font family (will fall back to system default if not available)
  theme_classic(base_size = 10, base_family = 'Arial') +
  theme(
    axis.title = element_text(size = 11, face = 'bold'),
    plot.title = element_text(hjust = 0.5, size = 12, face = 'bold'),
    axis.text = element_text(size = 9),
    panel.grid.major = element_line(color = 'gray90', size = 0.3)
  )

ggsave('font_consistency.png', p, width = 7, height = 5,
       dpi = 300, bg = 'white')

cat("✓ Font family: Arial\n")
cat("✓ Consistent sizing across all text elements\n")
```

---

## 3.3 Axis Labels: Units and Clarity

### The Non-Negotiable Rule: Always Include Units

**Format: "Variable Name (units)"**

```
✓ CORRECT:
  "Temperature (°C)"
  "Time (hours)"
  "Concentration (μM)"
  "Expression Level (FPKM)"
  "Velocity (m/s)"

❌ WRONG:
  "Temperature" (What scale? Celsius? Kelvin? Fahrenheit?)
  "Time" (Seconds? Minutes? Hours? Days?)
  "Expression" (Arbitrary units? Fold change? FPKM?)
```

### Special Cases: Dimensionless and Normalized Data

**When data has no units:**

```
Acceptable labels for dimensionless quantities:
✓ "Correlation Coefficient (r)" → range -1 to 1, dimensionless
✓ "Probability (p)" → range 0 to 1, dimensionless
✓ "Fold Change (log₂)" → ratio, logarithm specified
✓ "Normalized Expression (AU)" → AU = Arbitrary Units (acknowledges normalization)
✓ "Relative Abundance (%)" → percentage is the unit
```

**When you've normalized but original units existed:**

```
✓ GOOD: "Normalized Expression (% of maximum)"
✓ GOOD: "Fluorescence Intensity (AU)" + note in methods about normalization
✓ ACCEPTABLE: "Response (normalized)" + full description in caption

❌ BAD: "Expression" with no indication of normalization
```

### Logical Consistency: Axes Must Match Data

**Common Error: Mismatch between label and actual data**

```
❌ INCONSISTENT:
Label: "Temperature (°C)"
Data range: 273-373
→ These are Kelvin values!

✓ FIX:
Label: "Temperature (K)" OR convert data to Celsius

---

❌ INCONSISTENT:
Label: "Time (hours)"
Tick marks: 0, 60, 120, 180, 240
→ These are minutes!

✓ FIX:
Label: "Time (minutes)" OR convert to hours (0, 1, 2, 3, 4)
```

---


## 3.4 Panel Labels and Multi-Figure Organization

### The Standard Convention: A, B, C, D...

When figures contain multiple panels, **panel labels** provide critical navigational structure that allows the main text to reference specific subfigures precisely.

**Standard Format:**
```
✓ STANDARD: Bold, uppercase letters (A, B, C, D...)
✓ Position: Top-left corner of each panel (outside or just inside)
✓ Size: Slightly larger than axis labels (13-14pt when axis labels are 11pt)
✓ Color: Black or very dark gray
```

**Why This Matters:**

```
In manuscript text:
"Treatment A showed significant improvement (Figure 2B),
while histological analysis revealed reduced inflammation (Figure 2C)."

Without clear panel labels:
"Treatment A showed improvement (see second panel in Figure 2?)"
→ Ambiguous, unprofessional
```

---

### Placement Options

**Option 1: Outside panel (most common in publications)**
```
Advantages:
✓ Doesn't obscure data
✓ Clear visual separation
✓ Consistent position across panels

Disadvantages:
✗ Requires extra space in figure layout
```

**Option 2: Inside panel, top-left corner**
```
Advantages:
✓ Space-efficient
✓ Works well when panels have white space in corner

Disadvantages:
✗ Can obscure data if corner contains information
✗ May need background box for visibility
```

**Code Example (Python) - Panel Labels:**

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Sample data for each panel
for idx, ax in enumerate(axes.flat):
    # Different plot type for each panel
    if idx == 0:
        # Scatter plot
        x = np.random.randn(50)
        y = 2*x + np.random.randn(50)
        ax.scatter(x, y, color='#3498DB', s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
        ax.set_xlabel('Variable X (units)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Variable Y (units)', fontsize=11, fontweight='bold')
        ax.set_title('Correlation Analysis', fontsize=12, fontweight='bold')

    elif idx == 1:
        # Bar chart
        categories = ['Group 1', 'Group 2', 'Group 3']
        values = [25, 32, 28]
        ax.bar(categories, values, color=['#7F8C8D', '#3498DB', '#E74C3C'],
              edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Response (AU)', fontsize=11, fontweight='bold')
        ax.set_title('Group Comparison', fontsize=12, fontweight='bold')

    elif idx == 2:
        # Line plot
        time = np.linspace(0, 10, 50)
        signal = np.sin(time) + np.random.randn(50)*0.1
        ax.plot(time, signal, color='#27AE60', linewidth=2.5)
        ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Signal (mV)', fontsize=11, fontweight='bold')
        ax.set_title('Temporal Dynamics', fontsize=12, fontweight='bold')

    else:
        # Heatmap
        data = np.random.randn(10, 10)
        im = ax.imshow(data, cmap='RdBu_r', aspect='auto')
        ax.set_xlabel('Sample', fontsize=11, fontweight='bold')
        ax.set_ylabel('Gene', fontsize=11, fontweight='bold')
        ax.set_title('Expression Pattern', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # PANEL LABEL: Outside, top-left
    panel_label = chr(65 + idx)  # A, B, C, D
    ax.text(-0.15, 1.05, panel_label,
           transform=ax.transAxes,  # Coordinates relative to panel
           fontsize=16, fontweight='bold',
           va='top', ha='right',
           color='black')

    # Style
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('panel_labels_example.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Panel labels: A, B, C, D (bold, outside panels)")
```

**Code Example (R) - Panel Labels:**

```r
library(ggplot2)
library(patchwork)

set.seed(42)

# Panel A: Scatter plot
p_a <- ggplot(data.frame(x = rnorm(50), y = 2*rnorm(50) + rnorm(50)),
             aes(x = x, y = y)) +
  geom_point(color = '#3498DB', size = 3, alpha = 0.7) +
  labs(x = 'Variable X (units)', y = 'Variable Y (units)',
       title = 'Correlation Analysis') +
  theme_classic(base_size = 11) +
  theme(
    axis.title = element_text(face = 'bold', size = 11),
    plot.title = element_text(hjust = 0.5, face = 'bold', size = 12),
    panel.grid.major = element_line(color = 'gray90', size = 0.3)
  )

# Panel B: Bar chart
p_b <- ggplot(data.frame(group = c('Group 1', 'Group 2', 'Group 3'),
                        value = c(25, 32, 28)),
             aes(x = group, y = value, fill = group)) +
  geom_bar(stat = 'identity', color = 'black', size = 1) +
  scale_fill_manual(values = c('#7F8C8D', '#3498DB', '#E74C3C')) +
  labs(y = 'Response (AU)', title = 'Group Comparison') +
  theme_classic(base_size = 11) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_text(face = 'bold', size = 11),
    plot.title = element_text(hjust = 0.5, face = 'bold', size = 12),
    legend.position = 'none',
    panel.grid.major.y = element_line(color = 'gray90', size = 0.3)
  )

# Panel C: Line plot
p_c <- ggplot(data.frame(time = seq(0, 10, length.out = 50),
                        signal = sin(seq(0, 10, length.out = 50)) + rnorm(50, 0, 0.1)),
             aes(x = time, y = signal)) +
  geom_line(color = '#27AE60', size = 1.5) +
  labs(x = 'Time (s)', y = 'Signal (mV)',
       title = 'Temporal Dynamics') +
  theme_classic(base_size = 11) +
  theme(
    axis.title = element_text(face = 'bold', size = 11),
    plot.title = element_text(hjust = 0.5, face = 'bold', size = 12),
    panel.grid.major = element_line(color = 'gray90', size = 0.3)
  )

# Panel D: Heatmap
library(reshape2)
heatmap_data <- melt(matrix(rnorm(100), 10, 10))
p_d <- ggplot(heatmap_data, aes(x = Var2, y = Var1, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = '#2166AC', mid = 'white', high = '#B2182B',
                       midpoint = 0, name = 'Value') +
  labs(x = 'Sample', y = 'Gene', title = 'Expression Pattern') +
  theme_minimal(base_size = 11) +
  theme(
    axis.title = element_text(face = 'bold', size = 11),
    plot.title = element_text(hjust = 0.5, face = 'bold', size = 12),
    panel.grid = element_blank()
  )

# Combine with panel labels
combined <- (p_a + p_b) / (p_c + p_d) +
  plot_annotation(tag_levels = 'A') +  # Automatic A, B, C, D labeling
  plot_layout(guides = 'collect') &
  theme(
    plot.tag = element_text(size = 16, face = 'bold'),
    plot.tag.position = c(0, 1)  # Top-left
  )

ggsave('panel_labels_example.png', combined, width = 12, height = 10,
       dpi = 300, bg = 'white')

cat("✓ Panel labels: A, B, C, D (bold, outside panels)\n")
```

---

### Logical Panel Organization

**Principle: Arrange panels to support narrative flow**

**Reading Order Conventions:**
```
Western convention (most journals):
└─ Left-to-right, top-to-bottom (like text)
   A  B
   C  D

Some journals accept:
└─ Top-to-bottom, left-to-right (column-wise)
   A  C
   B  D

✓ State clearly in caption if non-standard order
```

**Logical Grouping Strategies:**

**Strategy 1: Temporal Progression**
```
Panel A: Baseline (time 0)
Panel B: Early response (6 hours)
Panel C: Late response (24 hours)
Panel D: Recovery (48 hours)

→ Left-to-right naturally represents time flow
```

**Strategy 2: Methodological Hierarchy**
```
Panel A: Raw data (microscopy image)
Panel B: Processed data (segmented/quantified)
Panel C: Summary statistics (bar chart)
Panel D: Model fitting (correlation plot)

→ Top-to-bottom represents analysis pipeline
```

**Strategy 3: Comparative Structure**
```
      Control    Treatment
A, B: [Image]    [Image]
C, D: [Quant]    [Quant]

→ Columns = conditions, Rows = data types
→ Facilitates direct comparison
```

---

## 3.5 Statistical Annotations: Significance Markers

### The Standard Notation System

**Asterisk convention (most common):**
```
*     p < 0.05  (significant)
**    p < 0.01  (highly significant)
***   p < 0.001 (very highly significant)
n.s.  p ≥ 0.05  (not significant)
```

**When to Use:**
- Quick visual indication of significance
- Comparing multiple groups
- Space constraints prevent detailed statistics

**When NOT to Use:**
- Exact p-values are critical (report actual values)
- Only one comparison (just state p-value)
- Effect sizes matter more than p-values

---

### Proper Placement and Formatting

**Code Example (Python) - Statistical Annotations:**

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Sample data: comparing 4 groups
groups = ['Control', 'Drug A', 'Drug B', 'Drug C']
values = [25, 32, 28, 35]
errors = [3, 3.5, 3, 4]

# P-values for comparisons (vs. control)
p_values = [None, 0.008, 0.15, 0.0003]  # None for control (no self-comparison)

fig, ax = plt.subplots(figsize=(9, 6))

# Colors: gray for control, blue/red for treatments
colors = ['#7F8C8D', '#3498DB', '#3498DB', '#E74C3C']

bars = ax.bar(groups, values, color=colors,
              edgecolor='black', linewidth=1.5, width=0.6)
ax.errorbar(groups, values, yerr=errors, fmt='none',
           ecolor='black', capsize=8, linewidth=2)

# Add significance annotations
for i, (group, value, error, p) in enumerate(zip(groups, values, errors, p_values)):
    if p is not None:
        # Determine significance marker
        if p < 0.001:
            marker = '***'
        elif p < 0.01:
            marker = '**'
        elif p < 0.05:
            marker = '*'
        else:
            marker = 'n.s.'

        # Position above error bar
        y_pos = value + error + 1.5

        # Bracket connecting to control
        bracket_y = value + error + 0.5
        ax.plot([0, i], [bracket_y, bracket_y], 'k-', linewidth=1.2)
        ax.plot([0, 0], [bracket_y - 0.3, bracket_y], 'k-', linewidth=1.2)
        ax.plot([i, i], [bracket_y - 0.3, bracket_y], 'k-', linewidth=1.2)

        # Marker
        ax.text(i/2, y_pos, marker, ha='center', va='bottom',
               fontsize=12, fontweight='bold')

# Labels
ax.set_ylabel('Response (AU)', fontsize=12, fontweight='bold')
ax.set_title('Statistical Annotations: Comparison to Control',
            fontsize=13, fontweight='bold', pad=15)

# Legend for significance
legend_text = '* p < 0.05\n** p < 0.01\n*** p < 0.001\nn.s. = not significant'
ax.text(0.98, 0.98, legend_text, transform=ax.transAxes,
       fontsize=9, va='top', ha='right',
       bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

ax.set_ylim(0, 50)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('statistical_annotations.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R) - Statistical Annotations:**

```r
library(ggplot2)
library(ggsignif)

# Data
data <- data.frame(
  group = factor(c('Control', 'Drug A', 'Drug B', 'Drug C'),
                levels = c('Control', 'Drug A', 'Drug B', 'Drug C')),
  value = c(25, 32, 28, 35),
  error = c(3, 3.5, 3, 4)
)

colors <- c('Control' = '#7F8C8D', 'Drug A' = '#3498DB',
            'Drug B' = '#3498DB', 'Drug C' = '#E74C3C')

# P-values
comparisons <- list(
  c('Control', 'Drug A'),
  c('Control', 'Drug B'),
  c('Control', 'Drug C')
)

p_values <- c(0.008, 0.15, 0.0003)

# Convert to annotation format
annotations <- sapply(p_values, function(p) {
  if (p < 0.001) '***'
  else if (p < 0.01) '**'
  else if (p < 0.05) '*'
  else 'n.s.'
})

p <- ggplot(data, aes(x = group, y = value, fill = group)) +
  geom_bar(stat = 'identity', color = 'black', size = 1, width = 0.6) +
  geom_errorbar(aes(ymin = value - error, ymax = value + error),
               width = 0.25, size = 1) +
  scale_fill_manual(values = colors) +

  # Add significance brackets
  geom_signif(comparisons = comparisons,
              annotations = annotations,
              y_position = c(36, 33, 40),
              tip_length = 0.02,
              textsize = 4,
              fontface = 'bold') +

  labs(y = 'Response (AU)',
       title = 'Statistical Annotations: Comparison to Control') +

  # Legend for markers
  annotate('text', x = 3.7, y = 48,
           label = '* p < 0.05\n** p < 0.01\n*** p < 0.001\nn.s. = not significant',
           hjust = 0, vjust = 1, size = 3,
           lineheight = 0.9) +

  ylim(0, 50) +

  theme_classic(base_size = 12) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_text(face = 'bold', size = 12),
    plot.title = element_text(hjust = 0.5, face = 'bold', size = 13),
    legend.position = 'none',
    panel.grid.major.y = element_line(color = 'gray90', size = 0.3)
  )

ggsave('statistical_annotations.png', p, width = 9, height = 6,
       dpi = 300, bg = 'white')
```

---

### Reporting Exact P-Values

**When exact values matter:**

```
✓ GOOD: Include in caption or on figure
"Drug A vs. Control: p = 0.008"
"Drug C vs. Control: p = 0.0003"

✓ BETTER: Combine visual markers with exact values
Markers on figure (**)
Caption: "** p = 0.008, *** p = 0.0003 (unpaired t-test)"
```

**Journal requirements vary:**
- Nature: Often prefers exact p-values in text/caption
- Cell: Accepts asterisk notation with key
- Science: Varies by article type

**Check target journal guidelines!**

---

## 3.6 Direct Labeling vs. Legends

### The Usability Principle: Minimize Eye Travel

**Legend disadvantages:**
- Requires back-and-forth eye movement
- Matching colors to categories requires working memory
- Easy to misinterpret if legend order doesn't match visual prominence

**Direct labeling advantages:**
- Instant association (no memory needed)
- Faster interpretation
- Better for colorblind readers (less reliance on color alone)

**When to use each:**

**Use DIRECT LABELS when:**
```
✓ Few categories (≤5)
✓ Space available near data
✓ Lines/points spatially separated
✓ Presentation/poster context (distant viewing)
```

**Use LEGEND when:**
```
✓ Many categories (>5)
✓ Data elements overlap (no space for labels)
✓ Consistency across multi-panel figure (shared legend)
```

**Code Example (Python) - Direct Labeling:**

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
time = np.linspace(0, 24, 100)

# Three treatment groups
data = {
    'Control': 100 + np.cumsum(np.random.randn(100) * 2),
    'Treatment A': 100 + np.cumsum(np.random.randn(100) * 2 + 0.3),
    'Treatment B': 100 + np.cumsum(np.random.randn(100) * 2 + 0.6)
}

colors = {
    'Control': '#7F8C8D',
    'Treatment A': '#3498DB',
    'Treatment B': '#E74C3C'
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Traditional legend
ax1 = axes[0]
for group, values in data.items():
    ax1.plot(time, values, color=colors[group], linewidth=2.5, label=group)

ax1.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Response (%)', fontsize=11, fontweight='bold')
ax1.set_title('A. Traditional Legend\n(Requires eye travel)',
             fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', frameon=True, fontsize=10)
ax1.grid(alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Panel B: Direct labeling
ax2 = axes[1]
for group, values in data.items():
    ax2.plot(time, values, color=colors[group], linewidth=2.5)

    # Place label at end of line
    ax2.text(time[-1] + 0.5, values[-1], group,
            color=colors[group], fontsize=10, fontweight='bold',
            va='center', ha='left')

ax2.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Response (%)', fontsize=11, fontweight='bold')
ax2.set_title('B. Direct Labeling\n(Instant association)',
             fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlim(0, 27)  # Extra space for labels

plt.tight_layout()
plt.savefig('direct_labeling_vs_legend.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R) - Direct Labeling:**

```r
library(ggplot2)
library(dplyr)
library(patchwork)
library(directlabels)

set.seed(42)
time <- seq(0, 24, length.out = 100)

data <- data.frame(
  time = rep(time, 3),
  response = c(
    100 + cumsum(rnorm(100, 0, 2)),
    100 + cumsum(rnorm(100, 0.3, 2)),
    100 + cumsum(rnorm(100, 0.6, 2))
  ),
  group = rep(c('Control', 'Treatment A', 'Treatment B'), each = 100)
)

colors <- c('Control' = '#7F8C8D',
            'Treatment A' = '#3498DB',
            'Treatment B' = '#E74C3C')

# Panel A: Traditional legend
p_a <- ggplot(data, aes(x = time, y = response, color = group)) +
  geom_line(size = 1.5) +
  scale_color_manual(values = colors) +
  labs(x = 'Time (hours)', y = 'Response (%)',
       title = 'A. Traditional Legend\n(Requires eye travel)',
       color = NULL) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', size = 12),
    axis.title = element_text(face = 'bold'),
    legend.position = c(0.2, 0.85),
    legend.background = element_rect(fill = 'white', color = 'black', size = 0.5),
    panel.grid.major = element_line(color = 'gray90', size = 0.3)
  )

# Panel B: Direct labeling
p_b <- ggplot(data, aes(x = time, y = response, color = group)) +
  geom_line(size = 1.5) +
  scale_color_manual(values = colors) +
  geom_text(data = data %>% group_by(group) %>% slice_tail(n = 1),
            aes(label = group, x = time + 1),
            hjust = 0, fontface = 'bold', size = 3.5) +
  labs(x = 'Time (hours)', y = 'Response (%)',
       title = 'B. Direct Labeling\n(Instant association)') +
  xlim(0, 27) +  # Extra space for labels
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', size = 12),
    axis.title = element_text(face = 'bold'),
    legend.position = 'none',
    panel.grid.major = element_line(color = 'gray90', size = 0.3)
  )

combined <- p_a | p_b
ggsave('direct_labeling_vs_legend.png', combined, width = 14, height = 5,
       dpi = 300, bg = 'white')
```

---

**Summary of Chapter 3 so far:**

✓ **Text hierarchy**: Critical (bold, large) > Important (regular) > Supporting (small, light)
✓ **Font choice**: Sans-serif (Arial/Helvetica) for figures
✓ **Axis labels**: Always include units: "Variable (unit)"
✓ **Panel labels**: Bold A, B, C, D in top-left corners
✓ **Statistical annotations**: Asterisks with clear legend; exact p-values when needed
✓ **Direct labeling**: Preferred over legends when space allows


## 3.7 Figure Captions: Complete but Concise

### The Essential Function of Captions

A well-written caption allows a figure to be **self-contained**—a reader should understand the key message without reading the full manuscript text. This is critical because:

1. **Readers scan figures first** before committing to reading the full paper
2. **Figures get reused** in presentations, reviews, where full context isn't available
3. **Journal requirements** mandate self-explanatory figures
4. **Accessibility** for readers who skip to results

---

### The Standard Caption Structure

**Template Format:**
```
[Figure Number]. [One-sentence summary]. [Detailed description].
[Sample size/statistics]. [Significance indicators]. [Technical details].
```

**Example Anatomy:**

```
Figure 3. Treatment A reduces tumor volume in xenograft mice.
(A) Representative H&E staining of tumor sections from control and
treatment groups at day 21 (scale bar: 100 μm). (B) Quantification
of tumor volume over time (n=8 mice per group, mean ± SEM).
(C) Kaplan-Meier survival curves showing improved survival in
treatment group (log-rank test, ***p < 0.001). All experiments
performed in triplicate. Error bars: SEM. * p<0.05, ** p<0.01,
*** p<0.001 (two-way ANOVA with Tukey post-hoc).
```

**Breakdown:**

```
✓ Title sentence: "Treatment A reduces tumor volume..."
  → States main finding clearly

✓ Panel descriptions: "(A) Representative H&E staining..."
  → Maps to panel labels, describes content

✓ Sample sizes: "n=8 mice per group"
  → Essential for interpretation

✓ Statistical methods: "two-way ANOVA with Tukey post-hoc"
  → Allows reader to assess rigor

✓ Technical specs: "scale bar: 100 μm"
  → Critical for microscopy/imaging
```

---

### What MUST Be Included

**Non-negotiable elements:**

**1. Sample size (n)**
```
✓ "n=3 biological replicates"
✓ "n=50 cells per condition from 3 independent experiments"
✓ "n=15 patients per group"

❌ Never omit sample size
→ Reader cannot assess statistical power or reliability
```

**2. Error bar definition**
```
✓ "Error bars: SEM"
✓ "Error bars: SD"
✓ "Boxes: IQR, whiskers: 1.5×IQR"

❌ "Error bars shown" (which type?)
→ SEM vs SD dramatically affects interpretation
```

**3. Statistical test and significance threshold**
```
✓ "Unpaired t-test, *p<0.05"
✓ "One-way ANOVA with Dunnett's post-hoc, **p<0.01"
✓ "Mann-Whitney U test (non-parametric), p=0.003"

❌ "Groups were significantly different" (what test? what p?)
```

**4. Scale information (for images)**
```
✓ "Scale bar: 50 μm"
✓ "Field of view: 200×200 μm"
✓ "All images same magnification (40×)"

❌ No scale information on microscopy
→ Renders images scientifically useless
```

---

### Logical Caption Organization

**For multi-panel figures, use consistent structure:**

**Format A: Panel-by-panel description**
```
Figure 2. [Overall title].
(A) [Panel A description with methods and n].
(B) [Panel B description with methods and n].
(C) [Panel C description with methods and n].
[Shared statistical details]. [Abbreviations].
```

**Format B: Grouped description**
```
Figure 2. [Overall title].
(A-C) [Common description for panels A-C].
(D, E) [Related panels described together].
[All statistics and methods]. [Abbreviations].
```

**Code Example Caption (for figure from section 3.4):**

```
Figure 2. Multi-modal analysis of treatment effects on cellular response.
(A) Correlation between baseline marker X and response variable Y in
control cells (n=50 cells, Pearson r=0.73, p<0.001). (B) Quantification
of cellular response across treatment groups (n=3 biological replicates,
each with 200 cells per condition, mean ± SEM, one-way ANOVA with Tukey
post-hoc, **p<0.01 vs control). (C) Temporal dynamics of signal intensity
following treatment application (n=4 independent experiments, mean ± SEM,
sampling rate: 100 Hz). (D) Hierarchical clustering heatmap of gene
expression changes (n=10 genes, 8 samples per condition, color scale:
log₂ fold change, diverging blue-white-red). All experiments performed
at 37°C in standard culture conditions. AU: arbitrary units.
```

**What this caption does well:**
- Every panel explicitly labeled
- Sample sizes clear for each analysis
- Statistical methods specified
- Technical details included (sampling rate, temperature)
- Abbreviations defined
- Stands alone without main text

---

### Common Caption Mistakes

**Mistake 1: Vague descriptions**
```
❌ "Expression levels are shown"
✓ "qRT-PCR quantification of gene X expression normalized to β-actin
   (n=4 biological replicates, mean ± SD, **p<0.01, unpaired t-test)"
```

**Mistake 2: Missing critical methods**
```
❌ "Cells were stained and imaged"
✓ "Cells were immunostained with anti-tubulin antibody (1:500, Abcam
   ab11304) and imaged on confocal microscope (Zeiss LSM 880, 63×
   objective, scale bar: 10 μm)"
```

**Mistake 3: Undefined abbreviations**
```
❌ "FPKM values shown in heatmap"
✓ "FPKM (Fragments Per Kilobase Million) values shown in heatmap"

Or define at first use:
"Expression quantified as FPKM (Fragments Per Kilobase Million)"
```

**Mistake 4: No error bar definition**
```
❌ "Error bars represent variability"
✓ "Error bars: standard error of the mean (SEM)"
```

**Mistake 5: Statistical significance without context**
```
❌ "***p<0.001"
✓ "***p<0.001 by two-way ANOVA with Bonferroni correction"
```

---

### Length Guidelines

**Target length:**
```
Simple single-panel figure: 2-4 sentences
Complex multi-panel figure: 6-10 sentences
Avoid: >15 sentences (break into multiple figures)
```

**Balance completeness with readability:**
```
Too short (incomplete):
"Figure 1. Expression data. Error bars: SEM."
→ Missing methods, n, statistics

Too long (overwhelming):
"Figure 1. We performed RNA-seq on samples collected at multiple
timepoints using Illumina NextSeq with 75bp paired-end reads,
which were then aligned to the reference genome using STAR aligner
version 2.7.3a with default parameters except... [300 words]"
→ Methods belong in Methods section, not caption

Just right:
"Figure 1. Temporal gene expression dynamics. RNA-seq quantification
of differentially expressed genes (DESeq2, adjusted p<0.05) across
timepoints (n=3 biological replicates per timepoint). Heatmap colors:
log₂ fold change vs. t=0. See Methods for sequencing details."
→ Essential info + pointer to full methods
```

---

### Journal-Specific Requirements

**Check your target journal's guidelines:**

**Nature family:**
```
- Captions should be concise but complete
- Define all abbreviations
- Include statistical tests and n
- Methods details belong in Methods, not captions
```

**Cell family:**
```
- Similar to Nature
- Emphasize reproducibility info (n, replicates)
- Related panels can share some description
```

**PLOS family:**
```
- Very detailed captions encouraged
- Must be understandable without main text
- Can include more methods details than Nature/Cell
```

**IEEE/ACM (computational):**
```
- Focus on algorithm/model details
- Parameter settings in caption acceptable
- Less emphasis on biological replicates
```

---

### **3.9 Text Hierarchy: Information vs. Support Text**

**Principle:** Distinguish between **information text** (gene names, pathway labels) and **support text** (axes, legends).

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: BAD - All text same size (no hierarchy)
ax1 = axes[0]
genes = ['TP53', 'BRCA1', 'MYC', 'KRAS']
expression = [120, 85, 150, 95]

bars1 = ax1.barh(genes, expression, color='#3498DB', edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Expression Level (FPKM)', fontsize=10)  # Same size
ax1.set_ylabel('Gene', fontsize=10)  # Same size
ax1.set_title('Gene Expression Profile', fontsize=10)  # Same size!
ax1.set_xlim(0, 180)

# Add values - also same size
for i, (gene, val) in enumerate(zip(genes, expression)):
    ax1.text(val + 3, i, f'{val}', va='center', fontsize=10)  # Same size

ax1.set_title('❌ BAD: All Text Same Size\n(No hierarchy, everything equally important)',
              fontsize=12, fontweight='bold', color='red', pad=15)

# Panel B: GOOD - Clear hierarchy (information > support)
ax2 = axes[1]
bars2 = ax2.barh(genes, expression, color='#27AE60', edgecolor='black', linewidth=1.5)

# INFORMATION TEXT: Larger, bold (what you want reader to remember)
ax2.set_ylabel('Gene', fontsize=14, fontweight='bold', color='black')  # Larger
for i, gene in enumerate(genes):
    ax2.text(-5, i, gene, va='center', ha='right',
            fontsize=13, fontweight='bold', color='black')  # Key information!

# Add expression values (also information)
for i, (gene, val) in enumerate(zip(genes, expression)):
    ax2.text(val + 3, i, f'{val}', va='center',
            fontsize=12, fontweight='bold', color='black')

# SUPPORT TEXT: Smaller, lighter (provides context)
ax2.set_xlabel('Expression Level (FPKM)', fontsize=10, color='gray')  # Smaller, gray
ax2.tick_params(axis='x', labelsize=9, labelcolor='gray')  # Support info
ax2.set_yticks([])  # Remove y-tick labels (genes are directly labeled)
ax2.set_xlim(-20, 180)

ax2.set_title('✓ GOOD: Clear Text Hierarchy\n(Information text > Support text)',
              fontsize=13, fontweight='bold', color='green', pad=15)

plt.tight_layout()
plt.savefig('text_hierarchy_information_vs_support.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.show()

```

---

### **3.10 Coordinate Axes and Legend Simplification**

**Principle:** Simplify coordinate labels and legends to reduce visual clutter while maintaining clarity.

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# BAD Example 1: Over-detailed axis labels
ax1 = axes[0, 0]
time = np.arange(0, 24, 0.5)
signal = 50 + 10*np.sin(2*np.pi*time/12) + np.random.randn(len(time))*2

ax1.plot(time, signal, 'o-', color='#3498DB', linewidth=2, markersize=4)

# Over-detailed tick labels
ax1.set_xticks(np.arange(0, 25, 2))
ax1.set_xticklabels([f'{h}:00:00' for h in range(0, 25, 2)],
                     rotation=45, fontsize=8)  # Too detailed!
ax1.set_xlabel('Time (hours:minutes:seconds)', fontsize=10)
ax1.set_ylabel('Signal Intensity (arbitrary units, normalized to baseline)',
               fontsize=10)  # Too wordy!
ax1.set_title('❌ BAD: Over-Detailed Labels\n(Unnecessary precision, hard to read)',
              fontsize=12, fontweight='bold', color='red')
ax1.grid(alpha=0.3)

# GOOD Example 1: Simplified axis labels
ax2 = axes[0, 1]
ax2.plot(time, signal, 'o-', color='#27AE60', linewidth=2.5, markersize=5)

# Simplified tick labels (just hours)
ax2.set_xticks(np.arange(0, 25, 4))
ax2.set_xticklabels([f'{h}' for h in range(0, 25, 4)], fontsize=10)
ax2.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')  # Simple, clear
ax2.set_ylabel('Signal (AU)', fontsize=12, fontweight='bold')  # AU = Arbitrary Units
ax2.set_title('✓ GOOD: Simplified Labels\n(Essential information only)',
              fontsize=13, fontweight='bold', color='green')
ax2.grid(alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# BAD Example 2: Redundant legend
ax3 = axes[1, 0]
conditions = ['Control Group (n=10, Mean±SD)',
              'Treatment Group A (n=10, Mean±SD)',
              'Treatment Group B (n=10, Mean±SD)']
colors = ['#7F8C8D', '#3498DB', '#E74C3C']

for i, (cond, color) in enumerate(zip(conditions, colors)):
    y = 50 + i*10 + np.random.randn(len(time))*2
    ax3.plot(time, y, 'o-', color=color, linewidth=2,
            markersize=4, label=cond)

ax3.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Response (AU)', fontsize=11, fontweight='bold')
ax3.set_title('❌ BAD: Redundant Legend Text\n(Too much detail in legend)',
              fontsize=12, fontweight='bold', color='red')
ax3.legend(loc='upper left', fontsize=8, frameon=True)
ax3.grid(alpha=0.3)

# GOOD Example 2: Simplified legend
ax4 = axes[1, 1]
conditions_simple = ['Control', 'Treatment A', 'Treatment B']

for i, (cond, color) in enumerate(zip(conditions_simple, colors)):
    y = 50 + i*10 + np.random.randn(len(time))*2
    ax4.plot(time, y, 'o-', color=color, linewidth=2.5,
            markersize=5, label=cond, alpha=0.8)

ax4.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Response (AU)', fontsize=12, fontweight='bold')
ax4.set_title('✓ GOOD: Simplified Legend\n(Details in caption: "n=10, Mean±SD")',
              fontsize=13, fontweight='bold', color='green')
ax4.legend(loc='upper left', fontsize=11, frameon=True,
          framealpha=0.9, edgecolor='black')
ax4.grid(alpha=0.3)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# Add note about caption
ax4.text(0.5, -0.15, 'Note: "All data shown as mean ± SD, n=10 per group"',
        transform=ax4.transAxes, ha='center', fontsize=9,
        style='italic', color='gray')

plt.tight_layout()
plt.savefig('simplify_axes_legend.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

**Guidelines for Axis Simplification:**

```python
# Simplification Rules:

# 1. TIME AXES:
# BAD:  "Time (hours:minutes:seconds)"
# GOOD: "Time (hours)"  → Put precision in Methods

# 2. LONG VARIABLE NAMES:
# BAD:  "Relative Fluorescence Intensity (normalized to control)"
# GOOD: "Relative Intensity (AU)"  → Full description in caption

# 3. UNITS:
# BAD:  "Concentration (micromolar, μM)"
# GOOD: "Concentration (μM)"  → One unit symbol is enough

# 4. STATISTICAL INFO:
# BAD:  "Response (Mean ± Standard Error of Mean, n=5)"
# GOOD: "Response (AU)"  → Statistical details in caption

# 5. LEGEND DETAILS:
# BAD:  "Wild-type mice, age 8-12 weeks, n=15, p<0.05 vs control"
# GOOD: "Wild-type"  → Details in caption or Methods
```

---

### **3.11 Information Text: Gene and Pathway Labels**

**Principle:** Make critical information (gene names, pathways) prominent and clear, but avoid overwhelming detail.

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

np.random.seed(42)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# BAD: Too many gene labels (overwhelming)
ax1 = axes[0]
n_genes = 50
x = np.random.randn(n_genes)
y = np.random.randn(n_genes)
colors = np.random.choice(['#3498DB', '#E74C3C', '#27AE60'], n_genes)
gene_names = [f'Gene{i+1}' for i in range(n_genes)]

ax1.scatter(x, y, c=colors, s=80, alpha=0.6, edgecolors='black', linewidths=0.5)

# Label ALL genes (too many!)
for i, gene in enumerate(gene_names):
    ax1.text(x[i], y[i], gene, fontsize=6, ha='center')

ax1.set_xlabel('PC1', fontsize=11, fontweight='bold')
ax1.set_ylabel('PC2', fontsize=11, fontweight='bold')
ax1.set_title('❌ BAD: Too Many Labels\n(Overwhelming, unreadable)',
              fontsize=13, fontweight='bold', color='red')
ax1.grid(alpha=0.3)

# GOOD: Only label important genes
ax2 = axes[1]
ax2.scatter(x, y, c=colors, s=100, alpha=0.6, edgecolors='black', linewidths=0.5)

# Identify top 5 "significant" genes (example: most extreme positions)
distances = np.sqrt(x**2 + y**2)
top_indices = np.argsort(distances)[-5:]

# Label only important genes with clear callouts
for idx in top_indices:
    # Gene point
    ax2.scatter(x[idx], y[idx], s=200, facecolors='none',
               edgecolors='black', linewidths=3, zorder=10)

    # Callout annotation
    ax2.annotate(gene_names[idx],
                xy=(x[idx], y[idx]),
                xytext=(20, 20), textcoords='offset points',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5',
                         facecolor='yellow', alpha=0.8,
                         edgecolor='black', linewidth=1.5),
                arrowprops=dict(arrowstyle='->',
                               connectionstyle='arc3,rad=0.3',
                               lw=2, color='black'))

ax2.set_xlabel('PC1', fontsize=12, fontweight='bold')
ax2.set_ylabel('PC2', fontsize=12, fontweight='bold')
ax2.set_title('✓ GOOD: Label Only Key Genes\n(Clear focus on important information)',
              fontsize=13, fontweight='bold', color='green')
ax2.grid(alpha=0.3)

# Add note
ax2.text(0.02, 0.02, f'Top 5 of {n_genes} genes labeled\n(Full list in Table S1)',
        transform=ax2.transAxes, fontsize=9, style='italic',
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('information_text_gene_labels.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

**Pathway/Network Diagram Information Text:**

```python
import matplotlib.pyplot as plt
import networkx as nx

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Create example pathway network
G = nx.DiGraph()
nodes = ['Receptor', 'Kinase1', 'Kinase2', 'TF', 'Gene']
edges = [('Receptor', 'Kinase1'), ('Kinase1', 'Kinase2'),
         ('Kinase2', 'TF'), ('TF', 'Gene')]
G.add_nodes_from(nodes)
G.add_edges_from(edges)

pos = {
    'Receptor': (0, 0),
    'Kinase1': (1, 0.5),
    'Kinase2': (2, 0.5),
    'TF': (3, 0.3),
    'Gene': (4, 0)
}

# BAD: Too much detail on diagram
ax1 = axes[0]
node_labels = {
    'Receptor': 'EGFR\n(Epidermal Growth\nFactor Receptor)\nMembrane protein\nBinds EGF ligand',
    'Kinase1': 'RAF\n(Rapidly Accelerated\nFibrosarcoma)\nSerine/threonine kinase',
    'Kinase2': 'MEK\n(MAP/ERK Kinase)\nDual-specificity kinase',
    'TF': 'ERK\n(Extracellular signal-\nRegulated Kinase)\nTranscription regulator',
    'Gene': 'Target Gene\n(e.g., FOS, MYC)\nExpression regulation'
}

nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='#3498DB',
                      node_size=3000, alpha=0.7)
nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='black',
                      width=2, arrowsize=20, arrowstyle='->')
nx.draw_networkx_labels(G, pos, node_labels, ax=ax1,
                       font_size=7, font_weight='bold')

ax1.set_xlim(-0.5, 4.5)
ax1.set_ylim(-1, 1.5)
ax1.axis('off')
ax1.set_title('❌ BAD: Too Much Detail\n(Cluttered, hard to read)',
              fontsize=13, fontweight='bold', color='red', pad=20)

# GOOD: Simplified labels, details in caption
ax2 = axes[1]
node_labels_simple = {
    'Receptor': 'EGFR',
    'Kinase1': 'RAF',
    'Kinase2': 'MEK',
    'TF': 'ERK',
    'Gene': 'Target\nGene'
}

# Color-code by function
node_colors = ['#E74C3C', '#3498DB', '#3498DB', '#F39C12', '#27AE60']

nx.draw_networkx_nodes(G, pos, ax=ax2, node_color=node_colors,
                      node_size=2500, alpha=0.8, edgecolors='black', linewidths=2)
nx.draw_networkx_edges(G, pos, ax=ax2, edge_color='black',
                      width=3, arrowsize=25, arrowstyle='->')
nx.draw_networkx_labels(G, pos, node_labels_simple, ax=ax2,
                       font_size=12, font_weight='bold')

ax2.set_xlim(-0.5, 4.5)
ax2.set_ylim(-1, 1.5)
ax2.axis('off')
ax2.set_title('✓ GOOD: Simplified Labels\n(Details in caption)',
              fontsize=13, fontweight='bold', color='green', pad=20)

# Add legend for colors
legend_elements = [
    mpatches.Patch(facecolor='#E74C3C', edgecolor='black',
                   label='Receptor'),
    mpatches.Patch(facecolor='#3498DB', edgecolor='black',
                   label='Kinases'),
    mpatches.Patch(facecolor='#F39C12', edgecolor='black',
                   label='Transcription Factor'),
    mpatches.Patch(facecolor='#27AE60', edgecolor='black',
                   label='Gene')
]
ax2.legend(handles=legend_elements, loc='upper right',
          fontsize=10, frameon=True, title='Component Type')

# Caption text suggestion
fig.text(0.5, 0.02,
        'Caption example: "EGFR/RAF/MEK/ERK signaling cascade. '\
        'EGFR: Epidermal Growth Factor Receptor (membrane receptor). '\
        'RAF/MEK/ERK: Sequential kinase cascade (detailed in Methods)."',
        ha='center', fontsize=9, style='italic', wrap=True)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('pathway_information_text.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

---

## 3.12 Common Typography Mistakes and How to Avoid Them

### Mistake 1: Inconsistent Font Sizing

**The Problem:**

```
❌ BAD EXAMPLE:
- Axis labels: 11pt
- Panel A title: 14pt
- Panel B title: 12pt (inconsistent!)
- Legend: 8pt in panel A, 10pt in panel B
→ Looks unprofessional, distracts reader
```

**The Fix:**

```
✓ ESTABLISH HIERARCHY ONCE, APPLY EVERYWHERE:

Font size scale:
- Panel labels: 14pt bold
- Titles: 12pt bold
- Axis labels: 11pt bold
- Tick labels: 9pt regular
- Legends: 9pt regular
- Annotations: 9pt regular/italic
- Footnotes: 8pt regular

Code implementation: Set rcParams globally (Python) or theme (R)
```

**Code Example (Python) - Consistent Sizing:**

```python
import matplotlib.pyplot as plt

# SET ONCE at beginning of script
FONT_SIZES = {
    'panel_label': 14,
    'title': 12,
    'axis_label': 11,
    'tick_label': 9,
    'legend': 9,
    'annotation': 9
}

plt.rcParams.update({
    'font.size': FONT_SIZES['tick_label'],
    'axes.labelsize': FONT_SIZES['axis_label'],
    'axes.titlesize': FONT_SIZES['title'],
    'legend.fontsize': FONT_SIZES['legend'],
    'xtick.labelsize': FONT_SIZES['tick_label'],
    'ytick.labelsize': FONT_SIZES['tick_label']
})

# Now all figures use consistent sizing
# Panel labels added manually with FONT_SIZES['panel_label']
```

**Code Example (R) - Consistent Sizing:**

```r
library(ggplot2)

# DEFINE ONCE
FONT_SIZES <- list(
  panel_label = 14,
  title = 12,
  axis_label = 11,
  tick_label = 9,
  legend = 9,
  annotation = 9
)

# Create reusable theme
my_theme <- theme_classic(base_size = FONT_SIZES$tick_label) +
  theme(
    axis.title = element_text(size = FONT_SIZES$axis_label, face = 'bold'),
    plot.title = element_text(size = FONT_SIZES$title, face = 'bold', hjust = 0.5),
    legend.text = element_text(size = FONT_SIZES$legend),
    legend.title = element_text(size = FONT_SIZES$legend, face = 'bold'),
    axis.text = element_text(size = FONT_SIZES$tick_label)
  )

# Apply to all plots
p <- ggplot(...) + ... + my_theme
```

---

### Mistake 2: Poor Contrast (Low Readability)

**The Problem:**

```
❌ Light gray text on white background
❌ Yellow text on white background (invisible)
❌ Low-contrast colors for critical labels

Result: Illegible when printed, invisible in bright rooms
```

**The Fix:**

```
✓ ALWAYS test contrast ratio:

Minimum WCAG AA standard: 4.5:1 for normal text
Better: 7:1 (AAA standard)

Safe text colors on white background:
✓ Black (#000000) - 21:1 contrast
✓ Dark gray (#333333) - 12.6:1 contrast
✓ Dark blue (#003366) - 10.4:1 contrast

Unsafe:
❌ Light gray (#CCCCCC) - 1.6:1 contrast
❌ Pastel yellow (#FFFF99) - 1.2:1 contrast
```

**Testing tool:**
```
WebAIM Contrast Checker: webaim.org/resources/contrastchecker/
Input: Background color + Text color
Output: Pass/Fail for WCAG standards
```

---

### Mistake 3: Overlapping Text

**The Problem:**

```
❌ Axis labels collide with tick labels
❌ Data labels overlap each other
❌ Legend obscures data points
```

**The Fix:**

```
✓ Rotate text when necessary:
- X-axis labels at 45° if long category names
- Never rotate >90° (unreadable)

✓ Adjust label positions programmatically:
- matplotlib: adjustText library for automatic label placement
- R: ggrepel package for non-overlapping labels

✓ Increase figure size if crowded
✓ Consider splitting into multiple panels
```

**Code Example (Python) - Avoiding Overlap:**

```python
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text  # pip install adjustText

np.random.seed(42)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Generate scatter data
x = np.random.rand(20) * 10
y = np.random.rand(20) * 10
labels = [f'Gene{i+1}' for i in range(20)]

# BAD: Overlapping labels
ax1 = axes[0]
ax1.scatter(x, y, s=50, color='#3498DB', edgecolors='black', linewidths=0.5)
for i, label in enumerate(labels):
    ax1.text(x[i], y[i], label, fontsize=8, ha='center')

ax1.set_title('❌ BAD: Overlapping Labels', fontsize=12, fontweight='bold', color='red')
ax1.set_xlabel('Variable X', fontsize=11, fontweight='bold')
ax1.set_ylabel('Variable Y', fontsize=11, fontweight='bold')

# GOOD: Adjusted labels (non-overlapping)
ax2 = axes[1]
ax2.scatter(x, y, s=50, color='#3498DB', edgecolors='black', linewidths=0.5)

texts = []
for i, label in enumerate(labels):
    texts.append(ax2.text(x[i], y[i], label, fontsize=8, ha='center'))

# Automatically adjust positions to avoid overlap
adjust_text(texts, ax=ax2, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

ax2.set_title('✓ GOOD: Adjusted Labels', fontsize=12, fontweight='bold', color='green')
ax2.set_xlabel('Variable X', fontsize=11, fontweight='bold')
ax2.set_ylabel('Variable Y', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('overlapping_text_fix.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R) - Avoiding Overlap:**

```r
library(ggplot2)
library(ggrepel)  # For non-overlapping labels

set.seed(42)
data <- data.frame(
  x = runif(20, 0, 10),
  y = runif(20, 0, 10),
  label = paste0('Gene', 1:20)
)

# BAD: Overlapping labels
p_bad <- ggplot(data, aes(x = x, y = y)) +
  geom_point(color = '#3498DB', size = 3) +
  geom_text(aes(label = label), size = 3, hjust = 0.5, vjust = 0.5) +
  labs(title = '❌ BAD: Overlapping Labels',
       x = 'Variable X', y = 'Variable Y') +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'red', size = 12),
    axis.title = element_text(face = 'bold')
  )

# GOOD: Adjusted labels with ggrepel
p_good <- ggplot(data, aes(x = x, y = y)) +
  geom_point(color = '#3498DB', size = 3) +
  geom_text_repel(aes(label = label), size = 3,
                  box.padding = 0.5, point.padding = 0.3,
                  segment.color = 'gray50', segment.size = 0.3) +
  labs(title = '✓ GOOD: Adjusted Labels',
       x = 'Variable X', y = 'Variable Y') +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'darkgreen', size = 12),
    axis.title = element_text(face = 'bold')
  )

library(patchwork)
combined <- p_bad | p_good
ggsave('overlapping_text_fix.png', combined, width = 14, height = 5,
       dpi = 300, bg = 'white')
```

---

### Mistake 4: Unclear Axis Notation

**The Problem:**

```
❌ Scientific notation without clear base:
Axis label: "1e6" (is this 1×10⁶? 1e⁶?)

❌ Confusing multiplication symbols:
"Time (x10³ s)" — what does x mean here?

❌ Units in tick labels instead of axis label:
Y-axis ticks: "5 mg/mL", "10 mg/mL", "15 mg/mL"
→ Redundant, cluttered
```

**The Fix:**

```
✓ USE PROPER SUPERSCRIPTS for exponents:
"Concentration (×10⁶ cells/mL)" or "Concentration (10⁶ cells/mL)"

✓ UNITS ON AXIS LABEL, not tick labels:
Label: "Concentration (mg/mL)"
Ticks: "5", "10", "15"

✓ EXPLICIT scientific notation:
Label: "Distance (10³ km)" meaning values are in thousands of km
Tick showing "5" = 5000 km
```

**Code Example (Python) - Proper Axis Notation:**

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Sample data in large numbers
x = np.linspace(0, 10, 50)
y = np.random.rand(50) * 1e6  # Values in millions

# BAD: Automatic scientific notation (unclear)
ax1 = axes[0]
ax1.plot(x, y, 'o-', color='#3498DB', linewidth=2, markersize=4)
ax1.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Cell Count', fontsize=11, fontweight='bold')  # No units!
ax1.set_title('❌ BAD: Unclear Notation', fontsize=12, fontweight='bold', color='red')
ax1.grid(alpha=0.3)

# GOOD: Explicit scaling in label
ax2 = axes[1]
ax2.plot(x, y/1e6, 'o-', color='#27AE60', linewidth=2, markersize=4)  # Scale data
ax2.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Cell Count (×10⁶)', fontsize=11, fontweight='bold')  # Clear units
ax2.set_title('✓ GOOD: Explicit Scaling', fontsize=12, fontweight='bold', color='green')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('axis_notation_fix.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R) - Proper Axis Notation:**

```r
library(ggplot2)
library(scales)
library(patchwork)

set.seed(42)
data <- data.frame(
  time = seq(0, 10, length.out = 50),
  count = runif(50, 0, 1) * 1e6  # Values in millions
)

# BAD: Automatic scientific notation
p_bad <- ggplot(data, aes(x = time, y = count)) +
  geom_line(color = '#3498DB', size = 1.2) +
  geom_point(color = '#3498DB', size = 2) +
  labs(x = 'Time (s)', y = 'Cell Count',  # No scale indicated!
       title = '❌ BAD: Unclear Notation') +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'red', size = 12),
    axis.title = element_text(face = 'bold'),
    panel.grid.major = element_line(color = 'gray90', size = 0.3)
  )

# GOOD: Explicit scaling
p_good <- ggplot(data, aes(x = time, y = count / 1e6)) +  # Scale data
  geom_line(color = '#27AE60', size = 1.2) +
  geom_point(color = '#27AE60', size = 2) +
  labs(x = 'Time (s)', y = 'Cell Count (×10⁶)',  # Clear units
       title = '✓ GOOD: Explicit Scaling') +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'darkgreen', size = 12),
    axis.title = element_text(face = 'bold'),
    panel.grid.major = element_line(color = 'gray90', size = 0.3)
  )

combined <- p_bad | p_good
ggsave('axis_notation_fix.png', combined, width = 12, height = 5,
       dpi = 300, bg = 'white')
```

---

### Mistake 5: Ambiguous Legends

**The Problem:**

```
❌ Legend says: "Treatment"
→ Which treatment? A, B, or C?

❌ Legend uses symbols without explanation:
"●" (what does circle mean?)
"▲" (triangle significance?)

❌ Legend order doesn't match visual prominence
```

**The Fix:**

```
✓ SPECIFIC labels:
Not: "Treatment"
But: "Treatment A (10 μM)", "Treatment B (50 μM)"

✓ COMBINE symbols with text:
"● Control (n=15)"
"▲ Drug A (n=15)"
"■ Drug B (n=15)"

✓ ORDER logically:
- Control first
- Experimental conditions in ascending dose/time
- Or match left-to-right/top-to-bottom order in figure
```

---

### Typography Checklist Before Submission

- [ ] **Font family consistent** across all panels (Arial/Helvetica)
- [ ] **Font sizes consistent** within hierarchy (use template)
- [ ] **High contrast** (black or #333333 on white)
- [ ] **No overlapping text** (use adjustText/ggrepel if needed)
- [ ] **Axis labels include units** in format "Variable (unit)"
- [ ] **Scientific notation explicit** (×10⁶ not 1e6)
- [ ] **Legends specific** ("Drug A 10μM" not just "Drug A")
- [ ] **Panel labels bold** (A, B, C...) and consistently placed
- [ ] **Statistical annotations clear** (*, **, *** defined)
- [ ] **Caption complete**: n, statistics, error bar type, scale bars

---

**End of Chapter 3: Typography, Annotation & Labels**

**Key Takeaways:**
- **Text is functional infrastructure**, not decoration
- **Hierarchy guides attention**: Bold/large for critical, regular/small for supporting
- **Sans-serif fonts** (Arial/Helvetica) for all figures
- **Axis labels must include units**: "Variable (unit)"
- **Panel labels** (A, B, C) enable precise referencing
- **Statistical annotations** need clear notation and legend
- **Direct labeling > Legends** when space allows
- **Captions must be self-contained**: n, methods, statistics, definitions
- **Consistency** in sizing, spacing, and formatting signals professionalism

---

