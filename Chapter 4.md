# Chapter 4: Data Encoding & Graph Selection

## 4.1 The Fundamental Principle: Match Plot Type to Data Structure

### Why Graph Selection Matters

The choice of visualization type is **not arbitrary or aesthetic**—it must be determined by:

1. **Data structure** (continuous, categorical, temporal, hierarchical, etc.)
2. **Comparison type** (distribution, relationship, composition, temporal change)
3. **Number of variables** (univariate, bivariate, multivariate)
4. **Message priority** (what insight should jump out immediately?)

**Using the wrong plot type is not a stylistic choice—it's a scientific error that misrepresents your data.**

---

### The Data-Plot Decision Tree

```
START: What am I trying to show?

├─ COMPARISON between groups
│   ├─ Few groups (<10), one variable
│   │   └─ Bar chart, dot plot, box plot
│   ├─ Many groups (>10), one variable
│   │   └─ Violin plot, ridgeline plot, small multiples
│   └─ Two+ variables across groups
│       └─ Grouped bar chart, heatmap, PCA plot
│
├─ DISTRIBUTION of one variable
│   ├─ Single distribution
│   │   └─ Histogram, density plot, box plot
│   ├─ Multiple distributions to compare
│   │   └─ Overlapping density, violin plot, box plot grid
│   └─ Distribution + summary stats
│       └─ Box plot, violin plot with mean/median
│
├─ RELATIONSHIP between two variables
│   ├─ Both continuous (quantitative)
│   │   └─ Scatter plot, line graph (if ordered), hexbin (if dense)
│   ├─ One continuous, one categorical
│   │   └─ Box plot, violin plot, strip plot
│   └─ Both categorical
│       └─ Heatmap, mosaic plot, grouped bar chart
│
├─ CHANGE OVER TIME (temporal)
│   ├─ Single time series
│   │   └─ Line graph
│   ├─ Multiple time series (few)
│   │   └─ Line graph with multiple lines
│   ├─ Multiple time series (many)
│   │   └─ Small multiples, spaghetti plot, heatmap
│   └─ Cyclic patterns
│       └─ Circular/polar plot, seasonal decomposition
│
├─ COMPOSITION (parts of a whole)
│   ├─ Static composition
│   │   └─ Stacked bar chart (avoid pie charts)
│   ├─ Composition changing over time
│   │   └─ Area chart, stacked bar chart over time
│   └─ Hierarchical composition
│       └─ Treemap, sunburst diagram
│
└─ SPATIAL/NETWORK relationships
    ├─ Geographic data
    │   └─ Choropleth map, point map, heat map
    ├─ Network/graph data
    │   └─ Node-link diagram, adjacency matrix
    └─ Hierarchical relationships
        └─ Dendrogram, tree diagram, sankey diagram
```

---

## 4.2 Comparison: Showing Differences Between Groups

### Bar Charts: The Standard for Categorical Comparisons

**When to use:**
- Comparing discrete categories
- Showing absolute values (not proportions)
- Few to moderate number of categories (<15)
- When exact values matter

**Best practices:**
```
✓ Always start y-axis at zero (bars represent magnitude)
✓ Order categories logically (not alphabetically unless necessary)
✓ Use consistent bar width
✓ Leave space between bars (width:gap ratio ~3:1)
✓ Add error bars if showing means (specify SEM or SD)
```

**Common mistakes:**
```
❌ Truncated y-axis (exaggerates small differences)
❌ 3D bars (perspective distortion)
❌ Too many categories (visual clutter)
❌ Inconsistent ordering across panels
```

**Code Example (Python) - Proper Bar Chart:**

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Sample data: treatment comparison
categories = ['Control', 'Drug A\n(10 μM)', 'Drug B\n(50 μM)', 'Drug C\n(100 μM)']
values = [25.3, 32.7, 28.1, 35.9]
errors = [2.8, 3.1, 3.5, 4.2]
n_samples = [15, 15, 15, 15]

# Color scheme: gray for control, blue/red for treatments
colors = ['#7F8C8D', '#3498DB', '#3498DB', '#E74C3C']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# BAD: Truncated axis, no error bars
ax1 = axes[0]
ax1.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5, width=0.6)
ax1.set_ylim(20, 40)  # Truncated!
ax1.set_ylabel('Cell Viability (%)', fontsize=11, fontweight='bold')
ax1.set_title('❌ BAD: Truncated Axis\n(Exaggerates differences)',
             fontsize=12, fontweight='bold', color='red')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# GOOD: Full axis, error bars, sample sizes
ax2 = axes[1]
bars = ax2.bar(categories, values, color=colors,
              edgecolor='black', linewidth=1.5, width=0.6)
ax2.errorbar(categories, values, yerr=errors, fmt='none',
            ecolor='black', capsize=8, linewidth=2)

# Add sample sizes on bars
for i, (bar, n) in enumerate(zip(bars, n_samples)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + errors[i] + 1,
            f'n={n}', ha='center', va='bottom', fontsize=9)

ax2.set_ylim(0, 50)  # Full scale from zero
ax2.set_ylabel('Cell Viability (%)', fontsize=11, fontweight='bold')
ax2.set_title('✓ GOOD: Full Scale + Error Bars + n\n(Honest representation)',
             fontsize=12, fontweight='bold', color='green')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', alpha=0.3)

# Add statistical comparison
ax2.plot([1, 3], [42, 42], 'k-', linewidth=1.5)
ax2.text(2, 43, '**', ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('bar_chart_best_practices.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R) - Proper Bar Chart:**

```r
library(ggplot2)
library(dplyr)
library(patchwork)

# Data
data <- data.frame(
  category = factor(c('Control', 'Drug A\n(10 μM)', 'Drug B\n(50 μM)', 'Drug C\n(100 μM)'),
                   levels = c('Control', 'Drug A\n(10 μM)', 'Drug B\n(50 μM)', 'Drug C\n(100 μM)')),
  value = c(25.3, 32.7, 28.1, 35.9),
  error = c(2.8, 3.1, 3.5, 4.2),
  n = c(15, 15, 15, 15)
)

colors <- c('Control' = '#7F8C8D', 'Drug A\n(10 μM)' = '#3498DB',
            'Drug B\n(50 μM)' = '#3498DB', 'Drug C\n(100 μM)' = '#E74C3C')

# BAD: Truncated axis
p_bad <- ggplot(data, aes(x = category, y = value, fill = category)) +
  geom_bar(stat = 'identity', color = 'black', size = 1, width = 0.6) +
  scale_fill_manual(values = colors) +
  coord_cartesian(ylim = c(20, 40)) +  # Truncated!
  labs(y = 'Cell Viability (%)',
       title = '❌ BAD: Truncated Axis\n(Exaggerates differences)') +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'red', size = 12),
    axis.title.x = element_blank(),
    axis.title.y = element_text(face = 'bold'),
    legend.position = 'none'
  )

# GOOD: Full scale, error bars, sample sizes
p_good <- ggplot(data, aes(x = category, y = value, fill = category)) +
  geom_bar(stat = 'identity', color = 'black', size = 1, width = 0.6) +
  geom_errorbar(aes(ymin = value - error, ymax = value + error),
               width = 0.25, size = 1) +
  scale_fill_manual(values = colors) +

  # Add sample sizes
  geom_text(aes(label = paste0('n=', n), y = value + error + 2),
           size = 3, vjust = 0) +

  # Statistical annotation
  annotate('segment', x = 2, xend = 4, y = 42, yend = 42, size = 1) +
  annotate('text', x = 3, y = 43, label = '**', size = 5, fontface = 'bold') +

  ylim(0, 50) +
  labs(y = 'Cell Viability (%)',
       title = '✓ GOOD: Full Scale + Error Bars + n\n(Honest representation)') +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'darkgreen', size = 12),
    axis.title.x = element_blank(),
    axis.title.y = element_text(face = 'bold'),
    legend.position = 'none',
    panel.grid.major.y = element_line(color = 'gray90', size = 0.3)
  )

combined <- p_bad | p_good
ggsave('bar_chart_best_practices.png', combined, width = 14, height = 5,
       dpi = 300, bg = 'white')
```

---

### Box Plots: Showing Full Distribution

**When to use:**
- Showing distribution shape, not just mean
- Comparing distributions across groups
- Detecting outliers
- When sample size is moderate to large (n>20 recommended)

**What box plots show:**

```
Components:
├─ Box: Interquartile range (IQR = 25th to 75th percentile)
├─ Line inside box: Median (50th percentile)
├─ Whiskers: Typically extend to 1.5×IQR or min/max
└─ Points beyond whiskers: Outliers

✓ Shows: distribution shape, spread, skewness, outliers
❌ Hides: exact sample size (unless noted), bimodality (if subtle)

```

**Advantages over bar charts:**
- Shows data distribution, not just mean
- Highlights outliers automatically
- More informative for skewed data
- Better for comparing variability

**Code Example (Python) - Box Plot:**

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Generate data with different distributions
control = np.random.normal(25, 5, 50)
drug_a = np.random.normal(32, 4, 50)
drug_b = np.concatenate([np.random.normal(28, 3, 45),
                         np.array([15, 18, 42, 45, 48])])  # With outliers
drug_c = np.random.lognormal(3.5, 0.3, 50)  # Skewed distribution

data = [control, drug_a, drug_b, drug_c]
labels = ['Control', 'Drug A', 'Drug B', 'Drug C']
colors = ['#7F8C8D', '#3498DB', '#3498DB', '#E74C3C']

fig, ax = plt.subplots(figsize=(9, 6))

# Create box plot
bp = ax.boxplot(data, labels=labels, patch_artist=True,
               widths=0.6,
               boxprops=dict(linewidth=1.5),
               whiskerprops=dict(linewidth=1.5),
               capprops=dict(linewidth=1.5),
               medianprops=dict(color='red', linewidth=2),
               flierprops=dict(marker='o', markerfacecolor='black',
                             markersize=6, alpha=0.5))

# Color boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add mean markers (in addition to median line)
means = [np.mean(d) for d in data]
ax.scatter(range(1, len(means)+1), means, marker='D', s=80,
          color='white', edgecolors='black', linewidths=2,
          zorder=3, label='Mean')

ax.set_ylabel('Cell Viability (%)', fontsize=12, fontweight='bold')
ax.set_title('Box Plot: Full Distribution Comparison',
            fontsize=13, fontweight='bold')

# Add legend
ax.legend(loc='upper left', frameon=True, fontsize=10)

# Add sample sizes
for i, n in enumerate([len(d) for d in data]):
    ax.text(i+1, ax.get_ylim()[0] + 2, f'n={n}',
           ha='center', fontsize=9, style='italic')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

# Add explanation annotations
ax.annotate('', xy=(1.2, np.percentile(control, 75)),
           xytext=(1.5, np.percentile(control, 75)),
           arrowprops=dict(arrowstyle='->', lw=1.5))
ax.text(1.6, np.percentile(control, 75), '75th percentile',
       fontsize=8, va='center')

ax.annotate('', xy=(1.2, np.percentile(control, 50)),
           xytext=(1.5, np.percentile(control, 50)),
           arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))
ax.text(1.6, np.percentile(control, 50), 'Median',
       fontsize=8, va='center', color='red')

plt.tight_layout()
plt.savefig('box_plot_example.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R) - Box Plot:**

```r
library(ggplot2)
library(dplyr)

set.seed(42)

# Generate data
data <- data.frame(
  group = rep(c('Control', 'Drug A', 'Drug B', 'Drug C'), each = 50),
  value = c(
    rnorm(50, 25, 5),
    rnorm(50, 32, 4),
    c(rnorm(45, 28, 3), c(15, 18, 42, 45, 48)),  # With outliers
    rlnorm(50, 3.5, 0.3)  # Skewed
  )
)

data$group <- factor(data$group, levels = c('Control', 'Drug A', 'Drug B', 'Drug C'))

colors <- c('Control' = '#7F8C8D', 'Drug A' = '#3498DB',
            'Drug B' = '#3498DB', 'Drug C' = '#E74C3C')

# Calculate means for overlay
means <- data %>%
  group_by(group) %>%
  summarise(mean_val = mean(value))

# Calculate sample sizes
sample_sizes <- data %>%
  group_by(group) %>%
  summarise(n = n(), min_val = min(value) - 5)

p <- ggplot(data, aes(x = group, y = value, fill = group)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 16, outlier.size = 2,
              outlier.alpha = 0.5, width = 0.6) +

  # Add mean markers
  geom_point(data = means, aes(y = mean_val),
            shape = 23, size = 4, fill = 'white', color = 'black', stroke = 1.5) +

  # Color scheme
  scale_fill_manual(values = colors) +

  # Add sample sizes
  geom_text(data = sample_sizes, aes(y = min_val, label = paste0('n=', n)),
           size = 3, fontface = 'italic') +

  labs(y = 'Cell Viability (%)',
       title = 'Box Plot: Full Distribution Comparison') +

  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', size = 13),
    axis.title.x = element_blank(),
    axis.title.y = element_text(face = 'bold'),
    legend.position = 'none',
    panel.grid.major.y = element_line(color = 'gray90', size = 0.3)
  )

# Add annotations
p <- p +
  annotate('segment', x = 1.2, xend = 1.5,
          y = quantile(data$value[data$group == 'Control'], 0.75),
          yend = quantile(data$value[data$group == 'Control'], 0.75),
          arrow = arrow(length = unit(0.2, 'cm')), size = 1) +
  annotate('text', x = 1.6, y = quantile(data$value[data$group == 'Control'], 0.75),
          label = '75th percentile', hjust = 0, size = 3) +
  annotate('segment', x = 1.2, xend = 1.5,
          y = median(data$value[data$group == 'Control']),
          yend = median(data$value[data$group == 'Control']),
          arrow = arrow(length = unit(0.2, 'cm')), size = 1, color = 'red') +
  annotate('text', x = 1.6, y = median(data$value[data$group == 'Control']),
          label = 'Median', hjust = 0, size = 3, color = 'red')

ggsave('box_plot_example.png', p, width = 9, height = 6,
       dpi = 300, bg = 'white')
```

---

### Violin Plots: Enhanced Distribution Visualization

**When to use:**
- Showing full distribution density (better than box plots for bimodal/multimodal data)
- Comparing distribution shapes across groups
- When you want both density and quartiles visible

**Advantages:**
- Shows distribution shape more clearly than box plots
- Reveals bimodality (two peaks) that box plots hide
- More informative than histograms for comparisons

**Code Example (Python) - Violin Plot:**

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

np.random.seed(42)

# Generate bimodal data (two peaks) - box plot would hide this!
control = np.random.normal(25, 3, 50)
drug_a = np.concatenate([np.random.normal(20, 2, 25),
                         np.random.normal(35, 2, 25)])  # Bimodal!
drug_b = np.random.normal(32, 4, 50)

# Combine into DataFrame for seaborn
import pandas as pd
df = pd.DataFrame({
    'Group': ['Control']*50 + ['Drug A']*50 + ['Drug B']*50,
    'Value': np.concatenate([control, drug_a, drug_b])
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Box plot (HIDES bimodality)
ax1 = axes[0]
sns.boxplot(data=df, x='Group', y='Value', ax=ax1,
           palette=['#7F8C8D', '#3498DB', '#E74C3C'])
ax1.set_ylabel('Cell Viability (%)', fontsize=12, fontweight='bold')
ax1.set_xlabel('')
ax1.set_title('Box Plot: Hides Bimodal Distribution\n(Drug A appears normal)',
             fontsize=12, fontweight='bold', color='red')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Violin plot (REVEALS bimodality)
ax2 = axes[1]
sns.violinplot(data=df, x='Group', y='Value', ax=ax2,
              palette=['#7F8C8D', '#3498DB', '#E74C3C'],
              inner='box')  # Add box plot inside
ax2.set_ylabel('Cell Viability (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('')
ax2.set_title('Violin Plot: Reveals Bimodal Distribution\n(Drug A has two response populations)',
             fontsize=12, fontweight='bold', color='green')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Highlight bimodal peaks in Drug A
ax2.annotate('', xy=(1, 20), xytext=(1.3, 20),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))
ax2.text(1.4, 20, 'Peak 1:\nNon-responders', fontsize=9, va='center', color='red')

ax2.annotate('', xy=(1, 35), xytext=(1.3, 35),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))
ax2.text(1.4, 35, 'Peak 2:\nResponders', fontsize=9, va='center', color='red')

plt.tight_layout()
plt.savefig('violin_vs_box_bimodal.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

---

## 4.3 Distribution: Revealing Data Structure

### Histograms: Binning Continuous Data

**When to use:**
- Showing frequency distribution of continuous data
- Revealing distribution shape (normal, skewed, bimodal)
- Understanding data spread before statistical analysis
- Sample size: moderate to large (n>30 recommended)

**Critical Decision: Bin Width**

The choice of bin width (or number of bins) **dramatically affects interpretation**—this is NOT a trivial aesthetic choice.

**The Problem:**

```
Too few bins (wide bins):
→ Over-smoothing, lose detail, miss patterns

Too many bins (narrow bins):
→ Over-granular, random noise dominates, no pattern visible

Goldilocks zone:
→ Reveals true structure without artificial patterns
```

**Bin Selection Rules:**

```
Rule of thumb formulas:
1. Sturges' formula: k = ⌈log₂(n) + 1⌉
   → Works for normal distributions, n > 30

2. Freedman-Diaconis: bin width = 2×IQR×n^(-1/3)
   → Robust to outliers, good for skewed data

3. Scott's rule: bin width = 3.49×σ×n^(-1/3)
   → Optimal for normal distributions

Modern approach: Try multiple, choose most informative
```

**Code Example (Python) - Histogram Bin Selection:**

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Generate data with known structure (bimodal)
data = np.concatenate([
    np.random.normal(20, 3, 300),  # First peak
    np.random.normal(35, 4, 200)   # Second peak
])

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Too few bins (misses bimodality)
ax1 = axes[0, 0]
ax1.hist(data, bins=5, color='#3498DB', edgecolor='black', linewidth=1.5, alpha=0.7)
ax1.set_xlabel('Value', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('❌ TOO FEW BINS (n=5)\n(Misses bimodal structure)',
             fontsize=11, fontweight='bold', color='red')
ax1.grid(axis='y', alpha=0.3)

# Too many bins (over-granular, noisy)
ax2 = axes[0, 1]
ax2.hist(data, bins=100, color='#3498DB', edgecolor='black', linewidth=0.5, alpha=0.7)
ax2.set_xlabel('Value', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('❌ TOO MANY BINS (n=100)\n(Noisy, no clear pattern)',
             fontsize=11, fontweight='bold', color='red')
ax2.grid(axis='y', alpha=0.3)

# Just right: Sturges' formula
n_sturges = int(np.ceil(np.log2(len(data)) + 1))
ax3 = axes[1, 0]
ax3.hist(data, bins=n_sturges, color='#27AE60', edgecolor='black', linewidth=1.5, alpha=0.7)
ax3.set_xlabel('Value', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title(f'✓ STURGES\' RULE (n={n_sturges})\n(Reveals bimodal structure)',
             fontsize=11, fontweight='bold', color='green')
ax3.grid(axis='y', alpha=0.3)

# Freedman-Diaconis (alternative)
iqr = np.percentile(data, 75) - np.percentile(data, 25)
bin_width_fd = 2 * iqr * len(data)**(-1/3)
n_fd = int(np.ceil((data.max() - data.min()) / bin_width_fd))
ax4 = axes[1, 1]
ax4.hist(data, bins=n_fd, color='#E67E22', edgecolor='black', linewidth=1.5, alpha=0.7)
ax4.set_xlabel('Value', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title(f'✓ FREEDMAN-DIACONIS (n={n_fd})\n(Alternative, robust to outliers)',
             fontsize=11, fontweight='bold', color='green')
ax4.grid(axis='y', alpha=0.3)

# Highlight the two peaks in correct histograms
for ax in [ax3, ax4]:
    ax.axvline(20, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(35, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(20, ax.get_ylim()[1]*0.9, 'Peak 1', ha='center', fontsize=9,
           color='red', fontweight='bold')
    ax.text(35, ax.get_ylim()[1]*0.9, 'Peak 2', ha='center', fontsize=9,
           color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('histogram_bin_selection.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()

print(f"Data: n={len(data)}")
print(f"Sturges' bins: {n_sturges}")
print(f"Freedman-Diaconis bins: {n_fd}")
```

**Code Example (R) - Histogram Bin Selection:**

```r
library(ggplot2)
library(patchwork)

set.seed(42)

# Generate bimodal data
data <- data.frame(
  value = c(rnorm(300, 20, 3), rnorm(200, 35, 4))
)

# Calculate bin numbers
n <- nrow(data)
n_sturges <- ceiling(log2(n) + 1)

iqr <- IQR(data$value)
bin_width_fd <- 2 * iqr * n^(-1/3)
n_fd <- ceiling((max(data$value) - min(data$value)) / bin_width_fd)

# Too few bins
p1 <- ggplot(data, aes(x = value)) +
  geom_histogram(bins = 5, fill = '#3498DB', color = 'black',
                size = 1, alpha = 0.7) +
  labs(x = 'Value', y = 'Frequency',
       title = paste0('❌ TOO FEW BINS (n=5)\n(Misses bimodal structure)')) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'red', size = 11),
    axis.title = element_text(face = 'bold'),
    panel.grid.major.y = element_line(color = 'gray90', size = 0.3)
  )

# Too many bins
p2 <- ggplot(data, aes(x = value)) +
  geom_histogram(bins = 100, fill = '#3498DB', color = 'black',
                size = 0.3, alpha = 0.7) +
  labs(x = 'Value', y = 'Frequency',
       title = paste0('❌ TOO MANY BINS (n=100)\n(Noisy, no clear pattern)')) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'red', size = 11),
    axis.title = element_text(face = 'bold'),
    panel.grid.major.y = element_line(color = 'gray90', size = 0.3)
  )

# Sturges' rule
p3 <- ggplot(data, aes(x = value)) +
  geom_histogram(bins = n_sturges, fill = '#27AE60', color = 'black',
                size = 1, alpha = 0.7) +
  geom_vline(xintercept = c(20, 35), color = 'red', linetype = 'dashed', size = 1, alpha = 0.7) +
  annotate('text', x = 20, y = Inf, label = 'Peak 1', vjust = 1.5, hjust = 0.5,
          color = 'red', fontface = 'bold', size = 3) +
  annotate('text', x = 35, y = Inf, label = 'Peak 2', vjust = 1.5, hjust = 0.5,
          color = 'red', fontface = 'bold', size = 3) +
  labs(x = 'Value', y = 'Frequency',
       title = paste0('✓ STURGES\' RULE (n=', n_sturges, ')\n(Reveals bimodal structure)')) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'darkgreen', size = 11),
    axis.title = element_text(face = 'bold'),
    panel.grid.major.y = element_line(color = 'gray90', size = 0.3)
  )

# Freedman-Diaconis
p4 <- ggplot(data, aes(x = value)) +
  geom_histogram(bins = n_fd, fill = '#E67E22', color = 'black',
                size = 1, alpha = 0.7) +
  geom_vline(xintercept = c(20, 35), color = 'red', linetype = 'dashed', size = 1, alpha = 0.7) +
  annotate('text', x = 20, y = Inf, label = 'Peak 1', vjust = 1.5, hjust = 0.5,
          color = 'red', fontface = 'bold', size = 3) +
  annotate('text', x = 35, y = Inf, label = 'Peak 2', vjust = 1.5, hjust = 0.5,
          color = 'red', fontface = 'bold', size = 3) +
  labs(x = 'Value', y = 'Frequency',
       title = paste0('✓ FREEDMAN-DIACONIS (n=', n_fd, ')\n(Alternative, robust to outliers)')) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'darkgreen', size = 11),
    axis.title = element_text(face = 'bold'),
    panel.grid.major.y = element_line(color = 'gray90', size = 0.3)
  )

combined <- (p1 | p2) / (p3 | p4)
ggsave('histogram_bin_selection.png', combined, width = 12, height = 10,
       dpi = 300, bg = 'white')

cat(paste0("Data: n=", n, "\n"))
cat(paste0("Sturges' bins: ", n_sturges, "\n"))
cat(paste0("Freedman-Diaconis bins: ", n_fd, "\n"))
```

---

### Density Plots: Smooth Distribution Curves

**When to use:**
- Comparing multiple distributions (overlay easier than multiple histograms)
- Emphasizing distribution shape over exact frequencies
- When you want a smoothed representation (less dependent on binning)

**Advantages over histograms:**
- No bin width decision needed
- Cleaner visual for overlapping distributions
- Better for presentations (smoother = more professional appearance)

**Disadvantages:**
- Can over-smooth and hide real features
- Requires bandwidth parameter (similar issue to bin width)
- May extend beyond actual data range (creates impossible values)

**Code Example (Python) - Density Plot Comparison:**

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(42)

# Three treatment groups with different distributions
control = np.random.normal(25, 5, 100)
drug_a = np.random.normal(32, 4, 100)
drug_b = np.random.lognormal(3.3, 0.3, 100)  # Skewed

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Overlapping histograms (cluttered)
ax1 = axes[0]
ax1.hist(control, bins=20, alpha=0.5, label='Control', color='#7F8C8D', edgecolor='black')
ax1.hist(drug_a, bins=20, alpha=0.5, label='Drug A', color='#3498DB', edgecolor='black')
ax1.hist(drug_b, bins=20, alpha=0.5, label='Drug B', color='#E74C3C', edgecolor='black')
ax1.set_xlabel('Response', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('Histograms: Overlapping (Cluttered)',
             fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', frameon=True, fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Density plots (cleaner)
ax2 = axes[1]

# Calculate density
density_control = stats.gaussian_kde(control)
density_drug_a = stats.gaussian_kde(drug_a)
density_drug_b = stats.gaussian_kde(drug_b)

x_range = np.linspace(0, 60, 300)

ax2.plot(x_range, density_control(x_range), color='#7F8C8D',
        linewidth=3, label='Control')
ax2.fill_between(x_range, density_control(x_range), alpha=0.3, color='#7F8C8D')

ax2.plot(x_range, density_drug_a(x_range), color='#3498DB',
        linewidth=3, label='Drug A')
ax2.fill_between(x_range, density_drug_a(x_range), alpha=0.3, color='#3498DB')

ax2.plot(x_range, density_drug_b(x_range), color='#E74C3C',
        linewidth=3, label='Drug B')
ax2.fill_between(x_range, density_drug_b(x_range), alpha=0.3, color='#E74C3C')

ax2.set_xlabel('Response', fontsize=11, fontweight='bold')
ax2.set_ylabel('Density', fontsize=11, fontweight='bold')
ax2.set_title('Density Plots: Smooth Comparison',
             fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', frameon=True, fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# Annotate distribution characteristics
ax2.text(32, 0.08, 'Normal\n(symmetric)', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#3498DB', alpha=0.3))
ax2.text(35, 0.05, 'Skewed right\n(long tail)', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#E74C3C', alpha=0.3))

plt.tight_layout()
plt.savefig('density_vs_histogram.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R) - Density Plot Comparison:**

```r
library(ggplot2)
library(dplyr)
library(patchwork)

set.seed(42)

# Generate data
data <- data.frame(
  value = c(
    rnorm(100, 25, 5),          # Control
    rnorm(100, 32, 4),          # Drug A
    rlnorm(100, 3.3, 0.3)       # Drug B (skewed)
  ),
  group = rep(c('Control', 'Drug A', 'Drug B'), each = 100)
)

colors <- c('Control' = '#7F8C8D', 'Drug A' = '#3498DB', 'Drug B' = '#E74C3C')

# Overlapping histograms
p1 <- ggplot(data, aes(x = value, fill = group)) +
  geom_histogram(bins = 20, alpha = 0.5, color = 'black', size = 0.5, position = 'identity') +
  scale_fill_manual(values = colors) +
  labs(x = 'Response', y = 'Frequency',
       title = 'Histograms: Overlapping (Cluttered)',
       fill = NULL) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', size = 12),
    axis.title = element_text(face = 'bold'),
    legend.position = c(0.85, 0.85),
    legend.background = element_rect(fill = 'white', color = 'black', size = 0.5),
    panel.grid.major.y = element_line(color = 'gray90', size = 0.3)
  )

# Density plots
p2 <- ggplot(data, aes(x = value, fill = group, color = group)) +
  geom_density(alpha = 0.3, size = 1.5) +
  scale_fill_manual(values = colors) +
  scale_color_manual(values = colors) +
  labs(x = 'Response', y = 'Density',
       title = 'Density Plots: Smooth Comparison',
       fill = NULL, color = NULL) +

  # Annotations
  annotate('text', x = 32, y = 0.08, label = 'Normal\n(symmetric)', hjust = 0.5,
          size = 3, lineheight = 0.9) +
  annotate('rect', xmin = 29, xmax = 35, ymin = 0.065, ymax = 0.095,
          fill = '#3498DB', alpha = 0.3) +
  annotate('text', x = 35, y = 0.05, label = 'Skewed right\n(long tail)', hjust = 0.5,
          size = 3, lineheight = 0.9) +
  annotate('rect', xmin = 32, xmax = 38, ymin = 0.035, ymax = 0.065,
          fill = '#E74C3C', alpha = 0.3) +

  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', size = 12),
    axis.title = element_text(face = 'bold'),
    legend.position = c(0.85, 0.85),
    legend.background = element_rect(fill = 'white', color = 'black', size = 0.5),
    panel.grid.major.y = element_line(color = 'gray90', size = 0.3)
  )

combined <- p1 | p2
ggsave('density_vs_histogram.png', combined, width = 14, height = 6,
       dpi = 300, bg = 'white')
```

---

## 4.4 Relationships: Scatter Plots and Correlation

### Scatter Plots: The Gold Standard for Continuous Relationships

**When to use:**
- Showing relationship between two continuous variables
- Detecting correlation, clusters, or outliers
- Visualizing raw data before statistical modeling

**Critical: Never Rely on Correlation Coefficient Alone**

**Anscombe's Quartet: The Classic Warning**

Four datasets with **identical** statistics:
- Mean of X: 9.0
- Mean of Y: 7.5
- Correlation: r = 0.816
- Linear regression: y = 3 + 0.5x

**But completely different visual patterns!**

**Code Example (Python) - Anscombe's Quartet:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Anscombe's Quartet data
anscombe = {
    'I': {
        'x': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        'y': [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
    },
    'II': {
        'x': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        'y': [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
    },
    'III': {
        'x': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        'y': [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
    },
    'IV': {
        'x': [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
        'y': [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]
    }
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (label, data) in enumerate(anscombe.items()):
    ax = axes[idx]
    x = np.array(data['x'])
    y = np.array(data['y'])

    # Scatter plot
    ax.scatter(x, y, s=100, color='#3498DB', edgecolors='black', linewidths=1.5, alpha=0.7)

    # Fit line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'y = {z[1]:.2f} + {z[0]:.2f}x')

    # Calculate correlation
    r = np.corrcoef(x, y)[0, 1]

    # Labels
    ax.set_xlabel('X', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y', fontsize=11, fontweight='bold')
    ax.set_title(f'Dataset {label}\nr = {r:.3f}, y = {z[1]:.2f} + {z[0]:.2f}x',
                fontsize=11, fontweight='bold')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', frameon=True, fontsize=9)

    # Add interpretation
    interpretations = {
        'I': 'Linear relationship\n(as expected)',
        'II': 'Nonlinear relationship\n(parabolic)',
        'III': 'Linear with outlier\n(one point distorts)',
        'IV': 'No relationship\n(one point creates false correlation)'
    }

    ax.text(0.98, 0.02, interpretations[label],
           transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.suptitle('Anscombe\'s Quartet: Same Statistics, Different Patterns\nALWAYS PLOT YOUR DATA!',
            fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('anscombes_quartet.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R) - Anscombe's Quartet:**

```r
library(ggplot2)
library(patchwork)
library(dplyr)

# Anscombe's quartet is built into R
data(anscombe)

# Reshape for plotting
anscombe_long <- data.frame(
  set = rep(c('I', 'II', 'III', 'IV'), each = 11),
  x = c(anscombe$x1, anscombe$x2, anscombe$x3, anscombe$x4),
  y = c(anscombe$y1, anscombe$y2, anscombe$y3, anscombe$y4)
)

# Calculate stats for each set
stats_df <- anscombe_long %>%
  group_by(set) %>%
  summarise(
    r = cor(x, y),
    lm_intercept = coef(lm(y ~ x))[1],
    lm_slope = coef(lm(y ~ x))[2]
  )

# Merge with data
anscombe_long <- anscombe_long %>%
  left_join(stats_df, by = 'set')

# Interpretations
interpretations <- data.frame(
  set = c('I', 'II', 'III', 'IV'),
  interpretation = c(
    'Linear relationship\n(as expected)',
    'Nonlinear relationship\n(parabolic)',
    'Linear with outlier\n(one point distorts)',
    'No relationship\n(one point creates false correlation)'
  )
)

anscombe_long <- anscombe_long %>%
  left_join(interpretations, by = 'set')

# Plot
plots <- lapply(c('I', 'II', 'III', 'IV'), function(dataset) {
  data_subset <- anscombe_long %>% filter(set == dataset)

  ggplot(data_subset, aes(x = x, y = y)) +
    geom_point(size = 4, color = '#3498DB', alpha = 0.7) +
    geom_smooth(method = 'lm', se = FALSE, color = 'red', linetype = 'dashed', size = 1.2) +

    labs(x = 'X', y = 'Y',
         title = paste0('Dataset ', dataset, '\n',
                       sprintf('r = %.3f, y = %.2f + %.2fx',
                              data_subset$r[1],
                              data_subset$lm_intercept[1],
                              data_subset$lm_slope[1]))) +

    annotate('label', x = 18, y = 2, label = unique(data_subset$interpretation),
            hjust = 1, vjust = 0, size = 3, lineheight = 0.9,
            fill = 'yellow', alpha = 0.3) +

    xlim(0, 20) +
    ylim(0, 14) +

    theme_classic(base_size = 11) +
    theme(
      plot.title = element_text(hjust = 0.5, face = 'bold', size = 10),
      axis.title = element_text(face = 'bold'),
      panel.grid.major = element_line(color = 'gray90', size = 0.3)
    )
})

combined <- (plots[[1]] | plots[[2]]) / (plots[[3]] | plots[[4]]) +
  plot_annotation(
    title = 'Anscombe\'s Quartet: Same Statistics, Different Patterns\nALWAYS PLOT YOUR DATA!',
    theme = theme(plot.title = element_text(hjust = 0.5, face = 'bold', size = 14))
  )

ggsave('anscombes_quartet.png', combined, width = 12, height = 10,
       dpi = 300, bg = 'white')
```

---

### Scatter Plot Best Practices

**1. Show ALL data points (unless n is very large)**

```
✓ GOOD: Every observation visible
→ Reveals outliers, clusters, patterns

❌ BAD: Only showing means or aggregated summaries
→ Hides individual variation and data structure
```

**2. Add transparency when points overlap**

```
# For overlapping points
plt.scatter(x, y, alpha=0.5)  # 50% transparency
```

**3. Include uncertainty/confidence intervals for trend lines**

```
# Show confidence band
sns.regplot(x='var1', y='var2', data=df, scatter_kws={'alpha':0.5})
```

**4. Report correlation AND test significance**

```
In caption:
"Pearson r = 0.73, p < 0.001, n = 50"

Not just:
"Strong correlation observed" (vague, unquantified)
```

**5. Consider scale transformations for skewed data**

```
If data spans orders of magnitude:
→ Log-transform one or both axes
→ State transformation in axis label: "log₁₀(Concentration)"
```

---

**Summary of Chapter 4 so far:**

✓ **Match plot type to data structure** (not personal preference)
✓ **Bar charts**: Start at zero, include error bars, show n
✓ **Box plots**: Show full distribution, reveal outliers
✓ **Violin plots**: Better for bimodal/complex distributions
✓ **Histograms**: Choose bin width carefully (Sturges', FD rule)
✓ **Density plots**: Cleaner for comparisons, but avoid over-smoothing
✓ **Scatter plots**: Show raw data, always plot (don't just report r)

---

## 4.5 Temporal Data: Time Series Best Practices

### Line Graphs: The Standard for Continuous Time

**When to use:**
- Showing change over continuous time
- Comparing trends across groups
- When order matters (temporal, sequential processes)

**Critical Principle: Lines Imply Continuity**

```
✓ USE LINES when:
- Data collected continuously or at regular intervals
- Interpolation between points is reasonable
- Time is truly continuous

❌ DO NOT use lines when:
- Data points are independent categories
- No meaningful values exist between points
- Comparing discrete, unordered groups
```

---

### The Multiple Time Series Challenge

**Problem:** More than 3-4 time series become visually cluttered

**Solutions:**

**Option 1: Small Multiples (Faceting)**
```
Instead of: 10 overlapping lines in one panel
Use: 10 small panels, each with one line
→ Clearer individual patterns
→ Easier to compare specific features
```

**Option 2: Highlight Key Series**
```
Background: All series in light gray (de-emphasized)
Foreground: Key series in bold color
→ Context without clutter
→ Focus on important comparison
```

**Option 3: Interactive (for digital formats)**
```
Hover/click to highlight specific series
Not viable for print, but powerful for web/presentations
```

**Code Example (Python) - Time Series Strategies:**

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Simulate 8 time series
time = np.linspace(0, 24, 100)
n_series = 8

# Generate different trends
series_data = []
for i in range(n_series):
    trend = 100 + (i-4)*5  # Different baselines
    noise = np.random.randn(100) * 3
    seasonal = 10 * np.sin(2 * np.pi * time / 24 + i * np.pi/4)
    series_data.append(trend + seasonal + noise)

labels = [f'Sample {i+1}' for i in range(n_series)]

fig = plt.figure(figsize=(16, 12))

# BAD: All series overlapping (cluttered)
ax1 = plt.subplot(3, 2, 1)
for i, data in enumerate(series_data):
    ax1.plot(time, data, linewidth=2, label=labels[i])
ax1.set_xlabel('Time (hours)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Response', fontsize=10, fontweight='bold')
ax1.set_title('❌ BAD: All Overlapping\n(Impossible to distinguish)',
             fontsize=11, fontweight='bold', color='red')
ax1.legend(loc='upper right', fontsize=7, ncol=2)
ax1.grid(alpha=0.3)

# GOOD: Small multiples
for i in range(n_series):
    ax = plt.subplot(3, 4, i+5)
    ax.plot(time, series_data[i], linewidth=2, color='#3498DB')
    ax.set_title(labels[i], fontsize=9, fontweight='bold')
    ax.set_ylim(70, 130)
    if i >= 4:
        ax.set_xlabel('Time (h)', fontsize=8)
    if i % 4 == 0:
        ax.set_ylabel('Response', fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)

# Add title for small multiples section
fig.text(0.55, 0.63, '✓ GOOD: Small Multiples\n(Each series clear)',
        fontsize=11, fontweight='bold', color='green', ha='center')

# GOOD: Highlight one series
ax3 = plt.subplot(3, 2, 2)
# Plot all in gray (background)
for i, data in enumerate(series_data):
    ax3.plot(time, data, linewidth=1, color='#CCCCCC', alpha=0.6)
# Highlight one in color
ax3.plot(time, series_data[3], linewidth=3, color='#E74C3C', label='Sample 4 (highlighted)')
ax3.set_xlabel('Time (hours)', fontsize=10, fontweight='bold')
ax3.set_ylabel('Response', fontsize=10, fontweight='bold')
ax3.set_title('✓ GOOD: Highlight Key Series\n(Context without clutter)',
             fontsize=11, fontweight='bold', color='green')
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('time_series_strategies.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R) - Time Series Strategies:**

```r
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)

set.seed(42)

# Generate data
time <- seq(0, 24, length.out = 100)
n_series <- 8

data_long <- do.call(rbind, lapply(1:n_series, function(i) {
  trend <- 100 + (i-4)*5
  noise <- rnorm(100, 0, 3)
  seasonal <- 10 * sin(2 * pi * time / 24 + i * pi/4)

  data.frame(
    time = time,
    value = trend + seasonal + noise,
    sample = paste0('Sample ', i)
  )
}))

# BAD: All overlapping
p_bad <- ggplot(data_long, aes(x = time, y = value, color = sample)) +
  geom_line(size = 1.2) +
  labs(x = 'Time (hours)', y = 'Response',
       title = '❌ BAD: All Overlapping\n(Impossible to distinguish)',
       color = NULL) +
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'red', size = 11),
    axis.title = element_text(face = 'bold'),
    legend.position = 'right',
    legend.text = element_text(size = 7),
    panel.grid.major = element_line(color = 'gray90', size = 0.3)
  )

# GOOD: Small multiples
p_small_multiples <- ggplot(data_long, aes(x = time, y = value)) +
  geom_line(color = '#3498DB', size = 1) +
  facet_wrap(~ sample, ncol = 4) +
  labs(x = 'Time (h)', y = 'Response',
       title = '✓ GOOD: Small Multiples\n(Each series clear)') +
  theme_classic(base_size = 9) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'darkgreen', size = 11),
    axis.title = element_text(face = 'bold', size = 8),
    strip.text = element_text(face = 'bold', size = 8),
    strip.background = element_rect(fill = 'gray90'),
    panel.grid.major = element_line(color = 'gray90', size = 0.2)
  )

# GOOD: Highlight key series
data_highlight <- data_long %>%
  mutate(highlight = ifelse(sample == 'Sample 4', 'Sample 4 (highlighted)', 'Other'))

p_highlight <- ggplot(data_highlight, aes(x = time, y = value, group = sample)) +
  # Background (all gray)
  geom_line(data = data_highlight %>% filter(highlight == 'Other'),
           color = '#CCCCCC', size = 0.8, alpha = 0.6) +
  # Foreground (highlighted)
  geom_line(data = data_highlight %>% filter(highlight == 'Sample 4 (highlighted)'),
           aes(color = highlight), size = 2) +
  scale_color_manual(values = c('Sample 4 (highlighted)' = '#E74C3C')) +
  labs(x = 'Time (hours)', y = 'Response',
       title = '✓ GOOD: Highlight Key Series\n(Context without clutter)',
       color = NULL) +
  theme_classic(base_size = 10) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'darkgreen', size = 11),
    axis.title = element_text(face = 'bold'),
    legend.position = c(0.8, 0.9),
    legend.background = element_rect(fill = 'white', color = 'black', size = 0.5),
    panel.grid.major = element_line(color = 'gray90', size = 0.3)
  )

# Combine
combined <- (p_bad | p_highlight) / p_small_multiples +
  plot_layout(heights = c(1, 1.5))

ggsave('time_series_strategies.png', combined, width = 16, height = 12,
       dpi = 300, bg = 'white')
```

---

### Error Bands for Time Series

**When multiple replicates exist:**

```
Options for showing variability:
1. Mean line + shaded error band (SEM or SD)
2. Mean line + error bars at key timepoints (not every point)
3. Individual traces in light color + mean in bold

✓ BEST: Shaded band (cleanest, doesn't clutter)
✓ ACCEPTABLE: Error bars at sparse intervals
❌ AVOID: Error bars at every point (cluttered)
```

**Code Example (Python) - Error Bands:**

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Simulate repeated measurements
time = np.linspace(0, 24, 50)
n_replicates = 8

# Generate replicates
replicates = []
for i in range(n_replicates):
    baseline = 100
    signal = 20 * np.sin(2 * np.pi * time / 24)
    noise = np.random.randn(len(time)) * 5
    replicates.append(baseline + signal + noise)

replicates = np.array(replicates)

# Calculate statistics
mean_signal = np.mean(replicates, axis=0)
sem_signal = np.std(replicates, axis=0) / np.sqrt(n_replicates)
sd_signal = np.std(replicates, axis=0)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# CLUTTERED: Error bars at every point
ax1 = axes[0]
ax1.errorbar(time, mean_signal, yerr=sem_signal,
            fmt='o-', color='#3498DB', linewidth=2, markersize=4,
            capsize=3, capthick=1, elinewidth=1)
ax1.set_xlabel('Time (hours)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Response', fontsize=10, fontweight='bold')
ax1.set_title('❌ CLUTTERED: Error Bars Every Point\n(Visually overwhelming)',
             fontsize=11, fontweight='bold', color='red')
ax1.grid(alpha=0.3)
ax1.set_ylim(60, 140)

# ACCEPTABLE: Error bars at sparse intervals
ax2 = axes[1]
ax2.plot(time, mean_signal, 'o-', color='#3498DB', linewidth=2.5, markersize=3)
# Error bars only every 10th point
sparse_indices = range(0, len(time), 10)
ax2.errorbar(time[sparse_indices], mean_signal[sparse_indices],
            yerr=sem_signal[sparse_indices],
            fmt='none', ecolor='black', capsize=5, capthick=2, elinewidth=2)
ax2.set_xlabel('Time (hours)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Response', fontsize=10, fontweight='bold')
ax2.set_title('✓ ACCEPTABLE: Sparse Error Bars\n(Clear but informative)',
             fontsize=11, fontweight='bold', color='green')
ax2.grid(alpha=0.3)
ax2.set_ylim(60, 140)

# BEST: Shaded error band
ax3 = axes[2]
ax3.plot(time, mean_signal, color='#3498DB', linewidth=3, label='Mean')
ax3.fill_between(time, mean_signal - sem_signal, mean_signal + sem_signal,
                color='#3498DB', alpha=0.3, label='±SEM')
ax3.set_xlabel('Time (hours)', fontsize=10, fontweight='bold')
ax3.set_ylabel('Response', fontsize=10, fontweight='bold')
ax3.set_title('✓ BEST: Shaded Error Band\n(Clean and informative)',
             fontsize=11, fontweight='bold', color='green')
ax3.legend(loc='upper right', frameon=True, fontsize=9)
ax3.grid(alpha=0.3)
ax3.set_ylim(60, 140)

# Add sample size annotation
for ax in axes:
    ax.text(0.02, 0.98, f'n={n_replicates} replicates',
           transform=ax.transAxes, fontsize=9, va='top', ha='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('time_series_error_bands.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

---

## 4.6 When NOT to Use Certain Plot Types

### Pie Charts: Generally Avoid in Scientific Figures

**Why pie charts fail:**

```
Problems:
1. Humans are bad at comparing angles (worse than lengths)
2. Hard to read exact values
3. Doesn't work well for many categories (>5)
4. 3D pie charts are even worse (perspective distortion)
5. Slices arranged by size require extra cognitive processing

Evidence: Cleveland & McGill (1984) hierarchy of visual encoding:
Position along scale > Length > Angle > Area
→ Pie charts use weakest encoding (angle)
```

**Better alternatives:**

```
Instead of pie chart, use:
✓ Bar chart (horizontal if long labels)
✓ Dot plot (when space limited)
✓ Stacked bar chart (if showing composition over time/groups)
```

**Code Example (Python) - Pie Chart vs. Bar Chart:**

```python
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

# Sample data: cell type composition
categories = ['T cells', 'B cells', 'Macrophages', 'Neutrophils', 'NK cells', 'Other']
values = [32, 18, 25, 12, 8, 5]
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#95A5A6']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart (hard to compare)
ax1 = axes[0]
wedges, texts, autotexts = ax1.pie(values, labels=categories, colors=colors,
                                    autopct='%1.1f%%', startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)
ax1.set_title('❌ PIE CHART: Hard to Compare Angles\n(Which is larger: B cells or Neutrophils?)',
             fontsize=11, fontweight='bold', color='red')

# Bar chart (easy to compare)
ax2 = axes[1]
# Sort by value for easier comparison
sorted_indices = np.argsort(values)[::-1]
sorted_categories = [categories[i] for i in sorted_indices]
sorted_values = [values[i] for i in sorted_indices]
sorted_colors = [colors[i] for i in sorted_indices]

bars = ax2.barh(sorted_categories, sorted_values, color=sorted_colors,
               edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Percentage (%)', fontsize=11, fontweight='bold')
ax2.set_title('✓ BAR CHART: Easy to Compare Lengths\n(Clear ordering: T cells > Macrophages > B cells)',
             fontsize=11, fontweight='bold', color='green')
ax2.invert_yaxis()  # Highest at top
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, sorted_values)):
    ax2.text(val + 0.5, i, f'{val}%', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('pie_vs_bar_comparison.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

**Code Example (R) - Pie Chart vs. Bar Chart:**

```r
library(ggplot2)
library(patchwork)

# Data
data <- data.frame(
  category = factor(c('T cells', 'B cells', 'Macrophages', 'Neutrophils', 'NK cells', 'Other')),
  value = c(32, 18, 25, 12, 8, 5)
)

colors <- c('T cells' = '#E74C3C', 'B cells' = '#3498DB',
            'Macrophages' = '#2ECC71', 'Neutrophils' = '#F39C12',
            'NK cells' = '#9B59B6', 'Other' = '#95A5A6')

# Pie chart (ggplot doesn't encourage pies, but we'll make one to show why they're bad)
p_pie <- ggplot(data, aes(x = "", y = value, fill = category)) +
  geom_bar(stat = "identity", width = 1, color = 'black', size = 1) +
  coord_polar("y", start = 0) +
  geom_text(aes(label = paste0(value, '%')),
            position = position_stack(vjust = 0.5),
            color = 'white', fontface = 'bold', size = 3.5) +
  scale_fill_manual(values = colors) +
  labs(title = '❌ PIE CHART: Hard to Compare Angles\n(Which is larger: B cells or Neutrophils?)',
       fill = NULL) +
  theme_void() +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'red', size = 11),
    legend.position = 'right'
  )

# Bar chart (sorted for clarity)
data_sorted <- data %>%
  arrange(desc(value)) %>%
  mutate(category = factor(category, levels = category))

p_bar <- ggplot(data_sorted, aes(x = category, y = value, fill = category)) +
  geom_bar(stat = 'identity', color = 'black', size = 1, width = 0.7) +
  geom_text(aes(label = paste0(value, '%')),
           vjust = -0.5, fontface = 'bold', size = 3.5) +
  scale_fill_manual(values = colors) +
  labs(y = 'Percentage (%)',
       title = '✓ BAR CHART: Easy to Compare Lengths\n(Clear ordering: T cells > Macrophages > B cells)') +
  ylim(0, 40) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = 'bold', color = 'darkgreen', size = 11),
    axis.title.x = element_blank(),
    axis.title.y = element_text(face = 'bold'),
    legend.position = 'none',
    panel.grid.major.y = element_line(color = 'gray90', size = 0.3),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

combined <- p_pie | p_bar
ggsave('pie_vs_bar_comparison.png', combined, width = 14, height = 6,
       dpi = 300, bg = 'white')
```

---

### Dual Y-Axes: Use with Extreme Caution

**Why dual y-axes are problematic:**

```
Problems:
1. Scale manipulation can create false correlations
2. Cognitively demanding (which line matches which axis?)
3. Can be used to mislead (intentionally or accidentally)
4. No standard for which variable goes on which axis

Example of manipulation:
By adjusting scale ranges, you can make ANY two variables
appear correlated or anti-correlated
```

**When dual y-axes ARE acceptable:**

```
✓ Same variable, different units (e.g., °C and °F)
✓ Tightly coupled variables (input/output of same process)
✓ When separate panels would lose temporal alignment

BUT: Always justify in caption why dual axes are necessary
```

**Better alternatives:**

```
✓ Two separate panels (stacked vertically, aligned time axes)
✓ Normalize both variables to 0-100% scale
✓ Show correlation as scatter plot instead
```

**Code Example (Python) - Dual Axes Problem:**

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Two unrelated variables
time = np.linspace(0, 10, 100)
variable_1 = 50 + 10*np.sin(time) + np.random.randn(100)*2
variable_2 = 200 + 50*np.sin(time + 1.5) + np.random.randn(100)*5

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# MISLEADING: Dual axes with manipulated scales
ax1 = axes[0]
color1 = '#E74C3C'
color2 = '#3498DB'

# Plot variable 1
l1 = ax1.plot(time, variable_1, color=color1, linewidth=2.5, label='Variable 1')
ax1.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Variable 1 (units)', color=color1, fontsize=11, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(20, 80)  # Manipulated range

# Plot variable 2 on second y-axis
ax1_twin = ax1.twinx()
l2 = ax1_twin.plot(time, variable_2, color=color2, linewidth=2.5, label='Variable 2')
ax1_twin.set_ylabel('Variable 2 (units)', color=color2, fontsize=11, fontweight='bold')
ax1_twin.tick_params(axis='y', labelcolor=color2)
ax1_twin.set_ylim(120, 280)  # Manipulated to make lines appear correlated

ax1.set_title('❌ MISLEADING: Dual Y-Axes\n(Scales manipulated to suggest correlation)',
             fontsize=11, fontweight='bold', color='red')
ax1.grid(alpha=0.3)

# Add warning annotation
ax1.text(0.5, 0.5, 'WARNING:\nScales can be adjusted\nto create false patterns!',
         transform=ax1.transAxes, ha='center', va='center',
         fontsize=10, fontweight='bold', color='red',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# BETTER: Separate panels (aligned)
ax2 = axes[1]

# Normalize both to 0-100 scale for fair comparison
var1_norm = (variable_1 - variable_1.min()) / (variable_1.max() - variable_1.min()) * 100
var2_norm = (variable_2 - variable_2.min()) / (variable_2.max() - variable_2.min()) * 100

ax2.plot(time, var1_norm, color=color1, linewidth=2.5, label='Variable 1 (normalized)')
ax2.plot(time, var2_norm, color=color2, linewidth=2.5, label='Variable 2 (normalized)')
ax2.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Normalized Value (0-100%)', fontsize=11, fontweight='bold')
ax2.set_title('✓ BETTER: Normalized to Same Scale\n(Honest comparison)',
             fontsize=11, fontweight='bold', color='green')
ax2.legend(loc='upper right', frameon=True, fontsize=10)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('dual_axes_problem.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

---

### 3D Plots: Almost Always Avoid

**Why 3D plots fail in publications:**

```
Problems:
1. Perspective distortion (closer points look bigger)
2. Occlusion (front objects hide back objects)
3. Difficult to read exact values
4. Rotation angle is arbitrary (different angles = different story)
5. Doesn't print well (loses depth cues)

The 3D illusion only works on screen with rotation interaction
```

**When 3D is acceptable:**

```
✓ 3D molecular structures (actual 3D objects)
✓ Medical imaging (CT/MRI reconstructions)
✓ Spatial mapping (genuine 3D physical phenomena)

❌ NOT for: Bar charts, pie charts, scatter plots (use 2D instead)
```

**Alternatives to 3D scatter plots:**

```
Instead of 3D scatter:
✓ Color/size as 3rd dimension in 2D plot
✓ Multiple 2D projections (XY, XZ, YZ panels)
✓ Interactive HTML plots (for digital only)
✓ Contour plots or heatmaps
```

---

### The "Chartjunk" Principle

**Edward Tufte's concept: Maximize data-ink ratio**

```
Data-ink ratio = Ink used for data / Total ink used

✓ GOOD: High ratio (most ink shows data)
❌ BAD: Low ratio (decorations dominate)

Examples of chartjunk to avoid:
- 3D effects on 2D data
- Gradients/textures on bars (use solid colors)
- Grid lines that overwhelm data
- Excessive decorative elements
- Unnecessary backgrounds
- Ornamental fonts
```

**Modern Minimalism in Scientific Figures:**

```
Keep:
✓ Data points/lines
✓ Axes and labels
✓ Legend (if necessary)
✓ Minimal grid (light, unobtrusive)
✓ Error bars/confidence bands

Remove:
❌ Chart borders (top/right spines)
❌ Background colors (use white)
❌ 3D effects
❌ Drop shadows
❌ Decorative clip art
```
---

### **4.7 Bar Plot Error Bars: Correct Usage**

**Critical Rule:** Error bars must be scientifically justified and clearly labeled.

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Example data
conditions = ['Control', 'Treatment A', 'Treatment B']
n_replicates = 5

# Simulate biological replicates
data = {
    'Control': np.random.normal(25, 3, n_replicates),
    'Treatment A': np.random.normal(35, 4, n_replicates),
    'Treatment B': np.random.normal(30, 3.5, n_replicates)
}

means = [np.mean(data[c]) for c in conditions]
stds = [np.std(data[c], ddof=1) for c in conditions]  # Sample SD
sems = [np.std(data[c], ddof=1) / np.sqrt(len(data[c])) for c in conditions]  # SEM

# Panel A: BAD - No error bars 
ax1.text(0.5, 0.95, 'No variability shown!\nHow reliable is this?',
        transform=ax1.transAxes, ha='center', va='top',
        fontsize=10, style='italic', color='red',
        bbox=dict(boxstyle='round', facecolor='#FFCCCC', alpha=0.8))

# Panel B: UNCLEAR - Error bars without label
ax2 = axes[0, 1]
ax2.bar(conditions, means, yerr=stds, capsize=8,
       color='#3498DB', edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Response (AU)', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 50)
ax2.set_title('❌ UNCLEAR: Unlabeled Error Bars\n(SD? SEM? 95% CI?)',
              fontsize=12, fontweight='bold', color='red')

# Panel C: GOOD - SD with label
ax3 = axes[1, 0]
ax3.bar(conditions, means, yerr=stds, capsize=8,
       color='#3498DB', edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Response (AU)\n(Mean ± SD)', fontsize=11, fontweight='bold')
ax3.set_ylim(0, 50)
ax3.set_title('✓ GOOD: Labeled as SD\n(Shows data spread)',
              fontsize=12, fontweight='bold', color='green')

# Add sample sizes
for i, (cond, mean, std) in enumerate(zip(conditions, means, stds)):
    ax3.text(i, mean + std + 2, f'n={n_replicates}',
            ha='center', fontsize=9, style='italic')

# Panel D: ALSO GOOD - SEM with label
ax4 = axes[1, 1]
ax4.bar(conditions, means, yerr=sems, capsize=8,
       color='#27AE60', edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Response (AU)\n(Mean ± SEM)', fontsize=11, fontweight='bold')
ax4.set_ylim(0, 50)
ax4.set_title('✓ ALSO GOOD: Labeled as SEM\n(Shows precision of mean)',
              fontsize=12, fontweight='bold', color='green')

# Add sample sizes
for i, (cond, mean, sem) in enumerate(zip(conditions, means, sems)):
    ax4.text(i, mean + sem + 2, f'n={n_replicates}',
            ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('error_bars_correct_usage.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

**Decision Tree: Which Error Bar to Use?**

```python
"""
WHEN TO USE EACH ERROR BAR TYPE:

1. STANDARD DEVIATION (SD):
   ✓ Shows spread/variability of the data
   ✓ Use when: Describing population variability is important
   ✓ Interpretation: ~68% of data falls within ±1 SD
   ✓ Example: "Control group has high variability (SD=8)"

2. STANDARD ERROR OF THE MEAN (SEM):
   ✓ Shows precision of the mean estimate
   ✓ Use when: Comparing means between groups
   ✓ Interpretation: Narrower bars = more precise mean
   ✓ Formula: SEM = SD / sqrt(n)
   ✓ Note: ALWAYS report n (sample size)

3. 95% CONFIDENCE INTERVAL:
   ✓ Shows range likely to contain true mean
   ✓ Use when: Making statistical inference
   ✓ Interpretation: If bars don't overlap, likely significant difference
   ✓ Formula: CI = mean ± (t_critical × SEM)

GOLDEN RULE: ALWAYS LABEL WHICH TYPE YOU'RE USING!
"""

# Code example with all three types
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(42)

conditions = ['Control', 'Treatment']
n = 10  # Sample size
data_ctrl = np.random.normal(50, 10, n)
data_trt = np.random.normal(65, 12, n)

means = [data_ctrl.mean(), data_trt.mean()]
sds = [data_ctrl.std(ddof=1), data_trt.std(ddof=1)]
sems = [data_ctrl.std(ddof=1)/np.sqrt(n), data_trt.std(ddof=1)/np.sqrt(n)]

# Calculate 95% CI
t_crit = stats.t.ppf(0.975, df=n-1)  # Two-tailed, df=n-1
ci95 = [sem * t_crit for sem in sems]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: SD
ax1 = axes[0]
ax1.bar(conditions, means, yerr=sds, capsize=10,
       color='#3498DB', edgecolor='black', linewidth=2, width=0.5)
ax1.set_ylabel('Response (AU)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.set_title('Standard Deviation (SD)\nShows data spread',
              fontsize=13, fontweight='bold')
ax1.text(0.5, 0.95, f'n={n} per group', transform=ax1.transAxes,
        ha='center', va='top', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Plot 2: SEM
ax2 = axes[1]
ax2.bar(conditions, means, yerr=sems, capsize=10,
       color='#27AE60', edgecolor='black', linewidth=2, width=0.5)
ax2.set_ylabel('Response (AU)', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.set_title('Standard Error of Mean (SEM)\nShows precision of mean',
              fontsize=13, fontweight='bold')
ax2.text(0.5, 0.95, f'n={n} per group', transform=ax2.transAxes,
        ha='center', va='top', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Plot 3: 95% CI
ax3 = axes[2]
ax3.bar(conditions, means, yerr=ci95, capsize=10,
       color='#E74C3C', edgecolor='black', linewidth=2, width=0.5)
ax3.set_ylabel('Response (AU)', fontsize=12, fontweight='bold')
ax3.set_ylim(0, 100)
ax3.set_title('95% Confidence Interval\nLikely range of true mean',
              fontsize=13, fontweight='bold')
ax3.text(0.5, 0.95, f'n={n} per group', transform=ax3.transAxes,
        ha='center', va='top', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Add comparison note
fig.text(0.5, 0.02, 'Note: Same data, different error bar types → Different visual impression',
        ha='center', fontsize=11, style='italic', fontweight='bold')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('error_bar_types_comparison.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

**Critical Mistake: Using Bar Plots When You Shouldn't**

```python
# Bar plots are often MISUSED - here are better alternatives

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Simulate data with different distributions
control = np.random.normal(50, 10, 100)
treatment = np.concatenate([
    np.random.normal(40, 5, 50),   # Bimodal distribution!
    np.random.normal(70, 5, 50)
])

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# BAD: Bar plot hides bimodal distribution
ax1 = axes[0, 0]
means = [control.mean(), treatment.mean()]
stds = [control.std(), treatment.std()]
conditions = ['Control', 'Treatment']

ax1.bar(conditions, means, yerr=stds, capsize=8,
       color='#3498DB', edgecolor='black', linewidth=2)
ax1.set_ylabel('Response (AU)', fontsize=11, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.set_title('❌ BAD: Bar Plot\n(Hides important distribution shape)',
              fontsize=12, fontweight='bold', color='red')

# BETTER: Box plot shows distribution
ax2 = axes[0, 1]
bp = ax2.boxplot([control, treatment], labels=conditions,
                 patch_artist=True, widths=0.5)
for patch in bp['boxes']:
    patch.set_facecolor('#3498DB')
    patch.set_alpha(0.7)
ax2.set_ylabel('Response (AU)', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.set_title('✓ BETTER: Box Plot\n(Shows quartiles, outliers)',
              fontsize=12, fontweight='bold', color='green')
ax2.grid(axis='y', alpha=0.3)

# EVEN BETTER: Violin plot shows full distribution
ax3 = axes[1, 0]
parts = ax3.violinplot([control, treatment], positions=[1, 2],
                       widths=0.7, showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('#3498DB')
    pc.set_alpha(0.7)
ax3.set_xticks([1, 2])
ax3.set_xticklabels(conditions)
ax3.set_ylabel('Response (AU)', fontsize=11, fontweight='bold')
ax3.set_ylim(0, 100)
ax3.set_title('✓ EVEN BETTER: Violin Plot\n(Shows BIMODAL distribution in treatment!)',
              fontsize=12, fontweight='bold', color='green')
ax3.grid(axis='y', alpha=0.3)

# BEST: Show individual points + summary
ax4 = axes[1, 1]
# Scatter individual points with jitter
x_ctrl = np.random.normal(1, 0.04, len(control))
x_trt = np.random.normal(2, 0.04, len(treatment))
ax4.scatter(x_ctrl, control, alpha=0.4, s=30, color='#3498DB', edgecolors='black', linewidths=0.5)
ax4.scatter(x_trt, treatment, alpha=0.4, s=30, color='#E74C3C', edgecolors='black', linewidths=0.5)

# Overlay mean ± SEM
ax4.errorbar([1, 2], means, yerr=[s/np.sqrt(100) for s in stds],
            fmt='D', markersize=12, color='black', markerfacecolor='yellow',
            capsize=10, linewidth=3, label='Mean ± SEM')

ax4.set_xticks([1, 2])
ax4.set_xticklabels(conditions)
ax4.set_xlim(0.5, 2.5)
ax4.set_ylabel('Response (AU)', fontsize=11, fontweight='bold')
ax4.set_ylim(0, 100)
ax4.set_title('✓ BEST: Individual Points + Summary\n(Shows BOTH raw data and summary)',
              fontsize=12, fontweight='bold', color='green')
ax4.legend(loc='upper left', frameon=True, fontsize=10)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('when_not_to_use_barplot.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()
```

---

---

## Chapter 4 Summary: Complete Graph Selection Guide

**Quick Reference Table:**

| Data Type | Best Plot | Acceptable Alternatives | Avoid |
|-----------|-----------|------------------------|-------|
| **Compare groups (continuous)** | Bar chart, Box plot, Violin plot | Dot plot, Strip plot | Pie chart, 3D bars |
| **Compare groups (many)** | Small multiples, Heatmap | Grouped violin, Faceted box | Single cluttered plot |
| **Distribution (single)** | Histogram, Density, Box plot | Violin plot, ECDF | Pie chart (not applicable) |
| **Distribution (multiple)** | Overlapping density, Violin, Box | Ridgeline plot | Multiple pie charts |
| **Relationship (continuous)** | Scatter plot | Hexbin (if dense), 2D density | Line plot (if not ordered) |
| **Time series (few)** | Line graph | Area chart | Bar chart, Pie chart |
| **Time series (many)** | Small multiples, Highlight key | Heatmap, Spaghetti plot | All overlapping |
| **Composition (parts of whole)** | Stacked bar, Treemap | Stacked area (over time) | Pie chart |
| **Spatial data** | Map (choropleth/point), Heatmap | Contour plot | 3D surface (in print) |
| **Correlation matrix** | Heatmap, Corrplot | Network diagram | Table only |

---

**End of Chapter 4: Data Encoding & Graph Selection**

**Key Takeaways:**
- **Match plot to data structure**, not personal preference
- **Bar charts**: Zero baseline, error bars, sorted logically
- **Box/Violin plots**: Show full distribution, not just mean
- **Histograms**: Bin width matters (use Sturges' or FD rule)
- **Scatter plots**: Always plot raw data (Anscombe's warning)
- **Time series**: Use lines for continuous, small multiples for many
- **Avoid**: Pie charts, dual y-axes (unless justified), 3D effects
- **Maximize data-ink ratio**: Remove chartjunk, keep focus on data

---