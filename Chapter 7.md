# Chapter 7: Common Figure Types Deep Dive

## 7.1 Heatmaps: Visualizing Matrix Data

### When to Use Heatmaps

**Ideal scenarios:**
```
✓ Gene expression matrices (genes × samples)
✓ Correlation matrices (variables × variables)
✓ Spatial patterns (location × location)
✓ Time-course data (features × time points)
✓ Any large matrix where patterns > individual values
```

**When NOT to use:**
```
❌ Few data points (<10×10) — use bar charts or tables
❌ When exact values are critical — add numbers or use alternative
❌ Categorical data without ordering — consider other encodings
```

---

### Critical Design Decisions for Heatmaps

**Decision 1: Colormap Selection**

```
Type of data → Colormap choice

Sequential (one direction, e.g., 0 to max):
✓ Use: Viridis, Plasma, YlOrRd, Blues
Example: Gene expression (FPKM 0-1000)

Diverging (two directions from center):
✓ Use: RdBu (red-blue), RdYlGn, PiYG
Example: Fold change (log₂ -3 to +3, center at 0)

Categorical (unordered groups):
✓ Use: Distinct hues, equal saturation
Example: Cluster assignments
```

**Decision 2: Normalization Strategy**

```
Row normalization (Z-score by gene):
- Highlights relative patterns across samples
- Each row mean=0, SD=1
- Use when: Comparing patterns, not absolute levels

Column normalization (by sample):
- Highlights relative patterns across genes
- Use when: Sample-to-sample differences are key

No normalization (raw values):
- Shows absolute magnitudes
- Use when: Actual values matter (e.g., concentrations)
```

**Decision 3: Clustering and Ordering**

```
Hierarchical clustering:
✓ Reveals groups/patterns automatically
✓ Shows dendrogram (tree structure)
✓ Use: Exploratory analysis

Manual ordering:
✓ Test specific hypotheses
✓ Control presentation order
✓ Use: Confirmatory analysis, known groups

Ordered by value:
✓ Simple sorting (high to low)
✓ Use: Ranking, prioritization
```

---

**Code Example (Python) - Comprehensive Heatmap:**

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

np.random.seed(42)

# Generate synthetic gene expression data
n_genes = 50
n_samples = 12

# Three sample groups with different expression patterns
group1 = np.random.normal(5, 1, (n_genes, 4))
group2 = np.random.normal(8, 1.5, (n_genes, 4))
group3 = np.random.normal(6, 1, (n_genes, 4))

# Some genes upregulated in group2
group2[:15, :] += 3

data = np.hstack([group1, group2, group3])

# Sample and gene names
genes = [f'Gene{i+1}' for i in range(n_genes)]
samples = [f'Ctrl{i+1}' for i in range(4)] + \
          [f'TrtA{i+1}' for i in range(4)] + \
          [f'TrtB{i+1}' for i in range(4)]

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# Layout: Main heatmap + dendrograms + colorbars
gs = fig.add_gridspec(3, 3,
                      width_ratios=[0.5, 4, 0.15],  # Dendrogram | Heatmap | Colorbar
                      height_ratios=[0.5, 4, 0.15],  # Dendrogram | Heatmap | Sample labels
                      hspace=0.02, wspace=0.02)

# === PANEL 1: Basic heatmap (no clustering) ===
ax_basic = fig.add_subplot(3, 3, 1)
sns.heatmap(data[:20, :6], cmap='viridis', cbar=False, ax=ax_basic,
            xticklabels=samples[:6], yticklabels=genes[:20])
ax_basic.set_title('A. Basic Heatmap\n(No clustering)',
                   fontsize=11, fontweight='bold', pad=10)
ax_basic.set_xlabel('')
ax_basic.set_ylabel('Genes', fontsize=10, fontweight='bold')

# === PANEL 2: Hierarchical clustering ===
ax_clust = fig.add_subplot(3, 3, 2)

# Perform hierarchical clustering
row_linkage = linkage(data, method='average', metric='euclidean')
col_linkage = linkage(data.T, method='average', metric='euclidean')

# Plot with clustered rows and columns
sns.clustermap(data, cmap='RdBu_r', center=6.5,
               row_linkage=row_linkage, col_linkage=col_linkage,
               figsize=(8, 10), cbar_pos=(0.02, 0.8, 0.03, 0.15),
               dendrogram_ratio=0.15,
               xticklabels=samples, yticklabels=False)

# Note: clustermap creates its own figure, so we save separately
plt.savefig('heatmap_clustered.png', dpi=300, bbox_inches='tight')
plt.close()

# === PANEL 3: With annotations ===
ax_annot = fig.add_subplot(3, 3, 3)

# Normalized data (Z-score by row)
data_norm = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)

sns.heatmap(data_norm[:20, :6], cmap='RdBu_r', center=0,
            cbar=True, ax=ax_annot,
            xticklabels=samples[:6], yticklabels=False,
            cbar_kws={'label': 'Z-score'})
ax_annot.set_title('C. Row-Normalized\n(Z-score)',
                   fontsize=11, fontweight='bold', pad=10)
ax_annot.set_xlabel('Samples', fontsize=10, fontweight='bold')

# Add sample group annotations
for i in range(6):
    if i < 2:
        color = '#7F8C8D'  # Control
    elif i < 4:
        color = '#3498DB'  # Treatment A
    else:
        color = '#E74C3C'  # Treatment B

    ax_annot.add_patch(plt.Rectangle((i, -1), 1, 0.5,
                                     facecolor=color, edgecolor='black', linewidth=2,
                                     clip_on=False))

# === PANEL 4: Common mistakes ===
ax_bad = fig.add_subplot(3, 3, 4)

# Bad: Rainbow colormap (not perceptually uniform)
sns.heatmap(data[:20, :6], cmap='jet', cbar=False, ax=ax_bad,
            xticklabels=samples[:6], yticklabels=False)
ax_bad.set_title('❌ BAD: Rainbow Colormap\n(Not perceptually uniform)',
                 fontsize=11, fontweight='bold', color='red', pad=10)
ax_bad.set_xlabel('Samples', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('heatmap_comprehensive.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Heatmap examples created:")
print("1. heatmap_comprehensive.png - Multiple panels showing best practices")
print("2. heatmap_clustered.png - Full hierarchical clustering example")
```

**Code Example (R) - Comprehensive Heatmap:**

```r
library(ggplot2)
library(pheatmap)
library(RColorBrewer)
library(viridis)

set.seed(42)

# Generate synthetic data
n_genes <- 50
n_samples <- 12

group1 <- matrix(rnorm(n_genes * 4, mean = 5, sd = 1), n_genes, 4)
group2 <- matrix(rnorm(n_genes * 4, mean = 8, sd = 1.5), n_genes, 4)
group3 <- matrix(rnorm(n_genes * 4, mean = 6, sd = 1), n_genes, 4)

# Upregulate some genes in group2
group2[1:15, ] <- group2[1:15, ] + 3

data <- cbind(group1, group2, group3)

# Names
rownames(data) <- paste0('Gene', 1:n_genes)
colnames(data) <- c(paste0('Ctrl', 1:4),
                    paste0('TrtA', 1:4),
                    paste0('TrtB', 1:4))

# Sample annotations
sample_groups <- data.frame(
  Group = factor(rep(c('Control', 'Treatment A', 'Treatment B'), each = 4))
)
rownames(sample_groups) <- colnames(data)

# Annotation colors
ann_colors <- list(
  Group = c('Control' = '#7F8C8D',
            'Treatment A' = '#3498DB',
            'Treatment B' = '#E74C3C')
)

# === HEATMAP 1: Basic (no clustering) ===
png('heatmap_basic.png', width = 2400, height = 2000, res = 300)
pheatmap(data[1:20, 1:6],
         color = viridis(100),
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         main = 'Basic Heatmap (No clustering)',
         fontsize = 10,
         fontsize_row = 8,
         fontsize_col = 9)
dev.off()

# === HEATMAP 2: Hierarchical clustering with annotations ===
png('heatmap_clustered_annotated.png', width = 2800, height = 3000, res = 300)
pheatmap(data,
         color = colorRampPalette(rev(brewer.pal(n = 7, name = "RdBu")))(100),
         scale = "row",  # Z-score normalization
         clustering_distance_rows = "euclidean",
         clustering_distance_cols = "euclidean",
         clustering_method = "average",
         annotation_col = sample_groups,
         annotation_colors = ann_colors,
         show_rownames = FALSE,
         fontsize = 10,
         fontsize_col = 9,
         main = 'Hierarchical Clustering with Annotations')
dev.off()

# === HEATMAP 3: Row-normalized (Z-score) ===
data_norm <- t(scale(t(data)))  # Z-score by row

png('heatmap_normalized.png', width = 2800, height = 3000, res = 300)
pheatmap(data_norm,
         color = colorRampPalette(rev(brewer.pal(n = 7, name = "RdBu")))(100),
         breaks = seq(-3, 3, length.out = 101),  # Symmetric scale
         cluster_rows = TRUE,
         cluster_cols = TRUE,
         annotation_col = sample_groups,
         annotation_colors = ann_colors,
         show_rownames = FALSE,
         fontsize = 10,
         fontsize_col = 9,
         main = 'Row-Normalized (Z-score)')
dev.off()

# === HEATMAP 4: BAD EXAMPLE (rainbow colormap) ===
png('heatmap_bad_rainbow.png', width = 2400, height = 2000, res = 300)
pheatmap(data[1:20, 1:6],
         color = rainbow(100),  # BAD: Rainbow colormap
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         main = '❌ BAD: Rainbow Colormap (Not perceptually uniform)',
         fontsize = 10,
         fontsize_row = 8,
         fontsize_col = 9)
dev.off()

cat("Heatmap examples created:\n")
cat("1. heatmap_basic.png\n")
cat("2. heatmap_clustered_annotated.png\n")
cat("3. heatmap_normalized.png\n")
cat("4. heatmap_bad_rainbow.png\n")
```

---

### Heatmap Best Practices Checklist

```
Before creating heatmap:
 Decide normalization strategy (raw, row Z-score, column Z-score)
 Choose appropriate colormap (sequential vs. diverging)
 Determine clustering method (or manual ordering)
 Set symmetric scale if diverging (e.g., -3 to +3)

Visual elements:
 Colorbar with clear label and units
 Sample/row annotations if grouped
 Dendrogram if using hierarchical clustering
 Grid lines (subtle) if helpful for reading specific cells

Avoid:
❌ Rainbow colormap (jet, hsv)
❌ Asymmetric scales for diverging data
❌ Too many rows/columns (>100 makes individual cells unreadable)
❌ Missing colorbar or unlabeled colorbar
```

---

## 7.2 Volcano Plots: Differential Expression Analysis

### What Volcano Plots Show

**Purpose:** Simultaneously visualize:
1. **Magnitude of change** (x-axis: fold change)
2. **Statistical significance** (y-axis: -log₁₀(p-value))

**Common in:** Genomics, proteomics, metabolomics

---

### Anatomy of a Volcano Plot

```
Structure:
                  High significance
                         ↑
                         │
  Down-regulated    │    Up-regulated
  (significant)     │    (significant)
        ←───────────┼───────────→
     -log₂FC    Center   +log₂FC
                    (0)
                         │
                         ↓
                  Low significance

Quadrants:
- Top-left: Down-regulated + significant
- Top-right: Up-regulated + significant
- Bottom: Not significant (regardless of fold change)
```

---

**Code Example (Python) - Volcano Plot:**

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Simulate differential expression data
n_genes = 2000

# Log2 fold changes (mostly around 0, some large changes)
log2fc = np.random.normal(0, 1.5, n_genes)
log2fc[:50] += 3  # 50 upregulated genes
log2fc[50:100] -= 3  # 50 downregulated genes

# P-values (mostly high, some low for significant genes)
p_values = np.random.uniform(0.01, 1, n_genes)
p_values[:100] = np.random.uniform(0.0001, 0.01, 100)  # Significant genes

# Convert to -log10(p)
neg_log10_p = -np.log10(p_values)

# Define significance thresholds
fc_threshold = 1.0  # Log2 fold change threshold
p_threshold = 0.05  # P-value threshold
neg_log10_p_threshold = -np.log10(p_threshold)

# Classify genes
significant_up = (log2fc > fc_threshold) & (p_values < p_threshold)
significant_down = (log2fc < -fc_threshold) & (p_values < p_threshold)
not_significant = ~(significant_up | significant_down)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# === PANEL A: Basic volcano plot ===
ax1 = axes[0]

# Plot not significant genes (gray, in background)
ax1.scatter(log2fc[not_significant], neg_log10_p[not_significant],
           s=15, color='#CCCCCC', alpha=0.5, label='Not significant')

# Plot significant down-regulated (blue)
ax1.scatter(log2fc[significant_down], neg_log10_p[significant_down],
           s=30, color='#3498DB', alpha=0.8, edgecolors='black', linewidths=0.5,
           label=f'Down ({np.sum(significant_down)})')

# Plot significant up-regulated (red)
ax1.scatter(log2fc[significant_up], neg_log10_p[significant_up],
           s=30, color='#E74C3C', alpha=0.8, edgecolors='black', linewidths=0.5,
           label=f'Up ({np.sum(significant_up)})')

# Add threshold lines
ax1.axhline(neg_log10_p_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.axvline(fc_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.axvline(-fc_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

# Labels
ax1.set_xlabel('Log₂ Fold Change', fontsize=12, fontweight='bold')
ax1.set_ylabel('-Log₁₀ (P-value)', fontsize=12, fontweight='bold')
ax1.set_title('A. Volcano Plot: Differential Expression',
             fontsize=13, fontweight='bold')

# Add threshold annotations
ax1.text(fc_threshold + 0.2, ax1.get_ylim()[0] + 0.5,
        f'FC > {2**fc_threshold:.1f}×', fontsize=9, rotation=90, va='bottom')
ax1.text(-fc_threshold - 0.2, ax1.get_ylim()[0] + 0.5,
        f'FC < {2**-fc_threshold:.1f}×', fontsize=9, rotation=90, va='bottom', ha='right')
ax1.text(ax1.get_xlim()[0] + 0.5, neg_log10_p_threshold + 0.3,
        f'p < {p_threshold}', fontsize=9)

ax1.legend(loc='upper right', frameon=True, fontsize=10)
ax1.grid(alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# === PANEL B: Common mistakes ===
ax2 = axes[1]

# BAD: No fold change threshold (only p-value)
significant_p_only = p_values < p_threshold

ax2.scatter(log2fc[~significant_p_only], neg_log10_p[~significant_p_only],
           s=15, color='#CCCCCC', alpha=0.5, label='Not significant')
ax2.scatter(log2fc[significant_p_only], neg_log10_p[significant_p_only],
           s=30, color='#E74C3C', alpha=0.8, edgecolors='black', linewidths=0.5,
           label=f'p < {p_threshold} (ignoring FC)')

# Only p-value threshold line
ax2.axhline(neg_log10_p_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

ax2.set_xlabel('Log₂ Fold Change', fontsize=12, fontweight='bold')
ax2.set_ylabel('-Log₁₀ (P-value)', fontsize=12, fontweight='bold')
ax2.set_title('❌ B. BAD: No Fold Change Threshold\n(Includes small, insignificant changes)',
             fontsize=13, fontweight='bold', color='red')

ax2.legend(loc='upper right', frameon=True, fontsize=10)
ax2.grid(alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('volcano_plot_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Print summary statistics
print("Volcano Plot Summary:")
print(f"Total genes: {n_genes}")
print(f"Significant up-regulated: {np.sum(significant_up)}")
print(f"Significant down-regulated: {np.sum(significant_down)}")
print(f"Not significant: {np.sum(not_significant)}")
print(f"\nThresholds:")
print(f"- Fold change: ±{2**fc_threshold:.2f}× (log₂ ±{fc_threshold})")
print(f"- P-value: {p_threshold} (-log₁₀ = {neg_log10_p_threshold:.2f})")
```

**Code Example (R) - Volcano Plot:**

```r
library(ggplot2)
library(ggrepel)

set.seed(42)

# Simulate data
n_genes <- 2000

log2fc <- rnorm(n_genes, 0, 1.5)
log2fc[1:50] <- log2fc[1:50] + 3  # Upregulated
log2fc[51:100] <- log2fc[51:100] - 3  # Downregulated

p_values <- runif(n_genes, 0.01, 1)
p_values[1:100] <- runif(100, 0.0001, 0.01)  # Significant

neg_log10_p <- -log10(p_values)

# Thresholds
fc_threshold <- 1.0
p_threshold <- 0.05
neg_log10_p_threshold <- -log10(p_threshold)

# Create dataframe
data <- data.frame(
  gene = paste0('Gene', 1:n_genes),
  log2fc = log2fc,
  neg_log10_p = neg_log10_p,
  p_value = p_values
)

# Classify
data$status <- 'Not significant'
data$status[data$log2fc > fc_threshold & data$p_value < p_threshold] <- 'Up-regulated'
data$status[data$log2fc < -fc_threshold & data$p_value < p_threshold] <- 'Down-regulated'
data$status <- factor(data$status, levels = c('Down-regulated', 'Not significant', 'Up-regulated'))

# Colors
colors <- c('Down-regulated' = '#3498DB',
            'Not significant' = '#CCCCCC',
            'Up-regulated' = '#E74C3C')

# === PLOT A: Good volcano plot ===
p_good <- ggplot(data, aes(x = log2fc, y = neg_log10_p, color = status)) +
  geom_point(aes(size = status, alpha = status)) +
  scale_color_manual(values = colors) +
  scale_size_manual(values = c('Down-regulated' = 3, 'Not significant' = 1.5, 'Up-regulated' = 3)) +
  scale_alpha_manual(values = c('Down-regulated' = 0.8, 'Not significant' = 0.5, 'Up-regulated' = 0.8)) +

  # Threshold lines
  geom_hline(yintercept = neg_log10_p_threshold, linetype = 'dashed', size = 1, alpha = 0.7) +
  geom_vline(xintercept = c(-fc_threshold, fc_threshold), linetype = 'dashed', size = 1, alpha = 0.7) +

  # Annotations
  annotate('text', x = fc_threshold + 0.3, y = 1,
          label = paste0('FC > ', round(2^fc_threshold, 1), '×'),
          angle = 90, vjust = -0.5, size = 3) +
  annotate('text', x = -fc_threshold - 0.3, y = 1,
          label = paste0('FC < ', round(2^-fc_threshold, 1), '×'),
          angle = 90, vjust = 1.5, size = 3) +
  annotate('text', x = min(data$log2fc) + 1, y = neg_log10_p_threshold + 0.3,
          label = paste0('p < ', p_threshold), size = 3) +

  labs(x = 'Log₂ Fold Change',
       y = '-Log₁₀ (P-value)',
       title = 'A. Volcano Plot: Differential Expression',
       color = NULL, size = NULL, alpha = NULL) +

  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(face = 'bold', size = 13, hjust = 0.5),
    axis.title = element_text(face = 'bold', size = 12),
    legend.position = c(0.85, 0.85),
    legend.background = element_rect(fill = 'white', color = 'black', size = 0.5),
    panel.grid.major = element_line(color = 'gray90', size = 0.3)
  )

ggsave('volcano_plot_good.png', p_good, width = 8, height = 6, dpi = 300, bg = 'white')

# === PLOT B: BAD (no FC threshold) ===
data_bad <- data
data_bad$status_bad <- ifelse(data_bad$p_value < p_threshold,
                              'p < 0.05 (ignoring FC)',
                              'Not significant')

p_bad <- ggplot(data_bad, aes(x = log2fc, y = neg_log10_p, color = status_bad)) +
  geom_point(aes(size = status_bad, alpha = status_bad)) +
  scale_color_manual(values = c('p < 0.05 (ignoring FC)' = '#E74C3C',
                                'Not significant' = '#CCCCCC')) +
  scale_size_manual(values = c('p < 0.05 (ignoring FC)' = 3, 'Not significant' = 1.5)) +
  scale_alpha_manual(values = c('p < 0.05 (ignoring FC)' = 0.8, 'Not significant' = 0.5)) +

  geom_hline(yintercept = neg_log10_p_threshold, linetype = 'dashed', size = 1, alpha = 0.7) +

  labs(x = 'Log₂ Fold Change',
       y = '-Log₁₀ (P-value)',
       title = '❌ B. BAD: No Fold Change Threshold\n(Includes small, insignificant changes)',
       color = NULL, size = NULL, alpha = NULL) +

  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(face = 'bold', size = 13, hjust = 0.5, color = 'red'),
    axis.title = element_text(face = 'bold', size = 12),
    legend.position = c(0.8, 0.85),
    legend.background = element_rect(fill = 'white', color = 'black', size = 0.5),
    panel.grid.major = element_line(color = 'gray90', size = 0.3)
  )

ggsave('volcano_plot_bad.png', p_bad, width = 8, height = 6, dpi = 300, bg = 'white')

# Print summary
cat("Volcano Plot Summary:\n")
cat(sprintf("Total genes: %d\n", n_genes))
cat(sprintf("Significant up-regulated: %d\n", sum(data$status == 'Up-regulated')))
cat(sprintf("Significant down-regulated: %d\n", sum(data$status == 'Down-regulated')))
cat(sprintf("Not significant: %d\n", sum(data$status == 'Not significant')))
cat(sprintf("\nThresholds:\n"))
cat(sprintf("- Fold change: ±%.2f× (log₂ ±%.1f)\n", 2^fc_threshold, fc_threshold))
cat(sprintf("- P-value: %.2f (-log₁₀ = %.2f)\n", p_threshold, neg_log10_p_threshold))
```

---

### Volcano Plot Best Practices

```
Essential elements:
 Log₂ fold change on x-axis (NOT linear fold change)
 -Log₁₀ p-value on y-axis (emphasizes small p-values)
 BOTH fold change AND p-value thresholds (two-criteria filtering)
 Clear color distinction (up vs. down vs. not significant)
 Threshold lines marked

Common thresholds:
- Fold change: ±1 (log₂) = ±2× linear
- P-value: 0.05 or adjusted p-value (FDR < 0.05)

Avoid:
❌ Using only p-value threshold (includes biologically irrelevant small changes)
❌ Linear fold change on x-axis (compresses negative values)
❌ Unlabeled axes (reader can't interpret scale)
❌ Missing threshold lines
```

---


## 7.3 PCA Plots and Dimensionality Reduction

### Understanding PCA Visualization

**Principal Component Analysis (PCA):** Reduces high-dimensional data (e.g., thousands of genes) to 2-3 principal components for visualization.

**What PCA Shows:**

```
Purpose: Reveal overall structure in high-dimensional data
- Clusters: Similar samples group together
- Outliers: Samples that deviate from groups
- Variance: How much variation each PC explains
- Batch effects: Unwanted technical variation

Common in: Genomics, metabolomics, single-cell analysis
```

---

### Critical PCA Design Elements

**Element 1: Explained Variance**

```
✓ ALWAYS report variance explained by each PC:
"PC1 (45.3%), PC2 (18.7%)"

Why it matters:
- Low variance (e.g., PC1=12%) suggests weak signal or many factors
- High variance (e.g., PC1=80%) suggests dominant pattern
- Helps interpret biological meaning
```

**Element 2: Color Encoding**

```
Color by biological variable of interest:
✓ Treatment group, tissue type, disease status, time point

NOT by technical variables (unless investigating batch effects):
❌ Sequencing batch, extraction date, plate number (unless that's the focus)
```

**Element 3: Confidence Ellipses**

```
Show group spread:
✓ 95% confidence ellipses around groups
✓ Convex hulls (enclose all points in group)

Reveals:
- Within-group variability
- Between-group separation
- Overlap between conditions
```

---

**Code Example (Python) - Comprehensive PCA Plot:**

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

np.random.seed(42)

# Generate synthetic high-dimensional data (e.g., gene expression)
n_samples = 60
n_features = 1000

# Three groups with different patterns
group_size = 20

# Group 1: Control
group1 = np.random.randn(group_size, n_features) * 0.5
group1[:, :100] += 2  # Shift first 100 features

# Group 2: Treatment A
group2 = np.random.randn(group_size, n_features) * 0.6
group2[:, 100:200] += 3  # Shift different features

# Group 3: Treatment B
group3 = np.random.randn(group_size, n_features) * 0.55
group3[:, 200:300] += 2.5  # Yet different features

# Combine data
X = np.vstack([group1, group2, group3])
labels = np.array(['Control']*group_size + ['Treatment A']*group_size + ['Treatment B']*group_size)

# Standardize features (critical for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Extract PC scores
pc1 = X_pca[:, 0]
pc2 = X_pca[:, 1]
pc3 = X_pca[:, 2]

# Variance explained
var_explained = pca.explained_variance_ratio_ * 100

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# === PANEL A: Basic PCA (PC1 vs PC2) ===
ax1 = axes[0, 0]

colors_map = {'Control': '#7F8C8D', 'Treatment A': '#3498DB', 'Treatment B': '#E74C3C'}
markers_map = {'Control': 'o', 'Treatment A': 's', 'Treatment B': '^'}

for group in ['Control', 'Treatment A', 'Treatment B']:
    mask = labels == group
    ax1.scatter(pc1[mask], pc2[mask],
               s=80, color=colors_map[group], marker=markers_map[group],
               alpha=0.7, edgecolors='black', linewidths=1,
               label=f'{group} (n={np.sum(mask)})')

ax1.set_xlabel(f'PC1 ({var_explained[0]:.1f}% variance)', fontsize=11, fontweight='bold')
ax1.set_ylabel(f'PC2 ({var_explained[1]:.1f}% variance)', fontsize=11, fontweight='bold')
ax1.set_title('A. Basic PCA Plot\n(Without confidence regions)',
             fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', frameon=True, fontsize=10)
ax1.grid(alpha=0.3)
ax1.axhline(0, color='black', linewidth=0.8, alpha=0.3)
ax1.axvline(0, color='black', linewidth=0.8, alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Panel label
ax1.text(-0.12, 1.05, 'A', transform=ax1.transAxes,
        fontsize=16, fontweight='bold', va='top')

# === PANEL B: PCA with 95% confidence ellipses ===
ax2 = axes[0, 1]

def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """Draw confidence ellipse"""
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                     facecolor=facecolor, **kwargs)

    scaleov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# Plot points and ellipses
for group in ['Control', 'Treatment A', 'Treatment B']:
    mask = labels == group
    ax2.scatter(pc1[mask], pc2[mask],
               s=80, color=colors_map[group], marker=markers_map[group],
               alpha=0.7, edgecolors='black', linewidths=1,
               label=f'{group} (n={np.sum(mask)})')

    # Add 95% confidence ellipse
    confidence_ellipse(pc1[mask], pc2[mask], ax2, n_std=2.0,
                      edgecolor=colors_map[group], linewidth=2.5,
                      facecolor=colors_map[group], alpha=0.1)

ax2.set_xlabel(f'PC1 ({var_explained[0]:.1f}% variance)', fontsize=11, fontweight='bold')
ax2.set_ylabel(f'PC2 ({var_explained[1]:.1f}% variance)', fontsize=11, fontweight='bold')
ax2.set_title('B. PCA with 95% Confidence Ellipses\n(Shows group spread)',
             fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', frameon=True, fontsize=10)
ax2.grid(alpha=0.3)
ax2.axhline(0, color='black', linewidth=0.8, alpha=0.3)
ax2.axvline(0, color='black', linewidth=0.8, alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.text(-0.12, 1.05, 'B', transform=ax2.transAxes,
        fontsize=16, fontweight='bold', va='top')

# === PANEL C: Scree plot (variance explained) ===
ax3 = axes[1, 0]

n_components = min(10, X_scaled.shape[1])
pca_full = PCA(n_components=n_components)
pca_full.fit(X_scaled)

components = range(1, n_components + 1)
var_exp = pca_full.explained_variance_ratio_ * 100

ax3.bar(components, var_exp, color='#3498DB', edgecolor='black', linewidth=1.5, alpha=0.7)
ax3.plot(components, var_exp, 'ro-', linewidth=2, markersize=8)

ax3.set_xlabel('Principal Component', fontsize=11, fontweight='bold')
ax3.set_ylabel('Variance Explained (%)', fontsize=11, fontweight='bold')
ax3.set_title('C. Scree Plot\n(How many PCs to consider?)',
             fontsize=12, fontweight='bold')
ax3.set_xticks(components)
ax3.grid(axis='y', alpha=0.3)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Add cumulative variance line
ax3_twin = ax3.twinx()
cumvar = np.cumsum(var_exp)
ax3_twin.plot(components, cumvar, 'g^--', linewidth=2, markersize=8, alpha=0.7,
             label='Cumulative')
ax3_twin.set_ylabel('Cumulative Variance (%)', fontsize=11, fontweight='bold', color='green')
ax3_twin.tick_params(axis='y', labelcolor='green')
ax3_twin.spines['top'].set_visible(False)
ax3_twin.legend(loc='lower right', frameon=True, fontsize=9)

ax3.text(-0.12, 1.05, 'C', transform=ax3.transAxes,
        fontsize=16, fontweight='bold', va='top')

# === PANEL D: 3D PCA (PC1 vs PC2 vs PC3) ===
ax4 = fig.add_subplot(2, 2, 4, projection='3d')

for group in ['Control', 'Treatment A', 'Treatment B']:
    mask = labels == group
    ax4.scatter(pc1[mask], pc2[mask], pc3[mask],
               s=80, color=colors_map[group], marker=markers_map[group],
               alpha=0.7, edgecolors='black', linewidths=1,
               label=f'{group}')

ax4.set_xlabel(f'PC1 ({var_explained[0]:.1f}%)', fontsize=10, fontweight='bold')
ax4.set_ylabel(f'PC2 ({var_explained[1]:.1f}%)', fontsize=10, fontweight='bold')
ax4.set_zlabel(f'PC3 ({var_explained[2]:.1f}%)', fontsize=10, fontweight='bold')
ax4.set_title('D. 3D PCA Plot\n(PC1-PC2-PC3)', fontsize=12, fontweight='bold', pad=20)
ax4.legend(loc='upper left', frameon=True, fontsize=9)

ax4.text2D(-0.12, 1.05, 'D', transform=ax4.transAxes,
          fontsize=16, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('pca_comprehensive.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Print summary
print("PCA Analysis Summary:")
print(f"Total samples: {n_samples}")
print(f"Total features: {n_features}")
print(f"\nVariance explained:")
for i, var in enumerate(var_explained[:5]):
    print(f"  PC{i+1}: {var:.2f}%")
print(f"\nCumulative variance (first 3 PCs): {np.sum(var_explained[:3]):.2f}%")
```

**Code Example (R) - Comprehensive PCA Plot:**

```r
library(ggplot2)
library(ggrepel)
library(patchwork)
library(ggforce)  # For stat_ellipse

set.seed(42)

# Generate synthetic data
n_samples <- 60
n_features <- 1000
group_size <- 20

# Three groups
group1 <- matrix(rnorm(group_size * n_features, 0, 0.5), group_size, n_features)
group1[, 1:100] <- group1[, 1:100] + 2

group2 <- matrix(rnorm(group_size * n_features, 0, 0.6), group_size, n_features)
group2[, 101:200] <- group2[, 101:200] + 3

group3 <- matrix(rnorm(group_size * n_features, 0, 0.55), group_size, n_features)
group3[, 201:300] <- group3[, 201:300] + 2.5

X <- rbind(group1, group2, group3)
labels <- factor(rep(c('Control', 'Treatment A', 'Treatment B'), each = group_size))

# Standardize and perform PCA
X_scaled <- scale(X)
pca_result <- prcomp(X_scaled, center = FALSE, scale. = FALSE)

# Extract PC scores
pca_data <- data.frame(
  PC1 = pca_result$x[, 1],
  PC2 = pca_result$x[, 2],
  PC3 = pca_result$x[, 3],
  Group = labels
)

# Variance explained
var_explained <- summary(pca_result)$importance[2, ] * 100

# Colors
colors <- c('Control' = '#7F8C8D',
            'Treatment A' = '#3498DB',
            'Treatment B' = '#E74C3C')

# === PANEL A: Basic PCA ===
p_a <- ggplot(pca_data, aes(x = PC1, y = PC2, color = Group, shape = Group)) +
  geom_point(size = 4, alpha = 0.7) +
  scale_color_manual(values = colors) +
  scale_shape_manual(values = c(16, 15, 17)) +  # circle, square, triangle

  geom_hline(yintercept = 0, color = 'black', size = 0.5, alpha = 0.3) +
  geom_vline(xintercept = 0, color = 'black', size = 0.5, alpha = 0.3) +

  labs(x = sprintf('PC1 (%.1f%% variance)', var_explained[1]),
       y = sprintf('PC2 (%.1f%% variance)', var_explained[2]),
       title = 'A. Basic PCA Plot\n(Without confidence regions)',
       color = NULL, shape = NULL) +

  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = 'bold', size = 12, hjust = 0.5),
    axis.title = element_text(face = 'bold'),
    legend.position = c(0.85, 0.15),
    legend.background = element_rect(fill = 'white', color = 'black', size = 0.5),
    panel.grid.major = element_line(color = 'gray90', size = 0.3),
    plot.tag = element_text(size = 16, face = 'bold')
  )

# === PANEL B: PCA with confidence ellipses ===
p_b <- ggplot(pca_data, aes(x = PC1, y = PC2, color = Group, fill = Group, shape = Group)) +
  geom_point(size = 4, alpha = 0.7) +
  stat_ellipse(aes(fill = Group), geom = "polygon", alpha = 0.1, level = 0.95, size = 1.5) +

  scale_color_manual(values = colors) +
  scale_fill_manual(values = colors) +
  scale_shape_manual(values = c(16, 15, 17)) +

  geom_hline(yintercept = 0, color = 'black', size = 0.5, alpha = 0.3) +
  geom_vline(xintercept = 0, color = 'black', size = 0.5, alpha = 0.3) +

  labs(x = sprintf('PC1 (%.1f%% variance)', var_explained[1]),
       y = sprintf('PC2 (%.1f%% variance)', var_explained[2]),
       title = 'B. PCA with 95% Confidence Ellipses\n(Shows group spread)',
       color = NULL, fill = NULL, shape = NULL) +

  guides(fill = guide_legend(override.aes = list(alpha = 0.3))) +

  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = 'bold', size = 12, hjust = 0.5),
    axis.title = element_text(face = 'bold'),
    legend.position = c(0.85, 0.15),
    legend.background = element_rect(fill = 'white', color = 'black', size = 0.5),
    panel.grid.major = element_line(color = 'gray90', size = 0.3),
    plot.tag = element_text(size = 16, face = 'bold')
  )

# === PANEL C: Scree plot ===
scree_data <- data.frame(
  PC = 1:10,
  Variance = var_explained[1:10],
  Cumulative = cumsum(var_explained[1:10])
)

p_c <- ggplot(scree_data, aes(x = PC, y = Variance)) +
  geom_bar(stat = 'identity', fill = '#3498DB', color = 'black', size = 1, alpha = 0.7) +
  geom_line(color = 'red', size = 1.5) +
  geom_point(color = 'red', size = 3) +

  labs(x = 'Principal Component',
       y = 'Variance Explained (%)',
       title = 'C. Scree Plot\n(How many PCs to consider?)') +

  scale_x_continuous(breaks = 1:10) +

  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = 'bold', size = 12, hjust = 0.5),
    axis.title = element_text(face = 'bold'),
    panel.grid.major.y = element_line(color = 'gray90', size = 0.3),
    plot.tag = element_text(size = 16, face = 'bold')
  )

# Add cumulative line
p_c <- p_c +
  geom_line(aes(y = Cumulative), color = 'darkgreen', size = 1.5, linetype = 'dashed') +
  geom_point(aes(y = Cumulative), color = 'darkgreen', size = 3, shape = 17)

# Combine plots
combined <- (p_a | p_b) / (p_c | plot_spacer()) +
  plot_annotation(tag_levels = 'A') &
  theme(plot.tag = element_text(size = 16, face = 'bold'))

ggsave('pca_comprehensive.png', combined, width = 14, height = 12, dpi = 300, bg = 'white')

# Print summary
cat("PCA Analysis Summary:\n")
cat(sprintf("Total samples: %d\n", n_samples))
cat(sprintf("Total features: %d\n", n_features))
cat("\nVariance explained:\n")
for(i in 1:5) {
  cat(sprintf("  PC%d: %.2f%%\n", i, var_explained[i]))
}
cat(sprintf("\nCumulative variance (first 3 PCs): %.2f%%\n", sum(var_explained[1:3])))
```

---

### PCA Plot Best Practices Checklist

```
Essential elements:
 Variance explained in axis labels: "PC1 (45.3% variance)"
 Clear color/shape encoding by biological variable
 Confidence ellipses or convex hulls for groups
 Grid lines at x=0, y=0 (show PC center)
 Legend with sample sizes per group

Interpretation aids:
 Scree plot (show how many PCs capture variance)
 Loadings plot (which features drive separation) — optional
 3D plot if PC3 explains >10% variance

Avoid:
❌ No variance explained reported (reader can't assess importance)
❌ Coloring by technical batch without justification
❌ Truncated axes that hide outliers
❌ Using PCA on non-normalized data (scale features first!)
```

---

## 7.4 Survival Curves (Kaplan-Meier Plots)

### Understanding Survival Analysis Visualization

**Kaplan-Meier (KM) curve:** Shows probability of survival over time, accounting for censored data (subjects lost to follow-up or still alive at study end).

**Common in:** Clinical trials, cancer research, reliability engineering

**Key components:**

```
1. Survival probability (y-axis): 0 to 1 (or 0% to 100%)
2. Time (x-axis): Days, months, years since baseline
3. Step function: Drops at each event (death, failure)
4. Censored data markers: Tick marks for subjects without event
5. Confidence bands: Uncertainty around survival estimate
6. Log-rank test: Statistical comparison between groups
```

---

**Code Example (Python) - Kaplan-Meier Survival Curve:**

```python
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

np.random.seed(42)

# Simulate survival data for two treatment groups
n_control = 50
n_treatment = 50

# Control group: Worse survival
time_control = np.random.exponential(scale=12, size=n_control)  # Months
event_control = np.random.rand(n_control) < 0.7  # 70% events (deaths)

# Treatment group: Better survival
time_treatment = np.random.exponential(scale=20, size=n_treatment)  # Months
event_treatment = np.random.rand(n_treatment) < 0.5  # 50% events

# Fit Kaplan-Meier curves
kmf_control = KaplanMeierFitter()
kmf_control.fit(time_control, event_control, label='Control')

kmf_treatment = KaplanMeierFitter()
kmf_treatment.fit(time_treatment, event_treatment, label='Treatment')

# Statistical test
results = logrank_test(time_control, time_treatment,
                       event_control, event_treatment)
p_value = results.p_value

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# === PANEL A: Good KM curve ===
ax1 = axes[0]

# Plot survival curves
kmf_control.plot_survival_function(ax=ax1, ci_show=True,
                                   color='#7F8C8D', linewidth=3,
                                   alpha=0.8)
kmf_treatment.plot_survival_function(ax=ax1, ci_show=True,
                                     color='#E74C3C', linewidth=3,
                                     alpha=0.8)

# Add censored markers (tick marks)
# Note: lifelines plots these automatically as '+' markers

# Formatting
ax1.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
ax1.set_title('✓ Kaplan-Meier Survival Curves\n(With confidence bands)',
             fontsize=13, fontweight='bold', color='green')
ax1.set_ylim(0, 1.05)
ax1.set_xlim(0, None)
ax1.grid(alpha=0.3)
ax1.legend(loc='lower left', frameon=True, fontsize=11,
          title=f'Log-rank p = {p_value:.3f}')

# Add median survival lines
median_control = kmf_control.median_survival_time_
median_treatment = kmf_treatment.median_survival_time_

ax1.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax1.axvline(median_control, color='#7F8C8D', linestyle=':', linewidth=2, alpha=0.7)
ax1.axvline(median_treatment, color='#E74C3C', linestyle=':', linewidth=2, alpha=0.7)

# Annotate median survival
ax1.text(median_control + 1, 0.52, f'Median: {median_control:.1f} mo',
        fontsize=9, color='#7F8C8D', fontweight='bold')
ax1.text(median_treatment + 1, 0.47, f'Median: {median_treatment:.1f} mo',
        fontsize=9, color='#E74C3C', fontweight='bold')

# Number at risk table (simplified annotation)
ax1.text(0.02, 0.05,
        f'Numbers at risk:\nControl: {n_control} patients\nTreatment: {n_treatment} patients',
        transform=ax1.transAxes, fontsize=9, va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Panel label
ax1.text(-0.12, 1.05, 'A', transform=ax1.transAxes,
        fontsize=16, fontweight='bold', va='top')

# === PANEL B: Common mistakes ===
ax2 = axes[1]

# Plot without confidence bands (less informative)
kmf_control.plot_survival_function(ax=ax2, ci_show=False,
                                   color='#7F8C8D', linewidth=3,
                                   alpha=0.8)
kmf_treatment.plot_survival_function(ax=ax2, ci_show=False,
                                     color='#E74C3C', linewidth=3,
                                     alpha=0.8)

ax2.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
ax2.set_title('❌ Common Mistakes:\n(No confidence bands, no statistics)',
             fontsize=13, fontweight='bold', color='red')
ax2.set_ylim(0, 1.05)
ax2.set_xlim(0, None)
ax2.grid(alpha=0.3)
ax2.legend(loc='lower left', frameon=True, fontsize=11)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.text(-0.12, 1.05, 'B', transform=ax2.transAxes,
        fontsize=16, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('kaplan_meier_survival.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Print summary statistics
print("Survival Analysis Summary:")
print(f"\nControl group (n={n_control}):")
print(f"  Median survival: {median_control:.2f} months")
print(f"  Events (deaths): {np.sum(event_control)} ({np.sum(event_control)/n_control*100:.1f}%)")

print(f"\nTreatment group (n={n_treatment}):")
print(f"  Median survival: {median_treatment:.2f} months")
print(f"  Events (deaths): {np.sum(event_treatment)} ({np.sum(event_treatment)/n_treatment*100:.1f}%)")

print(f"\nLog-rank test:")
print(f"  p-value: {p_value:.4f}")
if p_value < 0.05:
    print("  Conclusion: Significant difference between groups")
else:
    print("  Conclusion: No significant difference between groups")
```

---

### Survival Curve Best Practices

```
Essential elements:
 Y-axis: 0 to 1 (or 0% to 100%), starts at 1.0
 X-axis: Starts at 0 (baseline)
 Step function (NOT smooth curve)
 Confidence bands (95% CI) shaded or as dashed lines
 Censored data marked (tick marks or '+' symbols)
 Log-rank test p-value reported
 Median survival times noted (optional lines)
 Number at risk table below plot (or in legend)

Caption must include:
- Sample size per group
- Number of events (deaths) per group
- Follow-up duration
- Statistical test used (log-rank test)

Avoid:
❌ Smooth interpolation (survival probability drops only at events)
❌ No confidence bands (hides uncertainty)
❌ Missing censored data indicators
❌ No statistical comparison (log-rank test)
❌ Y-axis not starting at 1.0
```

---

## 7.5 ROC Curves and Performance Metrics

### Understanding ROC (Receiver Operating Characteristic) Curves

**Purpose:** Evaluate classifier or diagnostic test performance across all possible decision thresholds.

**What ROC shows:**

```
X-axis: False Positive Rate (FPR) = 1 - Specificity
Y-axis: True Positive Rate (TPR) = Sensitivity

Key metrics:
- AUC (Area Under Curve): Overall discriminative ability
  * AUC = 1.0: Perfect classifier
  * AUC = 0.5: Random guessing (diagonal line)
  * AUC < 0.5: Worse than random (inverse predictions)

Common in: Machine learning, diagnostics, biomarker validation
```

---

### Critical ROC Design Elements

**Element 1: Diagonal Reference Line**

```
✓ ALWAYS include diagonal (y=x) line representing random chance
→ Shows baseline performance
→ Curves above = better than random
→ Curves below = worse than random (possibly inverted predictions)
```

**Element 2: AUC with Confidence Interval**

```
✓ Report: AUC = 0.85 (95% CI: 0.78-0.91)
❌ NEVER report AUC alone without CI
→ CI indicates uncertainty/sample size effect
```

**Element 3: Optimal Operating Point**

```
Mark optimal threshold on curve:
- Balance sensitivity and specificity
- Often chosen by Youden's index: max(Sensitivity + Specificity - 1)
- Or based on clinical cost/benefit considerations
```

---

**Code Example (Python) - ROC Curve Analysis:**

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import stats

np.random.seed(42)

# Simulate binary classification results
n_samples = 500

# True labels (0 = negative, 1 = positive)
y_true = np.random.binomial(1, 0.4, n_samples)

# Predicted probabilities (imperfect classifier)
# Positive cases have higher scores on average
y_scores_good = np.where(y_true == 1,
                         np.random.beta(7, 3, n_samples),  # Higher scores
                         np.random.beta(3, 7, n_samples))  # Lower scores

y_scores_poor = np.where(y_true == 1,
                         np.random.beta(5, 5, n_samples),  # Only slightly higher
                         np.random.beta(5, 5, n_samples))  # Similar distribution

# Calculate ROC curves
fpr_good, tpr_good, thresholds_good = roc_curve(y_true, y_scores_good)
fpr_poor, tpr_poor, thresholds_poor = roc_curve(y_true, y_scores_poor)

# Calculate AUC
auc_good = auc(fpr_good, tpr_good)
auc_poor = auc(fpr_poor, tpr_poor)

# Bootstrap confidence intervals for AUC (simplified)
def bootstrap_auc(y_true, y_scores, n_bootstrap=1000):
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            aucs.append(roc_auc_score(y_true[idx], y_scores[idx]))
        except:
            continue
    return np.percentile(aucs, [2.5, 97.5])

ci_good = bootstrap_auc(y_true, y_scores_good, 1000)
ci_poor = bootstrap_auc(y_true, y_scores_poor, 1000)

# Find optimal threshold (Youden's index)
j_good = tpr_good - fpr_good
optimal_idx_good = np.argmax(j_good)
optimal_threshold_good = thresholds_good[optimal_idx_good]
optimal_fpr_good = fpr_good[optimal_idx_good]
optimal_tpr_good = tpr_good[optimal_idx_good]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# === PANEL A: Good ROC curve ===
ax1 = axes[0]

# Plot diagonal (random chance)
ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Random (AUC=0.50)')

# Plot good classifier
ax1.plot(fpr_good, tpr_good, color='#E74C3C', linewidth=3,
        label=f'Good Classifier (AUC={auc_good:.3f}, 95% CI: {ci_good[0]:.3f}-{ci_good[1]:.3f})')

# Fill area under curve
ax1.fill_between(fpr_good, tpr_good, alpha=0.2, color='#E74C3C')

# Mark optimal operating point
ax1.plot(optimal_fpr_good, optimal_tpr_good, 'bo', markersize=12,
        label=f'Optimal threshold = {optimal_threshold_good:.3f}')
ax1.annotate(f'Sensitivity: {optimal_tpr_good:.3f}\nSpecificity: {1-optimal_fpr_good:.3f}',
            xy=(optimal_fpr_good, optimal_tpr_good),
            xytext=(optimal_fpr_good + 0.15, optimal_tpr_good - 0.15),
            fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            arrowprops=dict(arrowstyle='->', lw=1.5))

# Formatting
ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
ax1.set_title('✓ A. ROC Curve: Good Classifier\n(AUC >> 0.5)',
             fontsize=13, fontweight='bold', color='green')
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.05, 1.05)
ax1.set_aspect('equal')
ax1.legend(loc='lower right', frameon=True, fontsize=10)
ax1.grid(alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Panel label
ax1.text(-0.12, 1.05, 'A', transform=ax1.transAxes,
        fontsize=16, fontweight='bold', va='top')

# === PANEL B: Comparison with poor classifier ===
ax2 = axes[1]

# Plot diagonal
ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Random (AUC=0.50)')

# Plot good classifier
ax2.plot(fpr_good, tpr_good, color='#E74C3C', linewidth=3,
        label=f'Good Classifier (AUC={auc_good:.3f})')

# Plot poor classifier
ax2.plot(fpr_poor, tpr_poor, color='#7F8C8D', linewidth=3,
        label=f'Poor Classifier (AUC={auc_poor:.3f})')

# Statistical comparison annotation
ax2.text(0.5, 0.2, f'ΔAUC = {auc_good - auc_poor:.3f}',
        ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Formatting
ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
ax2.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
ax2.set_title('B. ROC Comparison:\nGood vs. Poor Classifier',
             fontsize=13, fontweight='bold')
ax2.set_xlim(-0.05, 1.05)
ax2.set_ylim(-0.05, 1.05)
ax2.set_aspect('equal')
ax2.legend(loc='lower right', frameon=True, fontsize=10)
ax2.grid(alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.text(-0.12, 1.05, 'B', transform=ax2.transAxes,
        fontsize=16, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('roc_curve_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Print summary
print("ROC Curve Analysis Summary:")
print(f"\nGood Classifier:")
print(f"  AUC: {auc_good:.4f}")
print(f"  95% CI: [{ci_good[0]:.4f}, {ci_good[1]:.4f}]")
print(f"  Optimal threshold: {optimal_threshold_good:.4f}")
print(f"  At optimal: Sensitivity={optimal_tpr_good:.3f}, Specificity={1-optimal_fpr_good:.3f}")

print(f"\nPoor Classifier:")
print(f"  AUC: {auc_poor:.4f}")
print(f"  95% CI: [{ci_poor[0]:.4f}, {ci_poor[1]:.4f}]")

print(f"\nInterpretation:")
if auc_good > 0.9:
    print("  Good classifier: Excellent discrimination")
elif auc_good > 0.8:
    print("  Good classifier: Good discrimination")
elif auc_good > 0.7:
    print("  Good classifier: Acceptable discrimination")
else:
    print("  Good classifier: Poor discrimination")
```

**Code Example (R) - ROC Curve Analysis:**

```r
library(ggplot2)
library(pROC)
library(patchwork)

set.seed(42)

# Simulate data
n_samples <- 500
y_true <- rbinom(n_samples, 1, 0.4)

# Good classifier
y_scores_good <- ifelse(y_true == 1,
                       rbeta(n_samples, 7, 3),
                       rbeta(n_samples, 3, 7))

# Poor classifier
y_scores_poor <- ifelse(y_true == 1,
                       rbeta(n_samples, 5, 5),
                       rbeta(n_samples, 5, 5))

# Calculate ROC curves using pROC
roc_good <- roc(y_true, y_scores_good, ci = TRUE)
roc_poor <- roc(y_true, y_scores_poor, ci = TRUE)

# Extract data for plotting
roc_good_df <- data.frame(
  fpr = 1 - roc_good$specificities,
  tpr = roc_good$sensitivities,
  classifier = 'Good'
)

roc_poor_df <- data.frame(
  fpr = 1 - roc_poor$specificities,
  tpr = roc_poor$sensitivities,
  classifier = 'Poor'
)

# Find optimal threshold (Youden's index)
optimal_idx <- which.max(roc_good$sensitivities + roc_good$specificities - 1)
optimal_point <- data.frame(
  fpr = 1 - roc_good$specificities[optimal_idx],
  tpr = roc_good$sensitivities[optimal_idx],
  threshold = roc_good$thresholds[optimal_idx]
)

# === PANEL A: Good ROC curve ===
p_a <- ggplot(roc_good_df, aes(x = fpr, y = tpr)) +
  # Diagonal reference line
  geom_abline(intercept = 0, slope = 1, linetype = 'dashed', size = 1.5, alpha = 0.5) +

  # ROC curve
  geom_line(color = '#E74C3C', size = 2) +
  geom_ribbon(aes(ymin = 0, ymax = tpr), fill = '#E74C3C', alpha = 0.2) +

  # Optimal point
  geom_point(data = optimal_point, aes(x = fpr, y = tpr),
            color = 'blue', size = 5, shape = 19) +
  annotate('text', x = optimal_point$fpr + 0.15, y = optimal_point$tpr - 0.12,
          label = sprintf('Optimal\nSensitivity: %.3f\nSpecificity: %.3f',
                         optimal_point$tpr, 1 - optimal_point$fpr),
          size = 3, hjust = 0,
          lineheight = 0.9) +

  annotate('segment', x = optimal_point$fpr, y = optimal_point$tpr,
          xend = optimal_point$fpr + 0.14, yend = optimal_point$tpr - 0.08,
          arrow = arrow(length = unit(0.3, 'cm')), size = 1) +

  # AUC annotation
  annotate('text', x = 0.6, y = 0.25,
          label = sprintf('AUC = %.3f\n95%% CI: [%.3f, %.3f]',
                         roc_good$auc, roc_good$ci[1], roc_good$ci[3]),
          size = 4, fontface = 'bold',
          hjust = 0, lineheight = 1.2) +

  labs(x = 'False Positive Rate (1 - Specificity)',
       y = 'True Positive Rate (Sensitivity)',
       title = '✓ A. ROC Curve: Good Classifier\n(AUC >> 0.5)') +

  coord_fixed(ratio = 1, xlim = c(0, 1), ylim = c(0, 1)) +

  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(face = 'bold', size = 13, hjust = 0.5, color = 'darkgreen'),
    axis.title = element_text(face = 'bold', size = 12),
    panel.grid.major = element_line(color = 'gray90', size = 0.3),
    plot.tag = element_text(size = 16, face = 'bold')
  )

# === PANEL B: Comparison ===
roc_combined <- rbind(roc_good_df, roc_poor_df)

p_b <- ggplot(roc_combined, aes(x = fpr, y = tpr, color = classifier)) +
  # Diagonal reference line
  geom_abline(intercept = 0, slope = 1, linetype = 'dashed', size = 1.5, alpha = 0.5,
             show.legend = FALSE) +

  # ROC curves
  geom_line(size = 2) +

  scale_color_manual(values = c('Good' = '#E74C3C', 'Poor' = '#7F8C8D'),
                    labels = c(sprintf('Good Classifier (AUC=%.3f)', roc_good$auc),
                              sprintf('Poor Classifier (AUC=%.3f)', roc_poor$auc))) +

  # Delta AUC annotation
  annotate('text', x = 0.5, y = 0.2,
          label = sprintf('ΔAUC = %.3f', roc_good$auc - roc_poor$auc),
          size = 4.5, fontface = 'bold',
          hjust = 0.5) +
  annotate('rect', xmin = 0.35, xmax = 0.65, ymin = 0.15, ymax = 0.25,
          fill = 'yellow', alpha = 0.3) +

  labs(x = 'False Positive Rate (1 - Specificity)',
       y = 'True Positive Rate (Sensitivity)',
       title = 'B. ROC Comparison:\nGood vs. Poor Classifier',
       color = NULL) +

  coord_fixed(ratio = 1, xlim = c(0, 1), ylim = c(0, 1)) +

  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(face = 'bold', size = 13, hjust = 0.5),
    axis.title = element_text(face = 'bold', size = 12),
    legend.position = c(0.65, 0.15),
    legend.background = element_rect(fill = 'white', color = 'black', size = 0.5),
    panel.grid.major = element_line(color = 'gray90', size = 0.3),
    plot.tag = element_text(size = 16, face = 'bold')
  )

# Combine plots
combined <- p_a | p_b
combined <- combined +
  plot_annotation(tag_levels = 'A') &
  theme(plot.tag = element_text(size = 16, face = 'bold'))

ggsave('roc_curve_analysis.png', combined, width = 14, height = 6, dpi = 300, bg = 'white')

# Print summary
cat("ROC Curve Analysis Summary:\n")
cat(sprintf("\nGood Classifier:\n"))
cat(sprintf("  AUC: %.4f\n", roc_good$auc))
cat(sprintf("  95%% CI: [%.4f, %.4f]\n", roc_good$ci[1], roc_good$ci[3]))
cat(sprintf("  Optimal threshold: %.4f\n", optimal_point$threshold))
cat(sprintf("  At optimal: Sensitivity=%.3f, Specificity=%.3f\n",
           optimal_point$tpr, 1 - optimal_point$fpr))

cat(sprintf("\nPoor Classifier:\n"))
cat(sprintf("  AUC: %.4f\n", roc_poor$auc))
cat(sprintf("  95%% CI: [%.4f, %.4f]\n", roc_poor$ci[1], roc_poor$ci[3]))

cat(sprintf("\nInterpretation:\n"))
if (roc_good$auc > 0.9) {
  cat("  Good classifier: Excellent discrimination\n")
} else if (roc_good$auc > 0.8) {
  cat("  Good classifier: Good discrimination\n")
} else if (roc_good$auc > 0.7) {
  cat("  Good classifier: Acceptable discrimination\n")
} else {
  cat("  Good classifier: Poor discrimination\n")
}
```

---

### ROC Curve Best Practices Checklist

```
Essential elements:
 Diagonal reference line (random chance, AUC=0.5)
 AUC value with 95% confidence interval
 Both axes 0 to 1, equal aspect ratio (square plot)
 Optimal operating point marked (if applicable)
 Sample size stated (n positives, n negatives)

Interpretation guide:
AUC = 0.90-1.00: Excellent
AUC = 0.80-0.90: Good
AUC = 0.70-0.80: Fair
AUC = 0.60-0.70: Poor
AUC = 0.50-0.60: Fail (barely better than random)

Common mistakes to avoid:
❌ No diagonal reference line
❌ AUC reported without confidence interval
❌ Non-square aspect ratio (distorts curve appearance)
❌ Missing sample size
❌ Comparing AUCs without statistical test
❌ No indication of optimal threshold
```

---

**End of Chapter 7 Core Content**

**Summary: Common Scientific Figure Types**

| Figure Type | Best For | Key Elements | Common Pitfalls |
|-------------|----------|--------------|-----------------|
| **Heatmap** | Matrix data, patterns | Colormap choice, clustering, colorbar | Rainbow colormap, asymmetric diverging scale |
| **Volcano Plot** | Differential analysis | Log2FC, -log10(p), thresholds | No FC threshold, missing significance lines |
| **PCA Plot** | Dimensionality reduction | Variance explained, confidence ellipses | No variance reported, wrong color encoding |
| **Survival Curve** | Time-to-event data | Step function, CI bands, log-rank test | Smooth curve, no CI, missing censored markers |
| **ROC Curve** | Classifier performance | Diagonal line, AUC+CI, optimal point | No reference line, AUC without CI |

---
