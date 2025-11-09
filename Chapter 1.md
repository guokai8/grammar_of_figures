# Chapter 1: Foundations of Visual Perception

## 1.1 How We See: Graphical Perception Principles

### The Science Behind Data Visualization

Before we can create effective scientific figures, we must understand the fundamental mechanisms of how humans perceive and interpret visual information. **Graphical perception** is the study of how people extract quantitative information from visual displays. Understanding these principles transforms figure design from an aesthetic exercise into a scientifically-grounded practice.

### Pre-attentive Processing: What We See Without Thinking

**Pre-attentive processing** refers to visual features that our brain processes automatically, before conscious attention. These features are detected in less than 200-250 milliseconds—essentially instantaneously.

**Pre-attentive Visual Features:**
- **Color hue** (red vs. blue)
- **Intensity/brightness** (dark vs. light)
- **Orientation** (vertical vs. horizontal lines)
- **Size** (large vs. small)
- **Shape** (circle vs. square)
- **Position** (spatial location)
- **Motion** (in dynamic displays)

**Example Demonstration:**

Imagine two scatter plots:
- **Plot A**: 100 gray circles, with 1 red circle
- **Plot B**: 100 gray circles in random positions, with 1 gray circle positioned differently

In Plot A, the red circle "pops out" immediately (pre-attentive color detection). In Plot B, you must consciously search for the differently positioned circle (attentive processing).

**Implication for Figure Design:**
Use pre-attentive features to direct attention to the most important data points or patterns. Don't bury your key finding in visual complexity that requires effortful search.

**Good Example:**
```
A volcano plot showing differential gene expression:
- Thousands of gray points (non-significant genes)
- Significant upregulated genes in red
- Significant downregulated genes in blue
→ The colored points immediately draw attention
```

**Bad Example:**
```
The same volcano plot where:
- All points are gray
- Significant genes marked only by slightly different symbols (triangle vs. circle)
→ Requires careful examination; key findings don't stand out
```

### The Hierarchy of Perceptual Accuracy: Cleveland & McGill

In the 1980s, psychologists William Cleveland and Robert McGill conducted landmark experiments to determine how accurately people can judge different types of visual encodings. Their findings established a **hierarchy of perceptual accuracy**:

**Ranking from MOST to LEAST Accurate:**

1. **Position along a common scale** (e.g., points on a scatter plot with shared axes)
2. **Position along non-aligned scales** (e.g., separate panels with different scales)
3. **Length** (e.g., bar heights)
4. **Direction/Angle** (e.g., slopes of lines)
5. **Area** (e.g., bubble sizes in bubble charts)
6. **Volume/Density** (e.g., 3D shapes)
7. **Color saturation/intensity** (e.g., heatmap gradients)
8. **Color hue** (e.g., categorical colors)

**What This Means for Practice:**

**Most Accurate Encoding (Position):**
```
Comparing treatment effects across conditions:
✓ GOOD: Dot plot with groups on x-axis, values on y-axis
  → Direct position comparison along common y-axis scale
✗ AVOID: Pie charts comparing percentages
  → Requires angle/area judgments, much less accurate
```

**Length vs. Area:**
```
Showing relative quantities:
✓ GOOD: Bar chart (comparing bar heights/lengths)
✗ WORSE: Bubble chart (comparing circle areas)
  → People underestimate area differences

Mathematical reality:
- A circle with 2× the radius has 4× the area
- But viewers perceive only ~2× difference
```

**Color Hue for Quantitative Data:**
```
✗ BAD: Using rainbow colors (red → yellow → blue) for continuous data
  → No inherent ordering; perceptually non-uniform
✓ GOOD: Using sequential color scale (light blue → dark blue)
  → Clear ordering; leverages intensity perception
```

### Practical Application: Choosing Plot Types

Based on the Cleveland-McGill hierarchy, here's how to select plot types:

**For Precise Quantitative Comparisons:**
1. **Scatter plots** (position vs. position)
2. **Line graphs** (position over continuous variable like time)
3. **Bar charts** (length comparison)
4. **Dot plots** (position along scale)

**For Approximate Patterns/Trends:**
5. **Heatmaps** (color intensity for patterns, not precise values)
6. **Bubble charts** (area for rough magnitude)
7. **Pie charts** (only for 2-3 very different proportions, if at all)

**For Categorical Relationships:**
8. **Box plots/Violin plots** (distribution shapes)
9. **Network diagrams** (connectivity, not precise values)

### Attention and Visual Search

Beyond pre-attentive processing, how we direct and sustain attention matters for complex figures.

**Limited Attention Resources:**
- We can only consciously attend to one thing at a time
- Complex figures require serial processing (scanning element by element)
- Each additional element increases cognitive load

**Implications:**
1. **Minimize visual clutter**: Every element should serve a purpose
2. **Create clear visual hierarchy**: Most important elements should be most salient
3. **Limit simultaneous comparisons**: Human working memory holds ~4-7 items
4. **Guide the viewer's path**: Use layout to create a reading order

**Example of Attention Management:**

**Bad: Cognitive Overload**
```
A single figure trying to show:
- Time course of 8 treatments (8 overlapping lines)
- Individual data points (hundreds of dots)
- Error bars at each timepoint
- Statistical significance markers between all pairs
- Three different y-axis scales
→ Viewer doesn't know where to look first; analysis paralysis
```

**Good: Staged Information**
```
Multi-panel approach:
Panel A: Overview - mean trajectories only (3 key treatments)
Panel B: Detailed comparison - 2 treatments with error bars
Panel C: Summary - final endpoint comparison with statistics
→ Clear narrative progression; each panel has one message
```

### Gestalt Principles in Data Visualization

Gestalt psychology describes how humans organize visual elements into groups or unified wholes. These principles are fundamental to effective figure layout.

#### 1. Proximity
**Principle:** Elements close together are perceived as belonging together.

**Application:**
```
Multi-panel figures:
- Place related panels near each other
- Use white space to separate conceptually different sections
- Group legends near their corresponding data

Example:
Panel A (experiment 1) | Panel B (experiment 2)
[placed adjacently]
vs.
Panel C (control data)
[separated by white space]
```

#### 2. Similarity
**Principle:** Elements that share visual properties (color, shape, size) are perceived as related.

**Application:**
```
Consistent encoding across figures:
- Same colors for same treatments throughout manuscript
- Same symbols for same data types
- Same line styles for same conditions

Example:
Figure 1: Control = blue circles, Treatment = red squares
Figure 2: Should maintain same scheme, not switch arbitrarily
```

#### 3. Continuity
**Principle:** The eye follows paths and lines naturally.

**Application:**
```
Line graphs:
- Smooth lines guide eye through temporal progression
- Connecting related data points clarifies relationships
- Avoid unnecessary breaks in lines (unless data truly missing)

Example in multi-panel layout:
Arrange panels in reading order (left → right, top → bottom)
that matches narrative progression
```

#### 4. Closure
**Principle:** We perceive complete figures even when parts are missing.

**Application:**
```
Axis design:
✓ Can omit top and right spines (brain completes the frame)
✗ Don't omit axis labels (brain cannot infer meaning)

Scatter plots:
- Can use partial grid lines (just at major ticks)
- Brain fills in the implicit grid structure
```

#### 5. Figure-Ground Separation
**Principle:** We distinguish foreground objects from background.

**Application:**
```
Data vs. Context:
- Data elements (points, lines) should be visually prominent
- Supporting elements (grid, axis lines) should recede
- Use contrast: bold/bright for data, light/thin for context

Example:
✓ Black data points on light gray grid
✗ Gray data points on black grid (inverts natural hierarchy)
```

### Change Blindness and Inattentional Blindness

**Change Blindness:** Failure to detect changes in visual scenes when they occur during brief interruptions.

**Relevance to Figures:**
- Readers may miss subtle differences between conditions if not highlighted
- Before/after comparisons need clear side-by-side presentation
- Don't rely on readers remembering Figure 2 when viewing Figure 5

**Solutions:**
```
✓ Direct comparisons within same panel or adjacent panels
✓ Use annotation arrows to highlight changes
✓ Repeat reference condition in multiple panels if needed
```

**Inattentional Blindness:** Failing to see unexpected objects when attention is focused elsewhere.

**Relevance to Figures:**
- Readers focused on one aspect may miss important secondary patterns
- Critical findings need explicit visual emphasis
- Cannot assume readers will notice everything you see

**Example:**
```
Scatter plot showing correlation:
- Main focus: positive correlation in Treatment A
- Hidden insight: 3 extreme outliers that might be artifacts
→ If outliers aren't highlighted (different color/size),
   readers focused on the trend will miss them
```

### Working Memory Constraints

**Miller's Law:** Human working memory can hold approximately 7±2 items simultaneously.

**Implications for Figures:**

**Bad: Exceeding Memory Limits**
```
Line graph with 15 different colored lines
→ Cannot keep track of which color means what
→ Constant back-and-forth reference to legend
→ Cognitive overload
```

**Good: Respecting Memory Limits**
```
Option 1: Show only 3-5 key lines, others in supplementary figure
Option 2: Use small multiples (separate panels for each line)
Option 3: Highlight 2-3 key comparisons, gray out others
```

**Legend Design Consideration:**
```
If legend requires >7 items, consider:
- Is this really one figure, or should it be split?
- Can you use direct labeling instead of legend?
- Can you group items into fewer categories?
```

### Perceptual Biases We Cannot Avoid

#### The Weber-Fechner Law
**Principle:** We perceive differences logarithmically, not linearly.

**Implications:**
```
Comparing values:
- Difference between 1 and 2 feels bigger than between 10 and 11
  (both are +1, but 1→2 is 100% increase vs. 10→11 is 10% increase)

Solution for large ranges:
- Use log scales when data spans orders of magnitude
- This makes proportional differences perceptually equal
```

#### The Area Perception Bias
**Principle:** We systematically underestimate area differences.

**Example:**
```
Circle A has diameter 10, Circle B has diameter 20
- Actual area ratio: (20/10)² = 4:1
- Perceived ratio: approximately 2:1

Implication:
✗ Bubble charts exaggerate small differences, hide large ones
✓ Use length/position encodings for accurate magnitude comparison
```

#### The Color Contrast Effect
**Principle:** Perceived color depends on surrounding colors.

**Example:**
```
Same gray square appears:
- Darker when surrounded by white
- Lighter when surrounded by black

Implication for heatmaps:
- Use consistent backgrounds across figures
- Be aware that adjacent colors influence perception
- Test figures in both digital (white background) and print contexts
```

### Temporal Perception: Animation and Change

For dynamic displays (less common in static publications, but relevant for presentations and supplementary materials):

**Effective Use:**
- Showing temporal progression of processes
- Revealing complex patterns in stages
- Guiding attention through complex figures

**Caution:**
- Motion is highly pre-attentive (can be distracting)
- Difficult to compare frames from memory
- Not suitable for precise quantitative reading
- Accessibility issues (motion sensitivity)

**Best Practice:**
```
For presentations:
✓ Use animation to build up complexity gradually
✓ Pause on key frames for analysis
✓ Provide static "key frame" summary

For publications:
✓ Convert animations to multi-panel static figures
✓ Show representative timepoints
```

---

### Exercise 1.1.1: Perceptual Hierarchy Experiment

**Objective:** Experience the Cleveland-McGill hierarchy firsthand

**Materials needed:** Graph paper or plotting software

**Instructions:**

1. **Create three versions of the same data** (5 categories, values: 20, 35, 45, 60, 75):
   - Version A: Bar chart (length encoding)
   - Version B: Pie chart (angle encoding)
   - Version C: Bubble chart (area encoding)

2. **Test yourself:**
   - Look at each for 3 seconds only
   - Without referring back, estimate:
     * Which category has the highest value?
     * What is the ratio of largest to smallest value?
     * Rank all categories from smallest to largest

3. **Reflection questions:**
   - Which version made estimation easiest?
   - Where did you make errors?
   - Which version required the most "mental calculation"?
   - What does this tell you about encoding choice?

**Expected finding:** Bar chart will be easiest and most accurate; pie chart most difficult.

---

### Exercise 1.1.2: Pre-attentive Feature Detection

**Objective:** Identify what "pops out" in your figures

**Instructions:**

1. **Take a figure you're currently working on**

2. **The 3-second test:**
   - Show figure to a colleague for exactly 3 seconds
   - Remove it
   - Ask: "What did you notice first?"

3. **Analysis:**
   - Was the first thing they noticed your intended main message?
   - If yes: What pre-attentive feature made it stand out? (color, size, position?)
   - If no: What competed for attention? How can you adjust?

4. **Redesign:**
   - Emphasize the main message using pre-attentive features
   - De-emphasize secondary elements
   - Retest with a different colleague

---

### Exercise 1.1.3: Gestalt Principles Audit

**Objective:** Apply Gestalt principles to improve figure organization

**Instructions:**

1. **Select a multi-panel figure** (your own or from a published paper)

2. **Analyze each Gestalt principle:**

   - **Proximity:** Are related panels grouped together? Is white space used to separate conceptually different elements?

   - **Similarity:** Do similar data types use consistent visual encoding? Are colors/shapes reused meaningfully?

   - **Continuity:** Does the layout create a natural reading path? Does it match your narrative order?

   - **Closure:** Are all frames necessary, or could you simplify by letting the brain "complete" structures?

   - **Figure-Ground:** Do data elements stand out from supporting elements (axes, grids)?

3. **Score each principle:** 1 (poor) to 5 (excellent)

4. **Identify weakest area** and sketch one specific improvement

**Example improvement:**
```
Original: Four panels scattered with inconsistent spacing
Problem: Violation of proximity principle
Solution: Group panels A-B (related experiments) close together,
          separate from panels C-D (control data) with more white space
```

---

## 1.2 Visual Channels and Their Effectiveness

### What Are Visual Channels?

A **visual channel** (or **visual variable**) is any controlled aspect of a graphical mark that can encode data. Understanding which channels are most effective for which data types is foundational to making good visualization choices.

### The Complete Channel Inventory

**Spatial Channels (Most Effective):**
1. **Position (X, Y coordinates)** - *Best for quantitative data*
2. **Length** - *Very good for quantitative data*
3. **Angle** - *Good for ordinal/limited quantitative*
4. **Slope** - *Good for trends and rates of change*

**Appearance Channels (Moderate Effectiveness):**
5. **Area** - *Moderate for quantitative, good for magnitude*
6. **Volume** - *Poor for quantitative, avoid if possible*
7. **Color Hue** - *Best for categorical/nominal data*
8. **Color Saturation** - *Good for ordinal/limited quantitative*
9. **Color Luminance** - *Good for quantitative (single hue)*

**Texture Channels (Lower Effectiveness):**
10. **Shape** - *Good for categorical (limited #)*
11. **Texture/Pattern** - *Moderate for categorical*
12. **Orientation** - *Moderate for categorical/ordinal*

### Matching Channels to Data Types

The type of data you have determines which channels are appropriate:

#### 1. Quantitative Data (Continuous Numbers)

**Best Channels (in order):**
- **Position** (scatter plots, line graphs)
- **Length** (bar charts)
- **Color luminance** (single-hue gradient heatmaps)

**Example:**

```
Showing gene expression levels across samples:
✓ EXCELLENT: Position (dot plot with values on y-axis)
✓ GOOD: Length (bar chart)
✓ ACCEPTABLE: Color luminance (heatmap, light→dark)
✗ POOR: Color hue (rainbow heatmap)
✗ TERRIBLE: Area/volume (3D bar chart)
```

**Why position is best:**
- Humans excel at judging positions along a common scale
- Precise reading possible with gridlines
- Direct comparison between elements
- Minimal perceptual distortion

**Code Example (Python):**

```
import matplotlib.pyplot as plt
import numpy as np

# Sample data: gene expression across 5 conditions
genes = ['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE']
expression = [2.3, 5.1, 3.7, 8.2, 4.9]

# BEST: Position encoding (dot plot)
fig, axes = plt.subplots(1, 3, figsize=(12, 3))

# Plot 1: Position (best)
axes[0].scatter(expression, genes, s=100, color='steelblue')
axes[0].set_xlabel('Expression Level')
axes[0].set_title('Position Channel (Best)')
axes[0].grid(axis='x', alpha=0.3)

# Plot 2: Length (good)
axes[1].barh(genes, expression, color='steelblue')
axes[1].set_xlabel('Expression Level')
axes[1].set_title('Length Channel (Good)')

# Plot 3: Area (poor - for comparison)
# Normalize for bubble sizing
sizes = (np.array(expression) ** 2) * 10
axes[2].scatter([1]*len(genes), genes, s=sizes,
                color='steelblue', alpha=0.6)
axes[2].set_xlim(0.5, 1.5)
axes[2].set_xticks([])
axes[2].set_title('Area Channel (Poor)')

plt.tight_layout()
plt.savefig('channel_comparison.png', dpi=300, bbox_inches='tight')
```

#### 2. Ordinal Data (Ranked/Ordered Categories)

**Best Channels:**
- **Position** (along an ordered axis)
- **Color luminance** (sequential palette)
- **Size** (small to large)
- **Angle** (for cyclic data like time of day)

**Example:**

```
Showing disease severity (mild, moderate, severe):
✓ EXCELLENT: Position (ordered categories on x-axis)
✓ GOOD: Color (light → dark gradient)
✓ ACCEPTABLE: Size (small → medium → large markers)
✗ POOR: Color hue (red, blue, green - no inherent order)
```

**Code Example (R):**

```
library(ggplot2)
library(RColorBrewer)

# Sample data
data <- data.frame(
  patient = 1:20,
  severity = factor(sample(c("Mild", "Moderate", "Severe"), 20, replace=TRUE),
                    levels = c("Mild", "Moderate", "Severe"),
                    ordered = TRUE),
  response = rnorm(20, 50, 10)
)

# GOOD: Position + sequential color for ordinal data
ggplot(data, aes(x = severity, y = response, color = severity)) +
  geom_jitter(size = 3, width = 0.2) +
  scale_color_brewer(palette = "YlOrRd") +  # Sequential palette
  labs(x = "Disease Severity", y = "Treatment Response",
       title = "Position + Sequential Color for Ordinal Data") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")  # Legend redundant when x-axis labeled
```

#### 3. Categorical/Nominal Data (Unordered Groups)

**Best Channels:**
- **Color hue** (distinct colors)
- **Shape** (different symbols - limit to ~6)
- **Position** (separate groups along axis)
- **Faceting** (separate panels)

**Example:**

```
Showing data from 4 different cell lines:
✓ EXCELLENT: Color hue (4 distinct colors)
✓ GOOD: Shape (4 different symbols: circle, square, triangle, diamond)
✓ ACCEPTABLE: Position (4 separate groups on x-axis)
✗ AVOID: Color luminance (implies ordering that doesn't exist)
```

**Code Example (Python):**

```
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample data
data = pd.DataFrame({
    'cell_line': np.repeat(['A', 'B', 'C', 'D'], 50),
    'time': np.tile(np.arange(50), 4),
    'growth': np.concatenate([
        np.cumsum(np.random.randn(50) + 0.5),
        np.cumsum(np.random.randn(50) + 0.3),
        np.cumsum(np.random.randn(50) + 0.7),
        np.cumsum(np.random.randn(50) + 0.4)
    ])
})

# GOOD: Color hue for categorical data
fig, ax = plt.subplots(figsize=(8, 5))

# Qualitative color palette
colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488']

for i, cell_line in enumerate(['A', 'B', 'C', 'D']):
    subset = data[data['cell_line'] == cell_line]
    ax.plot(subset['time'], subset['growth'],
            color=colors[i], linewidth=2, label=f'Cell Line {cell_line}')

ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('Cell Growth (AU)', fontsize=12)
ax.set_title('Color Hue for Categorical Data', fontsize=14, fontweight='bold')
ax.legend(frameon=True, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('categorical_channels.png', dpi=300, bbox_inches='tight')
```

### Channel Combination Rules

Often you need to encode multiple data dimensions simultaneously. Here's how to combine channels effectively:

#### Rule 1: Use the Strongest Channel for the Most Important Variable

```
Example: Scatter plot showing drug response

Primary variable (most important): Treatment outcome
Secondary variable: Patient age
Tertiary variable: Treatment group

Encoding:
✓ Position Y: Treatment outcome (strongest channel)
✓ Position X: Patient age (second strongest)
✓ Color: Treatment group (weaker channel, but effective for categories)
```

#### Rule 2: Avoid Channel Conflict

**Conflict Example:**
```
✗ BAD: Using both size AND color luminance to encode the same variable
  → Redundant and potentially contradictory
  → e.g., larger AND darker doesn't add information

✓ BETTER: Use size for one variable, color hue for another categorical variable
```

**Non-Conflict Example:**
```
Encoding three variables on a scatter plot:
- X-position: Time
- Y-position: Temperature
- Color: Location (categorical)
→ No conflict: each channel encodes different information
```

#### Rule 3: Leverage Redundancy for Accessibility

**Strategic Redundancy:**
```
✓ GOOD: Encoding the same categorical variable with BOTH color AND shape
  → Ensures colorblind accessibility
  → Shape alone works if color unavailable

Example:
Treatment A: Red circles
Treatment B: Blue squares
Treatment C: Green triangles

→ If colors indistinguishable, shapes still differentiate groups
```

#### Rule 4: Limit Channel Overload

**Too Many Channels:**
```
✗ BAD: Encoding 6 variables on one scatter plot:
  - X position
  - Y position
  - Color hue
  - Color saturation
  - Size
  - Shape

→ Cognitively overwhelming
→ Hard to decode

✓ BETTER: Use small multiples (faceting) to split some variables into panels
```

### Common Channel Mistakes

#### Mistake 1: Using Area for Quantitative Comparison

**The Problem:**
```
Bubble chart where bubble size represents quantity:
- Data: values of 100, 200, 300
- Bubble areas: A₁, A₂, A₃
- Perceptual ratio: ~1.5:2:2.3
- Actual ratio: 1:2:3

→ Systematic underestimation of larger values
```

**The Fix:**
```
✓ Use length encoding instead (bar chart)
✗ If bubble chart necessary, scale radius (not area) linearly
  But still expect perceptual error
```

**Code to Demonstrate:**

```
import matplotlib.pyplot as plt
import numpy as np

values = np.array([100, 200, 300, 400])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# WRONG: Area proportional to value (default behavior)
axes[0].scatter([1, 2, 3, 4], [1, 1, 1, 1], s=values, alpha=0.5)
axes[0].set_xlim(0, 5)
axes[0].set_ylim(0, 2)
axes[0].set_title('WRONG: Area ∝ Value\n(Perceptually misleading)')
axes[0].set_xticks([1, 2, 3, 4])
axes[0].set_xticklabels(values)

# BETTER: Length proportional to value
axes[1].bar([1, 2, 3, 4], values, width=0.6, alpha=0.7)
axes[1].set_title('BETTER: Length ∝ Value\n(Accurate perception)')
axes[1].set_xticks([1, 2, 3, 4])
axes[1].set_xticklabels(values)
axes[1].set_ylabel('Value')

plt.tight_layout()
plt.savefig('area_vs_length.png', dpi=300, bbox_inches='tight')
```

#### Mistake 2: Rainbow Color Maps for Quantitative Data

**The Problem:**

```
Using hue spectrum (rainbow: red→yellow→green→blue→violet) for continuous data:

Issues:
1. No perceptual ordering (is yellow > green?)
2. Non-uniform perceptual steps (yellow is much brighter)
3. False boundaries (sharp transitions in smooth data)
4. Accessibility issues (colorblind viewers lose information)
```

**The Fix:**

```
✓ Use sequential colormaps (single hue, varying luminance)
  Examples: Blues, Greens, Greys

✓ Use diverging colormaps for data with meaningful midpoint
  Examples: Blue-White-Red for positive/negative values

✓ Use perceptually uniform colormaps
  Examples: viridis, plasma, cividis
```

**Demonstration:**

```
import matplotlib.pyplot as plt
import numpy as np

# Create sample heatmap data
data = np.random.randn(10, 10).cumsum(axis=1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# BAD: Rainbow
im1 = axes[0].imshow(data, cmap='jet', aspect='auto')
axes[0].set_title('BAD: Rainbow (jet)\nNon-uniform, no order', fontsize=12)
plt.colorbar(im1, ax=axes[0])

# BETTER: Sequential
im2 = axes[1].imshow(data, cmap='Blues', aspect='auto')
axes[1].set_title('BETTER: Sequential\nClear low→high', fontsize=12)
plt.colorbar(im2, ax=axes[1])

# BEST: Perceptually uniform
im3 = axes[2].imshow(data, cmap='viridis', aspect='auto')
axes[2].set_title('BEST: Viridis\nPerceptually uniform', fontsize=12)
plt.colorbar(im3, ax=axes[2])

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig('colormap_comparison.png', dpi=300, bbox_inches='tight')
```

#### Mistake 3: Using 3D for 2D Data

**The Problem:**

```
3D bar charts, 3D pie charts, 3D scatter plots when z-axis adds no information:

Issues:
1. Perspective distortion (objects in back appear smaller)
2. Occlusion (bars hide each other)
3. Difficult to read exact values
4. Adds no information, only visual complexity
```

**The Fix:**

```
✓ Use 2D representations with position/length channels
✗ Only use 3D when you genuinely have 3 spatial dimensions of data
  (e.g., protein structure, geographic elevation)
```

---

### Exercise 1.2.1: Channel Selection Practice

**Objective:** Practice choosing appropriate channels for different data types

**Scenario:** You have the following dataset from a clinical trial:

- **Patient ID** (categorical, nominal)
- **Treatment Group** (categorical: A, B, C, Control)
- **Age** (quantitative, continuous: 20-80 years)
- **Disease Severity** (ordinal: Mild, Moderate, Severe)
- **Treatment Outcome** (quantitative, continuous: 0-100 scale)
- **Time to Response** (quantitative, continuous: days)

**Task:** Design visualizations for these questions, specifying which channels you'd use:

1. **Question:** How does treatment outcome vary by treatment group?
   - Primary channel for outcome: _______
   - Channel for treatment group: _______
   - Justification: _______

2. **Question:** Is there a relationship between age and treatment outcome?
   - Channel for age: _______
   - Channel for outcome: _______
   - Optional third variable (disease severity): _______
   - Justification: _______

3. **Question:** How do the four treatment groups compare across disease severity levels?
   - Channel for treatment group: _______
   - Channel for severity: _______
   - Channel for outcome: _______
   - Plot type: _______

**Deliverable:** Sketch rough layouts and write 2-3 sentences justifying each channel choice based on the Cleveland-McGill hierarchy.

---

### Exercise 1.2.2: Channel Effectiveness Audit

**Objective:** Evaluate channel usage in existing figures

**Instructions:**

1. **Find 3 published figures** from your field (preferably recent high-impact papers)

2. **For each figure, document:**
   - What channels are used? (position, length, color, size, shape, etc.)
   - What type of data is each channel encoding? (quantitative, ordinal, categorical)
   - Is this an effective match based on the hierarchy?

3. **Identify at least one figure where:**
   - A strong channel is wasted on less important data
   - A weak channel is used for critical quantitative data
   - Channel choice could be improved

4. **Redesign:** Sketch how you would reassign channels for better effectiveness

**Example Analysis:**

```
Figure: Scatter plot of gene expression

Current encoding:
- X-axis (position): Gene ID number (nominal) ❌
- Y-axis (position): Expression level (quantitative) ✓
- Color (hue): Experimental condition (categorical) ✓

Problem: X-axis position (strongest channel) wasted on arbitrary gene IDs

Improved encoding:
- X-axis (position): Time point

```


## 1.3 Gestalt Principles in Figure Design (Expanded)

### The Law of Prägnanz (Simplicity)

Beyond the basic Gestalt principles introduced in 1.1, the overarching **Law of Prägnanz** states that people perceive and interpret ambiguous or complex images in the simplest form possible. Our brains seek to impose order and structure on visual information.

**Implication for Scientific Figures:**
- Viewers will try to find patterns even in random data
- Clear, simple structures are processed faster and remembered better
- Unnecessary complexity causes cognitive strain

### Advanced Gestalt Applications

#### Common Fate
**Principle:** Elements moving in the same direction are perceived as a group.

**Static Figure Applications:**

```
Line graphs with multiple time series:
✓ Lines following similar trajectories naturally group together
✓ Diverging lines signal important differences
✓ Use this to your advantage: arrange panels so similar patterns are adjacent

Example:
Panel A: All upward trends (treatment responders)
Panel B: All downward trends (treatment non-responders)
→ Visual coherence within each panel strengthens message
```

**Code Example (Python):**

```
import matplotlib.pyplot as plt
import numpy as np

# Demonstrate common fate grouping
time = np.linspace(0, 10, 50)

# Group 1: Similar upward trajectories (common fate)
group1 = [time * 0.8 + np.random.randn(50) * 0.5,
          time * 0.9 + np.random.randn(50) * 0.5,
          time * 0.85 + np.random.randn(50) * 0.5]

# Group 2: Similar flat trajectories (common fate)
group2 = [5 + np.random.randn(50) * 0.5,
          5.5 + np.random.randn(50) * 0.5,
          4.8 + np.random.randn(50) * 0.5]

fig, ax = plt.subplots(figsize=(8, 5))

# Plot group 1 in shades of red (responders)
for i, trajectory in enumerate(group1):
    ax.plot(time, trajectory, color=plt.cm.Reds(0.5 + i*0.15),
            linewidth=2, label=f'Responder {i+1}' if i < 1 else '')

# Plot group 2 in shades of blue (non-responders)
for i, trajectory in enumerate(group2):
    ax.plot(time, trajectory, color=plt.cm.Blues(0.5 + i*0.15),
            linewidth=2, label=f'Non-responder {i+1}' if i < 1 else '')

ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('Biomarker Level', fontsize=12)
ax.set_title('Common Fate: Trajectories Naturally Group', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', frameon=True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('common_fate_example.png', dpi=300, bbox_inches='tight')
```

#### Symmetry and Balance
**Principle:** Symmetric elements are perceived as belonging together and forming a coherent whole.

**Applications in Multi-Panel Figures:**

```
✓ Symmetric grid layouts feel organized and professional
✓ Asymmetry draws attention (use deliberately for emphasis)

Example layouts:

Symmetric (balanced, harmonious):
[Panel A] [Panel B]
[Panel C] [Panel D]

Asymmetric for emphasis (main result gets more space):
[Panel A - large    ] [B]
[Panel C - large    ] [D]
→ Panels A & C visually dominate
```

**Code Example (R):**

```
library(ggplot2)
library(patchwork)

# Create sample plots
p1 <- ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  labs(title = "Panel A: Main Result") +
  theme_classic()

p2 <- ggplot(mtcars, aes(x = hp, y = mpg)) +
  geom_point() +
  labs(title = "Panel B: Control") +
  theme_classic()

p3 <- ggplot(mtcars, aes(x = disp, y = mpg)) +
  geom_point() +
  labs(title = "Panel C: Supplementary") +
  theme_classic()

p4 <- ggplot(mtcars, aes(x = qsec, y = mpg)) +
  geom_point() +
  labs(title = "Panel D: Supplementary") +
  theme_classic()

# Symmetric layout
symmetric <- (p1 + p2) / (p3 + p4)

# Asymmetric layout (emphasize A and C)
asymmetric <- (p1 + p2) / (p3 + p4) +
  plot_layout(heights = c(2, 1))

ggsave("symmetric_layout.png", symmetric, width = 8, height = 8, dpi = 300)
ggsave("asymmetric_layout.png", asymmetric, width = 8, height = 8, dpi = 300)
```

#### Connectedness
**Principle:** Elements connected by lines or enclosed in shapes are perceived as more strongly grouped than mere proximity.

**Applications:**

```
✓ Use lines to connect related data points (time series)
✓ Use boxes/frames to group related panels
✓ Use arrows to show causal relationships

Example:
Before/After comparison:
[Before image] --arrow--> [After image]
→ Arrow creates stronger association than just placement
```

---

## 1.4 Visual Hierarchy and Focus

### Creating Effective Visual Hierarchy

**Visual hierarchy** is the arrangement of elements in order of importance, guiding the viewer's attention through the figure in a deliberate sequence.

### The Three-Level Hierarchy

Every effective figure has three levels:

#### Level 1: Primary Focus (The Message)
**This should be seen first, within 1-2 seconds.**

**Techniques to create primary focus:**
1. **High contrast** - Make it darker, brighter, or more saturated than everything else
2. **Larger size** - Primary element should be visually dominant
3. **Central position** - Eye naturally goes to center first
4. **Enclosure** - Box, circle, or annotation around key element
5. **Isolation** - Surround with white space

**Example:**

```
Scatter plot showing drug efficacy:
Primary focus: The significant outlier responders
→ Make them larger, brighter color, add annotation arrow

Secondary: All other data points
→ Smaller, gray, lower opacity

Tertiary: Grid, axes, legend
→ Lightest gray, thin lines
```

**Code Example (Python):**

```
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Generate data
x = np.random.randn(100)
y = 2*x + np.random.randn(100)

# Identify "significant" outliers
outlier_idx = np.where((y > 4) | (y < -4))[0]

# Create figure with visual hierarchy
fig, ax = plt.subplots(figsize=(8, 6))

# Level 3: Grid (most subtle)
ax.grid(True, alpha=0.2, linewidth=0.5, color='gray', zorder=0)

# Level 2: Regular data points (secondary)
regular_idx = np.setdiff1d(np.arange(len(x)), outlier_idx)
ax.scatter(x[regular_idx], y[regular_idx],
           s=30, color='lightgray', alpha=0.6, zorder=2, label='Regular')

# Level 1: Outliers (PRIMARY FOCUS)
ax.scatter(x[outlier_idx], y[outlier_idx],
           s=200, color='#E63946', alpha=0.9, zorder=3,
           edgecolors='darkred', linewidths=2, label='Outliers')

# Annotate one key outlier
if len(outlier_idx) > 0:
    key_outlier = outlier_idx[0]
    ax.annotate('Key responder',
                xy=(x[key_outlier], y[key_outlier]),
                xytext=(x[key_outlier]-1, y[key_outlier]+1),
                fontsize=11, fontweight='bold', color='#E63946',
                arrowprops=dict(arrowstyle='->', color='#E63946', lw=2),
                zorder=4)

# Styling (de-emphasize axes - level 3)
ax.set_xlabel('Drug Concentration (μM)', fontsize=11)
ax.set_ylabel('Cell Response (AU)', fontsize=11)
ax.set_title('Visual Hierarchy: Outliers Stand Out', fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_color('gray')
ax.legend(loc='lower right', frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig('visual_hierarchy.png', dpi=300, bbox_inches='tight')

```

#### Level 2: Supporting Context
**Provides necessary information but doesn't compete with primary focus.**

**Elements at this level:**
- Main data that isn't the key finding
- Axis labels and titles
- Legend (if not redundant)
- Reference lines or regions

**Techniques:**
- Medium contrast
- Standard sizing
- Neutral colors (blacks, grays, muted tones)

#### Level 3: Infrastructure
**Essential for interpretation but should fade into background.**

**Elements at this level:**
- Grid lines
- Minor tick marks
- Axis lines
- Background shading

**Techniques:**
- Low contrast (light grays)
- Thin line weights
- Semi-transparency

### Contrast as the Primary Tool

**Types of contrast for creating hierarchy:**

**1. Value Contrast (Light vs. Dark)**

```
✓ Dark data on light background (most common)
✓ Light data on dark background (for presentations/posters)
✗ Medium gray data on medium gray background (no contrast)
```

**2. Saturation Contrast**

```
✓ Saturated color for key data, desaturated for context
Example: Bright red key points, pale pink supporting data
```

**3. Size Contrast**

```
✓ Larger primary elements, smaller supporting elements
Rule of thumb: 2:1 or 3:1 size ratio minimum
```

**4. Weight Contrast**

```
✓ Bold/thick for primary, light/thin for secondary
Applies to: lines, fonts, borders
```

### Common Hierarchy Mistakes

#### Mistake 1: Everything is Emphasized (Nothing Stands Out)

**Bad Example:**

```
✗ All data points large and bright
✗ Bold axis labels
✗ Thick grid lines
✗ Multiple colors all saturated
→ Result: Visual chaos, no clear message
```

**Fix:**

```
✓ Choose ONE element as primary focus
✓ Make everything else subordinate
✓ Ask: "If viewer sees only one thing, what should it be?"
```

#### Mistake 2: Wrong Element is Emphasized

**Bad Example:**

```
✗ Huge, bold legend in bright colors
✗ Decorative border or logo prominently featured
✗ Actual data is small and gray
→ Result: Reader focuses on metadata, not results
```

**Fix:**

```
✓ Data ink >> non-data ink (Tufte's principle)
✓ Legend should be smallest readable size
✓ Branding/logos should be minimal or absent
```

#### Mistake 3: Competing Focal Points

**Bad Example:**

```
✗ Three different elements all trying to be #1:
  - Large red title
  - Bright blue highlighted region
  - Orange outlier points
→ Result: Eye doesn't know where to go first
```

**Fix:**

```
✓ Clear priority: One primary (e.g., outliers)
✓ Others as secondary (title standard weight, region subtle shading)
✓ Use color consistently, not for multiple purposes
```

---

### Exercise 1.4.1: Visual Hierarchy Redesign

**Objective:** Transform a flat figure into one with clear visual hierarchy

**Materials:**
- Take a figure you've created (or find one online) where everything has similar visual weight

**Task:**

1. **Audit current state:**
   - List all visual elements in the figure
   - Rate each on "attention-grabbing" scale (1-10)
   - Identify if multiple elements score 8-10 (competing focus problem)

2. **Define hierarchy:**
   - What is THE main message? (This becomes Level 1)
   - What supports understanding? (Level 2)
   - What is infrastructure? (Level 3)

3. **Redesign with tools:**
   - **For Level 1:** Increase size by 2x, use high-contrast color, add annotation
   - **For Level 2:** Use medium gray or neutral colors, standard sizes
   - **For Level 3:** Reduce to light gray (alpha=0.2-0.3), thin lines

4. **Test:**
   - Show both versions to a colleague for 3 seconds each
   - Ask: "What did you notice first?"
   - Did the redesign direct attention correctly?

**Example Transformation:**

*Before:*
- All scatter points same size (50 pixels)
- All points same color (blue)
- Grid lines same thickness as data trendline
- Title and axis labels same font weight

*After:*
- Key points: 150 pixels, red
- Other points: 30 pixels, light gray
- Grid lines: alpha=0.2, thin
- Trendline: Bold, black
- Title: Bold, axis labels: regular weight

---

### **1.5 Theme Base Size and Font Scaling**

**Principle:** Set appropriate base font sizes that account for potential figure reduction.

```python
import matplotlib.pyplot as plt
import numpy as np

# Font size calculation for figure reduction

"""
FONT SIZE PLANNING:

Journals often reduce figures to fit column width:
- 50% reduction common (7" → 3.5")
- Font sizes reduce proportionally

Minimum readable size: 6-7 points after reduction

Calculation:
If figure reduced 50%, you need DOUBLE the font size:
- Want 8pt final → Use 16pt in original
- Want 10pt final → Use 20pt in original

Safe starting sizes (for 50% reduction):
- Title: 24-28pt → becomes 12-14pt ✓
- Axis labels: 20-22pt → becomes 10-11pt ✓
- Tick labels: 16-18pt → becomes 8-9pt ✓
- Legend: 16-18pt → becomes 8-9pt ✓
"""

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: TOO SMALL - Will be unreadable after reduction
ax1 = axes[0, 0]
x = np.linspace(0, 10, 50)
y = 2*x + np.random.randn(50)*2

ax1.scatter(x, y, s=30, color='#3498DB', alpha=0.7)
ax1.set_xlabel('Variable X (units)', fontsize=8)  # TOO SMALL
ax1.set_ylabel('Variable Y (units)', fontsize=8)  # TOO SMALL
ax1.set_title('Too Small Fonts', fontsize=10)  # TOO SMALL
ax1.tick_params(labelsize=7)  # TOO SMALL
ax1.grid(alpha=0.3)

ax1.text(0.5, 1.15, '❌ BAD: After 50% reduction\nTitle→5pt, Labels→4pt (UNREADABLE)',
        transform=ax1.transAxes, ha='center', fontsize=10,
        color='red', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#FFCCCC', alpha=0.8))

# Panel B: CORRECT - Accounts for reduction
ax2 = axes[0, 1]
ax2.scatter(x, y, s=50, color='#27AE60', alpha=0.7, edgecolors='black', linewidths=0.5)
ax2.set_xlabel('Variable X (units)', fontsize=14, fontweight='bold')  # GOOD
ax2.set_ylabel('Variable Y (units)', fontsize=14, fontweight='bold')  # GOOD
ax2.set_title('Correct Font Sizes', fontsize=16, fontweight='bold')  # GOOD
ax2.tick_params(labelsize=12)  # GOOD
ax2.grid(alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.text(0.5, 1.15, '✓ GOOD: After 50% reduction\nTitle→8pt, Labels→7pt (READABLE)',
        transform=ax2.transAxes, ha='center', fontsize=10,
        color='green', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#CCFFCC', alpha=0.8))

# Panel C: Theme base_size comparison
ax3 = axes[1, 0]

# Matplotlib rcParams approach
plt.rcParams.update({
    'font.size': 18,  # Base size accounting for reduction
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

ax3.scatter(x, y, s=50, color='#3498DB', alpha=0.7)
ax3.set_xlabel('Variable X (units)', fontweight='bold')
ax3.set_ylabel('Variable Y (units)', fontweight='bold')
ax3.set_title('Using rcParams Base Size', fontweight='bold')
ax3.grid(alpha=0.3)

code_example = """
# Set once at script start:
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 22

# All subsequent plots inherit these sizes
"""
ax3.text(0.5, -0.25, code_example, transform=ax3.transAxes,
        ha='center', fontsize=9, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

# Panel D: Test print guide
ax4 = axes[1, 1]
ax4.axis('off')

test_guide = """
FONT SIZE TEST PROCEDURE:

1. Create figure at FINAL SIZE
   fig, ax = plt.subplots(figsize=(3.5, 2.5))
   # Not (7, 5) if it will be reduced to (3.5, 2.5)!

2. Set fonts for FINAL SIZE
   ax.set_xlabel('Label', fontsize=11)
   # Not 22 if figure already at final size

3. Save and PRINT at 100% scale
   plt.savefig('test.pdf', dpi=300)
   # Print without "fit to page"

4. Check readability:
   • Can you read smallest text from 1 foot away?
   • Are subscripts/superscripts clear?
   • Is legend distinguishable?

5. If NO to any → Increase font sizes

ALTERNATIVE: Scale test
• Create at large size (7×5)
• Print and physically reduce by 50%
• Check readability
"""

ax4.text(0.05, 0.95, test_guide, transform=ax4.transAxes,
        verticalalignment='top', fontsize=10, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax4.set_title('Font Size Testing Guide',
              fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('font_size_reduction_planning.png', dpi=300,
           bbox_inches='tight', facecolor='white')
plt.close()

# Reset rcParams
plt.rcParams.update(plt.rcParamsDefault)
```

**R equivalent:**

```r
library(ggplot2)

# Base size in ggplot2 theme
# Will scale all text proportionally

# TOO SMALL (will be unreadable after reduction)
p_bad <- ggplot(data, aes(x, y)) +
  geom_point(size = 2, color = '#3498DB', alpha = 0.7) +
  labs(x = 'Variable X (units)',
       y = 'Variable Y (units)',
       title = 'Too Small Fonts') +
  theme_classic(base_size = 8) +  # TOO SMALL!
  theme(plot.title = element_text(face = 'bold'))

# GOOD (accounts for 50% reduction)
p_good <- ggplot(data, aes(x, y)) +
  geom_point(size = 3, color = '#27AE60', alpha = 0.7) +
  labs(x = 'Variable X (units)',
       y = 'Variable Y (units)',
       title = 'Correct Font Sizes') +
  theme_classic(base_size = 16) +  # GOOD!
  theme(
    plot.title = element_text(face = 'bold', size = 20),
    axis.title = element_text(face = 'bold', size = 18),
    axis.text = element_text(size = 14)
  )

# Save at final intended size
ggsave('figure_correct_size.png', p_good,
       width = 3.5, height = 2.5,  # Final size, not pre-reduction
       dpi = 300)
```

---

## 1.6 Common Perceptual Traps and How to Avoid Them

### Trap 1: The Truncated Axis

**The Problem:**
When y-axis doesn't start at zero for bar charts, small differences appear exaggerated.

**Example of Misuse:**

```
Sales data:
Competitor A: 98 units
Our product: 100 units

✗ BAD: Y-axis from 95-100
→ Bar for our product appears 2.5x taller (visual difference 5 units out of 5)
→ Actual difference: 2% (100 vs 98)
```

**When It's Acceptable:**

```
✓ Line graphs (emphasizing trend over absolute values)
✓ Scatter plots (focusing on relationship)
✓ When differences ARE meaningful relative to natural variation

Example: Temperature anomalies
- Baseline: 15.0°C
- Year 1: 15.1°C
- Year 2: 15.3°C
✓ OK to zoom to 14.8-15.5°C range
→ These small changes are scientifically significant
```

**Code Example - Demonstrating the Effect:**

```
import matplotlib.pyplot as plt
import numpy as np

categories = ['Competitor A', 'Our Product']
values = [98, 100]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# MISLEADING: Truncated axis for bar chart
axes[0].bar(categories, values, color=['gray', 'steelblue'])
axes[0].set_ylim(95, 101)
axes[0].set_ylabel('Sales (units)')
axes[0].set_title('MISLEADING: Truncated Axis\n(Appears 5x difference)',
                   fontsize=11, color='red')
axes[0].axhline(0, color='black', linewidth=0.8)

# HONEST: Full axis from zero
axes[1].bar(categories, values, color=['gray', 'steelblue'])
axes[1].set_ylim(0, 110)
axes[1].set_ylabel('Sales (units)')
axes[1].set_title('HONEST: Full Axis\n(Shows 2% difference)',
                   fontsize=11, color='green')

plt.tight_layout()
plt.savefig('truncated_axis_trap.png', dpi=300, bbox_inches='tight')
```

**Solution:**

```
For bar charts:
✓ Always start at zero (or explicitly show break if necessary)
✓ Add context: "Values differ by 2% (98 vs 100)"

For line graphs:
✓ Truncation is OK but add reference line at zero
✓ Clearly label axis range
✓ Consider showing full range in inset panel
```

### Trap 2: The Dual Axis Deception

**The Problem:**
Dual y-axes allow arbitrary scaling that can manufacture correlations or hide them.

**Example of Misuse:**

```
✗ Graph showing "Ice cream sales" and "Drowning deaths" over months
→ Both increase in summer
→ Dual axes scaled to make lines overlap perfectly
→ Implies causal relationship (which doesn't exist)
```

**Why It's Problematic:**
```
1. No inherent relationship between the two scales
2. Can be manipulated to show any pattern desired
3. Viewer cannot judge relative magnitudes
4. Confuses correlation with causation
```

**Better Alternatives:**

```
✓ Option 1: Two separate plots stacked vertically
  → Allows visual comparison without misleading single axis

✓ Option 2: Normalize both variables (0-100% scale or z-scores)
  → Same scale, honest comparison

✓ Option 3: Scatter plot (X = variable 1, Y = variable 2)
  → Shows relationship directly

✓ Option 4: If truly related, use secondary axis BUT:
  - Use same scale factor (e.g., both per capita)
  - Make relationship explicit ("°F" vs "°C" - mathematically linked)
```

**Code Example (R):**

```
library(ggplot2)
library(patchwork)

# Sample data
months <- 1:12
ice_cream <- c(20, 25, 40, 55, 70, 85, 90, 85, 65, 45, 30, 22)
drownings <- c(2, 3, 5, 8, 12, 15, 16, 14, 10, 6, 4, 2)

data <- data.frame(month = months, ice_cream = ice_cream, drownings = drownings)

# BAD: Dual axis (manipulable)
p_bad <- ggplot(data, aes(x = month)) +
  geom_line(aes(y = ice_cream), color = "orange", size = 1.5) +
  geom_line(aes(y = drownings * 6), color = "blue", size = 1.5) +  # Scaled to overlap
  scale_y_continuous(
    name = "Ice Cream Sales",
    sec.axis = sec_axis(~./6, name = "Drownings")
  ) +
  labs(title = "MISLEADING: Dual Axis (Scales Manipulated)") +
  theme_classic()

# BETTER: Separate panels
p1 <- ggplot(data, aes(x = month, y = ice_cream)) +
  geom_line(color = "orange", size = 1.5) +
  labs(y = "Ice Cream Sales", title = "Ice Cream Sales") +
  theme_classic()

p2 <- ggplot(data, aes(x = month, y = drownings)) +
  geom_line(color = "blue", size = 1.5) +
  labs(y = "Drownings", x = "Month", title = "Drowning Deaths") +
  theme_classic()

p_good <- p1 / p2 + plot_annotation(title = "BETTER: Separate Panels (Honest Comparison)")

ggsave("dual_axis_bad.png", p_bad, width = 7, height = 4, dpi = 300)
ggsave("dual_axis_good.png", p_good, width = 7, height = 6, dpi = 300)

```

### Trap 3: Cherry-Picked Axis Ranges

**The Problem:**

Selectively choosing axis ranges to emphasize or hide patterns.

**Example:**

```
Clinical trial results:

Full timeline (0-24 months):
→ Drug effect appears temporary, returns to baseline

Cropped timeline (0-6 months):
→ Drug appears highly effective
→ Hides the fact that effect disappears

✗ Publishing only the 6-month view is misleading
```

**Solution:**

```
✓ Show complete temporal or spatial range relevant to the study
✓ If cropping for focus, include inset with full range
✓ State explicit rationale for range choice
✓ Provide supplementary figures with full data
```

### Trap 4: Logarithmic Scale Without Clear Indication

**The Problem:**
Log scales dramatically change visual perception but are easy to miss.

**Example:**

```
Linear scale:
1, 2, 3, 4, 5, 10, 100, 1000
→ Visually: huge jump to 100

Log scale:
10⁰, 10¹, 10², 10³
→ Visually: even spacing
→ Can hide massive actual differences
```

**When Logarithmic Scales are Appropriate:**

```
✓ Data spanning multiple orders of magnitude
✓ Exponential growth/decay processes
✓ Multiplicative effects (fold-changes)
✓ When relative changes matter more than absolute

Examples:
- Gene expression (ranges from 0.01 to 10,000)
- Earthquake magnitudes (Richter scale)
- Sound intensity (decibels)
- Drug concentrations (dose-response curves)
```

**How to Use Responsibly:**

```
✓ Label axis clearly: "Log₁₀ Concentration"
✓ Show actual values on tick labels (1, 10, 100) not (0, 1, 2)
✓ Mention in caption: "Note logarithmic scale"
✓ Consider showing both linear and log versions
```

**Code Example (Python):**

```
import matplotlib.pyplot as plt
import numpy as np

# Exponential data
x = np.linspace(0, 10, 50)
y = np.exp(x/2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Linear scale
axes[0].plot(x, y, 'o-', color='steelblue', linewidth=2, markersize=4)
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Cell Count')
axes[0].set_title('LINEAR Scale\n(Emphasizes recent growth)', fontsize=12)
axes[0].grid(alpha=0.3)

# Log scale
axes[1].plot(x, y, 'o-', color='coral', linewidth=2, markersize=4)
axes[1].set_yscale('log')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Cell Count (log scale)', fontweight='bold')
axes[1].set_title('LOGARITHMIC Scale\n(Shows exponential trend clearly)', fontsize=12)
axes[1].grid(alpha=0.3, which='both')

# Highlight log scale usage
axes[1].text(0.5, 0.95, 'NOTE: Log scale', transform=axes[1].transAxes,
             fontsize=11, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('log_scale_comparison.png', dpi=300, bbox_inches='tight')
```

### Trap 5: The 3D Disaster

**The Problem:**
3D charts for 2D data introduce perspective distortion and occlusion without adding information.

**Common Offenders:**
- 3D pie charts
- 3D bar charts
- 3D scatter plots (when Z-axis is unused)

**Why They Fail:**

```
1. Perspective makes closer elements appear larger
2. Back elements occluded by front elements
3. Difficult to read exact values
4. Chartjunk (unnecessary visual complexity)
5. "Cool factor" often motivates use, not data needs
```

**Example:**

```
✗ 3D pie chart:
→ Slice in foreground appears larger than identical slice in background
→ Pure perceptual distortion

✓ 2D pie chart (or better, bar chart):
→ Accurate representation
→ Easy comparison
```

**When 3D is Justified:**
```
✓ Genuine spatial data (protein structures, geographic terrain)
✓ Physical specimens (medical imaging)
✓ Three independent quantitative variables

Even then, consider:
- Providing multiple 2D views (orthogonal projections)
- Interactive 3D (can rotate) for digital supplements
- Color/size encoding on 2D projection as alternative
```

---

### Exercise 1.6.1: Perceptual Trap Detection

**Objective:** Develop critical eye for misleading visualizations

**Materials:** News articles, advertisements, or scientific papers with quantitative graphics

**Task:**

1. **Find 5 figures** from various sources (news media, research papers, company reports)

2. **For each, check for these traps:**
   - [ ] Truncated axis (bar chart not starting at zero)
   - [ ] Dual axes (with manipulated scaling)
   - [ ] Cherry-picked range (timeline suspiciously cropped)
   - [ ] Unlabeled log scale
   - [ ] Unnecessary 3D
   - [ ] Misleading aspect ratio (tall/skinny vs. wide/short changes perception of slope)

3. **Analyze intent:**
   - Is this an honest mistake or deliberate distortion?
   - What is the figure trying to emphasize?
   - How would fixing the issue change the message?

4. **Redesign one:**
   - Choose the most misleading figure
   - Create an honest version
   - Write 2-3 sentences on how perception changes

**Example Analysis:**

```
Figure: Bar chart from Company X press release
Trap detected: Y-axis starts at 95%, goes to 100%
Effect: 96% → 98% improvement appears massive (fills whole chart)
Honest version: Y-axis 0-100% shows 2 percentage point change (barely visible)
Conclusion: Likely deliberate to exaggerate modest improvement
```

---

## Chapter 1 Summary

### Key Principles from Visual Perception Science

1. **Pre-attentive processing** guides immediate attention
   - Use color, size, position strategically
   - Key findings should "pop out" without effort

2. **Cleveland-McGill hierarchy** informs encoding choices
   - Position > Length > Angle > Area > Color hue
   - Match strongest channels to most important data

3. **Gestalt principles** create coherent organization
   - Proximity, similarity, continuity, closure
   - Work with brain's natural grouping tendencies

4. **Visual hierarchy** directs viewer's eye
   - Three levels: Primary focus, supporting context, infrastructure
   - Use contrast (value, saturation, size, weight) to differentiate

5. **Working memory limits** constrain complexity
   - 7±2 items maximum simultaneously
   - Simplify, chunk, or use small multiples

6. **Perceptual biases** require vigilance
   - We underestimate area differences
   - We perceive logarithmically
   - Dual axes and truncated ranges can mislead

### Practical Application Checklist

Before finalizing any figure, ask:

- [ ] **Can a naive viewer identify the main message in 3 seconds?**
- [ ] **Are the strongest visual channels used for the most important data?**
- [ ] **Does the layout group related elements (Gestalt proximity/similarity)?**
- [ ] **Is there a clear visual hierarchy (one primary focus)?**
- [ ] **Are comparisons easy (common scales, aligned axes)?**
- [ ] **Is the figure free from perceptual traps (truncation, dual axes, unnecessary 3D)?**
- [ ] **Would this work in grayscale (redundant encoding for accessibility)?**
- [ ] **Does it require less than 7 simultaneous comparisons?**

### Transition to Chapter 2

Now that we understand **how humans perceive visual information**, we can make informed decisions about **specific design elements**.

**Chapter 2: The Language of Color** will build on these perceptual foundations to explore:
- Color theory and color spaces
- Choosing palettes for different data types
- Accessibility and color-blind friendly design
- Avoiding common color mistakes

You now have the cognitive science framework; next we apply it to one of the most powerful (and most misused) visual channels: **color**.

---
