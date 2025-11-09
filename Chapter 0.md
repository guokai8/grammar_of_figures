## **Grammar of Figures: The Art and Science of Visualizing Data for Publication**

### **Preface**
- The critical role of figures in scientific communication
- Target audience: researchers, graduate students, data scientists preparing work for publication
- How to use this book (workflow-oriented approach)
- About the code examples (Python & R)


# Chapter 0: Foundations – Purpose, Audience & Workflow

## 0.1 Why Figures Matter: Communication Before Decoration

### The Central Role of Figures in Scientific Communication

In modern scientific literature, figures have evolved from mere supplementary illustrations to primary communication vehicles. Research shows that readers typically engage with a paper by first scanning the abstract, then examining the figures, and only afterward reading the full text. This "figure-first" reading pattern reflects a fundamental truth: **a well-designed figure can convey complex relationships, patterns, and evidence more efficiently than paragraphs of text.**

Consider this: a single scatter plot with error bars can simultaneously communicate:
- The relationship between two variables
- The distribution of data points
- The uncertainty in measurements
- The sample size
- Outliers or unusual patterns
- Statistical trends

Conveying the same information in text would require dense prose that few readers would parse with the same comprehension.

### The Cost of Poor Figures

Conversely, poorly designed figures impose significant costs:

**Bad Example Characteristics:**
- **Cognitive overload**: Too many colors, overlapping points, cluttered legends that force readers to decode rather than understand
- **Ambiguity**: Missing units, unlabeled axes, or unclear symbols that leave readers guessing
- **Misleading representation**: Truncated y-axes, inappropriate plot types, or manipulated scales that distort the data's story
- **Inaccessibility**: Color schemes that fail for colorblind readers, text too small after publication sizing, or unnecessarily complex visualizations

**Consequences:**
- Reviewers may reject your manuscript based on figure quality alone
- Your findings may be misunderstood or ignored
- Replication efforts may fail due to unclear methodology shown in figures
- Your work may receive fewer citations if readers cannot quickly extract value

### Good Figures: What Sets Them Apart

Excellent scientific figures share common attributes:

1. **Clarity of message**: The main point is immediately apparent
2. **Appropriate complexity**: No simpler than necessary, no more complex than required
3. **Self-sufficiency**: Figure + caption can stand alone without main text
4. **Visual integrity**: Honest representation without distortion
5. **Accessibility**: Readable by diverse audiences including those with visual impairments
6. **Professional polish**: Consistent styling, appropriate resolution, clean execution

**Example of Excellence:**
Think of landmark papers in your field. Their figures likely:
- Use consistent color schemes throughout
- Have legible text at publication size
- Employ white space effectively
- Guide your eye through a logical visual flow
- Make statistical comparisons obvious
- Include all necessary information without clutter

### Exercise 0.1.1
*Find three published figures in your field:*
1. One you consider excellent – list three specific design choices that make it effective
2. One that confused you initially – identify what made it difficult to understand
3. One that could be improved – sketch how you would redesign it

---

## 0.2 Define Before You Design: Message, Audience & Context

### The Three Questions Framework

Before opening your plotting software, answer these three fundamental questions:

#### Question 1: What is the ONE main message?

Every figure should have a primary message – a single key takeaway. Secondary details can exist, but must not compete with the main point.

**Good Example:**
- *Main message*: "Drug A reduces tumor growth more effectively than Drug B over 4 weeks"
- *Supporting details*: Variability in response, dose-dependent effects, time-course dynamics

**Bad Example (multiple competing messages):**
- Trying to show: treatment effects + dose response + time course + molecular mechanisms + patient demographics all in one figure
- *Result*: Cognitive overload, unclear priorities, no clear conclusion

**Exercise: The Elevator Test**
Write your figure's message in one sentence. If you cannot, your figure probably tries to communicate too much. Consider splitting it into multiple panels or separate figures.

#### Question 2: Who is your audience?

Different audiences require different approaches:

| Audience Type | Characteristics | Figure Strategy |
|--------------|----------------|-----------------|
| **Domain Experts** | Deep knowledge of methods, terminology, conventions | Can use specialized plot types, assume familiarity with standard representations, include technical details |
| **Interdisciplinary Scientists** | Scientific literacy but not domain-specific expertise | Require more context, clearer labels, standard plot types, avoid jargon in labels |
| **Educated General Public** | Basic scientific understanding | Need simple visualizations, extensive annotation, intuitive color schemes, metaphorical representations |
| **Students/Trainees** | Learning the field | Benefit from pedagogical clarity, step-by-step visual logic, examples of what to look for |

**Example Scenario:**
You have RNA sequencing data showing differential gene expression.

- **For specialists (molecular biology journal)**: Volcano plot with gene names, log2 fold change, adjusted p-values – standard representation
- **For interdisciplinary audience (Science/Nature)**: Side-by-side bar charts of top 10 upregulated/downregulated genes with fold-change values and clear labels like "Genes increased in treated cells"
- **For general public (university press release)**: Simple pictogram showing affected biological pathways with metaphorical illustrations

#### Question 3: Where will this figure appear?

Context shapes design decisions:

**Print Journal (single column: ~85mm width)**
- Design at final size
- 8-point minimum font size
- Limited colors (cost considerations historically)
- 300+ DPI for images
- Consider grayscale printing

**Print Journal (double column: ~180mm width)**
- More room for complexity
- Multi-panel layouts work well
- Can include more detail
- Still need high resolution

**Digital/Screen Display**
- RGB color space (vs CMYK for print)
- Can use interactive elements (in online versions)
- Lower DPI acceptable (96-150)
- Attention to screen contrast

**Poster Presentation**
- Viewed from 1-2 meters away
- Larger fonts (minimum 18-24 point)
- High contrast
- Simplified design
- More visual impact, less detail

**Slide Presentation**
- Minimal text
- High contrast for projector visibility
- One point per slide
- Animation can help reveal complexity
- Legible from back of room

### The Context-Driven Design Process

**Step-by-step workflow:**

1. **Start with the message**: Write it out explicitly
2. **Know your venue**: Check journal/conference requirements *before* designing
3. **Sketch first**: Rough draft on paper or whiteboard
4. **Consider alternatives**: Generate 2-3 different approaches
5. **Get feedback**: Show sketches to a colleague (preferably from your target audience)
6. **Code/design**: Only now open your software
7. **Test at size**: View your figure at actual publication size
8. **Iterate**: Refine based on the test

### Exercise 0.2.1
*Take a figure you're currently working on (or plan to create):*

1. Write out your message in one sentence
2. Describe your audience in 2-3 sentences
3. Specify the context (journal name, column width, or presentation venue)
4. List 3 design constraints this context imposes
5. Sketch two different ways to visualize this message
6. Which approach better serves your message + audience + context? Why?

---

## 0.3 The Iterative Design Process

### Why Iteration Matters

**No one creates a perfect figure on the first try.** Just as you revise your manuscript text multiple times, figures require iterative refinement. The difference is that figures fail in ways that are often invisible to their creators – we become blind to our own design choices.

### The Five-Stage Iterative Workflow

#### Stage 1: Sketch & Conceptualize (Paper/Whiteboard)

**Before touching code or software:**

- Sketch 3-5 rough layouts quickly (5 minutes each)
- Don't worry about accuracy or beauty
- Focus on structure: what goes where?
- Try different arrangements of the same data
- Ask: What catches the eye first in each version?

**Benefits:**
- Fast exploration without technical constraints
- Easy to share and discuss
- Reveals conceptual problems early
- Prevents tunnel vision on a single approach

**Example:**
For a time-series experiment with 3 treatment groups:
- *Sketch 1*: Three separate line plots stacked vertically
- *Sketch 2*: All three lines on one plot with different colors
- *Sketch 3*: Small multiples showing each group + control in separate panels
- *Sketch 4*: Combination plot: overview + detail insets

#### Stage 2: Draft & Prototype (Initial Code)

**Create a functional but not polished version:**

```python
# Python example - quick draft
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.linspace(0, 10, 50)
y1 = np.sin(x) + np.random.normal(0, 0.1, 50)
y2 = np.cos(x) + np.random.normal(0, 0.1, 50)

# Quick draft plot
fig, ax = plt.subplots()
ax.plot(x, y1, label='Treatment A')
ax.plot(x, y2, label='Treatment B')
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Response')
plt.tight_layout()
plt.savefig('draft_v1.png', dpi=150)
```

**At this stage:**
- Use default colors and styles
- Focus on getting data correctly represented
- Ensure all necessary elements are present
- Check that the data tell the story you intend

**Don't worry about:**
- Perfect aesthetics
- Final color choices
- Publication-quality resolution
- Typography details

#### Stage 3: Review & Critique (External Feedback)

**This is the most important stage and the one most researchers skip.**

**Who to ask:**
- **Colleague in your field**: Can they extract the main message in 10 seconds?
- **Scientist from different field**: Is it interpretable without domain knowledge?
- **Lab junior member**: Does it make sense to someone learning?

**Questions to ask reviewers:**
1. "What is the main point of this figure?" (without telling them first)
2. "What draws your attention first?"
3. "What confuses you or requires explanation?"
4. "If you could change one thing, what would it be?"
5. "Is there anything you expected to see that's missing?"

**How to handle feedback:**
- Don't defend your choices – listen
- Take notes on first impressions (they're most valuable)
- Look for patterns if multiple reviewers mention the same issue
- Not all feedback is equal – prioritize comments about clarity over aesthetics

**Example Feedback Session:**
```
You: "What's the main message?"
Reviewer: "Um... that the blue line is higher than the red line?"
You: (thinking: Oh no, I wanted them to notice the divergence at week 3)
→ ACTION: Need to add annotation highlighting week 3, or use panel layout emphasizing that timepoint
```

#### Stage 4: Refine & Polish (Implement Improvements)

**Now focus on quality and professionalism:**

```python
# Python example - polished version
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# Data (same as before)
x = np.linspace(0, 10, 50)
y1 = np.sin(x) + np.random.normal(0, 0.1, 50)
y2 = np.cos(x) + np.random.normal(0, 0.1, 50)

# Create figure at publication size
fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Single column width

# Plot with carefully chosen colors
color1 = '#2E86AB'  # Blue - colorblind safe
color2 = '#A23B72'  # Purple - colorblind safe

ax.plot(x, y1, color=color1, linewidth=2, label='Treatment A', zorder=3)
ax.plot(x, y2, color=color2, linewidth=2, label='Treatment B', zorder=3)

# Highlight key region (week 3 equivalent)
ax.axvspan(5, 7, alpha=0.1, color='gray', zorder=1)
ax.text(6, 1.2, 'Critical\nperiod', ha='center', fontsize=9,
        style='italic', color='dimgray')

# Clean up axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Time (weeks)', fontsize=10, fontweight='bold')
ax.set_ylabel('Response (arbitrary units)', fontsize=10, fontweight='bold')

# Legend with good placement
ax.legend(frameon=True, loc='upper left', fontsize=9)

# Adjust layout
plt.tight_layout()

# Save at publication quality
plt.savefig('figure_final.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure_final.png', dpi=300, bbox_inches='tight')
```

**Refinement checklist:**
- [ ] Colorblind-accessible palette
- [ ] All text legible at publication size
- [ ] Consistent font styles throughout
- [ ] Clear axis labels with units
- [ ] Legend positioned logically
- [ ] Annotations guide attention to key points
- [ ] White space used effectively
- [ ] No chartjunk or unnecessary elements
- [ ] Data-to-ink ratio optimized
- [ ] File format and resolution appropriate

#### Stage 5: Test & Validate (Final Check)

**Before submission, perform these tests:**

1. **Size test**: Print or view figure at actual publication size
   - Can you read all text?
   - Are differences visible?
   - Does it maintain impact when small?

2. **Grayscale test**: Convert to grayscale
   - Are different elements still distinguishable?
   - Some journals still print in B&W

3. **Colorblind test**: Use simulator (Color Oracle, Coblis)
   - Check all common types of colorblindness
   - Ensure information isn't lost

4. **Context test**: Place in manuscript
   - Does it fit the narrative?
   - Does caption + figure stand alone?
   - Is it referenced clearly in text?

5. **Fresh eyes test**: Come back after 24 hours
   - What's your first impression?
   - What would you change now?

### Managing Multiple Figures: Version Control

**For complex projects with many figures:**

```
project/
├── figures/
│   ├── drafts/
│   │   ├── fig1_v1_2024-01-15.png
│   │   ├── fig1_v2_2024-01-18.png
│   │   └── fig1_v3_2024-01-22.png
│   ├── final/
│   │   ├── figure1.pdf
│   │   └── figure1.png
│   └── source_code/
│       ├── generate_fig1.py
│       └── generate_fig1.R
├── data/
│   └── processed_data_for_figs.csv
└── README.md  # Documents what each figure shows
```

**Git for figure code:**

```
# Track changes to figure generation scripts
git add figures/source_code/generate_fig1.py
git commit -m "Added annotation highlighting critical period in Fig 1"

# Tag final versions
git tag -a v1.0-manuscript-submission -m "Figures for initial submission"
```

### When to Stop Iterating

**Beware of endless perfectionism.** Stop when:
- The message is clear to naive viewers
- All journal requirements are met
- External reviewers have no major concerns
- You've addressed accessibility requirements
- Further changes are purely aesthetic preference

**Remember:** A good figure submitted is better than a perfect figure that delays your publication for weeks.

### Exercise 0.3.1
*Take one of your current figures through the full iteration cycle:*

1. **Sketch**: Draw 3 alternative layouts by hand
2. **Draft**: Code the most promising version
3. **Review**: Show to 2 colleagues and note their feedback
4. **Refine**: Implement at least 3 improvements based on feedback
5. **Test**: Run all 5 validation tests
6. **Document**: Write a brief paragraph about what changed and why

---

## 0.4 Tools & Technologies

### Philosophy: Choose the Right Tool for the Task

There is no single "best" tool for creating scientific figures. Your choice depends on:
- Your programming proficiency
- Reproducibility requirements
- Complexity of the visualization
- Collaboration needs
- Time constraints
- Journal requirements

### The Tool Spectrum: Code vs. GUI

**Code-based approaches** (Python, R):
- **Pros**: Reproducible, automatable, version-controlled, handles large datasets, scriptable
- **Cons**: Steeper learning curve, slower for quick edits, limited interactive exploration

**GUI-based approaches** (Illustrator, GraphPad):
- **Pros**: Immediate visual feedback, intuitive, fast for polish and arrangement
- **Cons**: Not reproducible, manual process, difficult to update with new data, expensive software

**Hybrid approach** (Recommended):
1. Generate core visualization with code (data → plot)
2. Export to vector format (PDF, SVG)
3. Final polish in vector editor (labels, arrangement, annotations)
4. Keep both code and final file for reproducibility

---

### Python Ecosystem

**Core Libraries:**

#### 1. Matplotlib – The Foundation
```python
import matplotlib.pyplot as plt

# Basic but flexible
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
```

**When to use:**
- Maximum control over every element
- Custom or unusual plot types
- Publication-quality static figures
- When you need precise positioning

**Limitations:**
- Verbose syntax
- Default aesthetics need work
- Steeper learning curve

#### 2. Seaborn – Statistical Visualization
```python
import seaborn as sns

# High-level interface with good defaults
sns.set_style("whitegrid")
sns.boxplot(data=df, x='group', y='value', palette='Set2')
```

**When to use:**
- Statistical plots (distributions, relationships)
- Quick exploratory analysis
- Better default aesthetics than matplotlib
- Integrates well with pandas dataframes

**Limitations:**
- Less control than matplotlib
- Limited to statistical plot types
- Customization requires matplotlib knowledge

#### 3. Plotly – Interactive Figures
```python
import plotly.express as px

# Interactive plots with hover information
fig = px.scatter(df, x='var1', y='var2', color='group',
                 hover_data=['sample_id'])
fig.show()
```

**When to use:**
- Web-based presentations
- Exploratory data analysis
- Interactive dashboards
- Supplementary online materials

**Limitations:**
- File sizes can be large
- Not ideal for print journals
- Requires JavaScript for full functionality

#### 4. Specialized Libraries
- **NetworkX** + **matplotlib**: Network graphs
- **Biopython**: Phylogenetic trees, sequence alignments
- **Cartopy**: Geographic/map visualizations
- **Yellowbrick**: Machine learning visualization

**Example: Complete Python Workflow**

```python
# generate_figure1.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'Arial',
    'axes.linewidth': 1.0,
    'figure.dpi': 300
})

# Load data
data = pd.read_csv('../data/experiment_data.csv')

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

# Panel A: Time series
axes[0].plot(data['time'], data['control'],
             label='Control', color='#333333', linewidth=2)
axes[0].plot(data['time'], data['treatment'],
             label='Treatment', color='#E63946', linewidth=2)
axes[0].set_xlabel('Time (hours)')
axes[0].set_ylabel('Cell viability (%)')
axes[0].legend(frameon=False)
axes[0].text(-0.15, 1.05, 'A', transform=axes[0].transAxes,
             fontsize=12, fontweight='bold')

# Panel B: Bar chart with stats
sns.barplot(data=data, x='condition', y='final_viability',
            ax=axes[1], palette=['#333333', '#E63946'])
axes[1].set_xlabel('Condition')
axes[1].set_ylabel('Final viability (%)')
axes[1].text(-0.15, 1.05, 'B', transform=axes[1].transAxes,
             fontsize=12, fontweight='bold')

# Save
plt.tight_layout()
plt.savefig('../figures/figure1.pdf', bbox_inches='tight')
plt.savefig('../figures/figure1.png', dpi=300, bbox_inches='tight')
print("Figure saved successfully")
```

---

### R Ecosystem

**Core Libraries:**

#### 1. ggplot2 – Grammar of Graphics

```r
library(ggplot2)

# Layered approach
ggplot(data, aes(x = time, y = value, color = group)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  theme_minimal() +
  labs(x = "Time (hours)", y = "Response")
```

**When to use:**
- Elegant, consistent syntax
- Excellent for publication figures
- Strong statistical integration
- Extensive ecosystem (ggplot extensions)

**Strengths:**
- More intuitive than matplotlib for many
- Better default aesthetics
- Consistent API across plot types
- Huge community and resources

#### 2. Cowplot & Patchwork – Multi-panel Layouts

```r
library(cowplot)
library(patchwork)

# Combine multiple plots
p1 <- ggplot(...) + ...
p2 <- ggplot(...) + ...

# Using patchwork
combined <- p1 + p2 + plot_annotation(tag_levels = 'A')

# Or cowplot
plot_grid(p1, p2, labels = c('A', 'B'), ncol = 2)
```

**When to use:**
- Complex multi-panel figures
- Combining different plot types
- Adding annotations across panels
- Publication-ready layouts

#### 3. Specialized R Packages
- **pheatmap**, **ComplexHeatmap**: Heatmaps with clustering
- **ggtree**: Phylogenetic trees
- **ggmap**: Geographic visualization
- **survminer**: Survival curves
- **ggpubr**: Publication-ready plots with stats

**Example: Complete R Workflow**

```r
# generate_figure1.R
library(tidyverse)
library(cowplot)
library(scales)

# Set theme
theme_set(theme_classic(base_size = 10, base_family = "Arial"))

# Load data
data <- read_csv("../data/experiment_data.csv")

# Panel A: Time series
p1 <- ggplot(data, aes(x = time, y = viability, color = condition)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  scale_color_manual(values = c("Control" = "#333333",
                                  "Treatment" = "#E63946")) +
  labs(x = "Time (hours)",
       y = "Cell viability (%)",
       color = NULL) +
  theme(legend.position = c(0.8, 0.9),
        legend.background = element_blank())

# Panel B: Final values with stats
p2 <- ggplot(data %>% filter(time == max(time)),
             aes(x = condition, y = viability, fill = condition)) +
  geom_bar(stat = "summary", fun = "mean", width = 0.6) +
  geom_errorbar(stat = "summary", fun.data = "mean_se", width = 0.2) +
  scale_fill_manual(values = c("Control" = "#333333",
                                "Treatment" = "#E63946")) +
  labs(x = "Condition", y = "Final viability (%)") +
  theme(legend.position = "none")

# Combine panels
fig1 <- plot_grid(p1, p2, labels = c('A', 'B'),
                  ncol = 2, rel_widths = c(1.2, 1))

# Save
ggsave("../figures/figure1.pdf", fig1, width = 7, height = 3, units = "in")
ggsave("../figures/figure1.png", fig1, width = 7, height = 3, units = "in", dpi = 300)

message("Figure saved successfully")
```

---

### Vector Graphics Editors

**When code alone isn't enough, use professional editing software:**

#### Adobe Illustrator (Industry Standard)
**Pros:**
- Precise control over every element
- Professional typography tools
- Excellent for multi-panel assembly
- Widely used in scientific publishing

**Cons:**
- Expensive (20-50/month)
- Steeper learning curve
- Proprietary format
- Overkill for simple edits

**When to use:**
- Complex multi-panel figures requiring careful alignment
- Adding custom illustrations or diagrams
- Fine-tuning typography and spacing
- Preparing figures for high-impact journals

#### Inkscape (Free & Open-Source)
**Pros:**
- Free and open-source
- Cross-platform (Windows, Mac, Linux)
- SVG native format
- Most Illustrator features available

**Cons:**
- Slightly less polished interface
- Some advanced features missing
- Smaller user community
- Occasional compatibility issues with complex PDFs

**When to use:**
- Budget constraints
- Open-source workflow preference
- Basic to intermediate vector editing
- Learning before investing in Illustrator

#### Affinity Designer (One-Time Purchase)
**Pros:**
- One-time purchase (~70)
- Modern interface
- Good Illustrator compatibility
- Growing in scientific community

**Cons:**
- Smaller community than Illustrator
- Fewer tutorials and resources
- Some niche features missing

**Best Practices for Vector Editing:**

1. **Always start from code**: Generate the data visualization programmatically
2. **Export as vector**: PDF or SVG (never PNG/JPEG for initial export)
3. **Preserve layers**: Keep editable elements on separate layers
4. **Use guides**: Align panels precisely using grid and guides
5. **Outline fonts**: Convert text to paths before final export (prevents font issues)
6. **Save source files**: Keep both .ai/.svg and final PDF

**Example Workflow: Code → Vector Editor**
1. Python/R generates figure1_raw.pdf
2. Open in Illustrator/Inkscape
3. Edits:
   - Adjust panel spacing
   - Add panel labels (A, B, C)
   - Fine-tune legend position
   - Add connecting arrows or annotations
   - Ensure consistent font sizes
4. Save as figure1_working.ai (editable)
5. Export as figure1_final.pdf (for submission)
6. Keep both files in version control

---

### The Reproducibility vs. Polish Trade-Off

**Fully Code-Based (Maximum Reproducibility)**

```
data → script.py → figure.pdf
```
- Updates automatically if data changes
- Fully documented in code
- Easy to reproduce
- Limited aesthetic control

**Hybrid Approach (Recommended Balance)**

```
data → script.py → figure_raw.pdf → [illustrator] → figure_final.pdf
```
- Core visualization reproducible
- Manual polish for publication quality
- Document manual steps in README
- Keep both versions

**Fully Manual (Avoid If Possible)**

```
data → Excel → [copy to Illustrator] → figure.pdf
```
- Not reproducible
- Error-prone
- Hard to update
- Common in some fields but discouraged

### Tool Recommendations by Career Stage

**Graduate Students / Early Career:**
- **Learn**: Python (matplotlib + seaborn) OR R (ggplot2)
- **Practice with**: Free tools (Inkscape)
- **Invest time in**: Understanding principles, not just tools

**Postdocs / Mid-Career:**
- **Master**: One ecosystem deeply (Python or R)
- **Add**: Vector editor (Illustrator or Affinity)
- **Consider**: Specialized tools for your field

**Established Researchers / Lab Heads:**
- **Standardize**: Lab-wide tools and templates
- **Delegate**: Train students/postdocs
- **Invest in**: Site licenses for software
- **Focus on**: Consistency and reproducibility across lab

### Exercise 0.4.1

**Tool Assessment & Setup:**

1. **Evaluate your current tools:**
   - What do you currently use for figures?
   - What frustrates you about your current workflow?
   - What do you wish you could do but can't?

2. **Choose your primary platform:**
   - Based on this chapter, pick: Python ecosystem, R ecosystem, or mixed
   - Write 2-3 sentences justifying your choice

3. **Set up your environment:**
   - Install your chosen tools
   - Create a template script/project structure
   - Generate one simple figure (scatter plot with your data)

4. **Test the workflow:**
   - Create a figure with code
   - Export to vector format
   - Make one manual edit in a vector editor
   - Document the process in a README file

## Chapter 0 Summary

### Key Takeaways

1. **Figures are primary communication vehicles** – they're not decorative but essential to scientific storytelling

2. **Define before you design** – clarify message, audience, and context before touching software

3. **Iterate deliberately** – sketch → draft → review → refine → test

4. **Choose tools strategically** – balance reproducibility, control, and efficiency; hybrid approaches often work best

5. **Establish workflow early** – consistent processes save time and improve quality across projects

### Before Moving to Chapter 1

You should now be able to:
- [ ] Articulate why a figure matters in scientific communication
- [ ] Define the message, audience, and context for any figure you create
- [ ] Execute the five-stage iterative design process
- [ ] Choose appropriate tools for your needs and skill level
- [ ] Set up a reproducible workflow for figure generation

### Looking Ahead

In **Chapter 1**, we'll dive into the foundational science of visual perception – understanding *how* and *why* humans process visual information. This will provide the cognitive framework for all subsequent design decisions.

Before continuing, ensure you've completed at least Exercise 0.3.1 (iterative refinement of one of your figures) and Exercise 0.4.1 (tool setup and workflow test).

