import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Node definitions ──
# (label, background colour)
# Use '\n' to separate lines — they will be rendered rotated 90°
nodes = [
    ('Input\n3 × 112 × 112',                                        '#F2F2F2'),
    ('Input Layer\nConv3×3 → BN → PReLU\n64 × 112 × 112',          '#F8CECC'),
    ('Stage 1\n3× BasicBlockIR\n64 × 56 × 56',                      '#DAE8FC'),
    ('Stage 2\n4× BasicBlockIR\n128 × 28 × 28',                     '#DAE8FC'),
    ('Stage 3\n14× BasicBlockIR\n256 × 14 × 14',                    '#DAE8FC'),
    ('Stage 4\n3× BasicBlockIR\n512 × 7 × 7',                       '#DAE8FC'),
    ('CoordAtt\n512 × 7 × 7',                                       '#D5E8D4'),
    ('Output Layer\nBN → Dropout(0.4)\nFlatten → Linear\n(25088→512)\n→ BN1d',  '#F8CECC'),
    ('Embedding\n512-dim',                                           '#FFE6CC'),
    ('AdaFace / ArcFace\nMargin Layer\nN classes',                   '#E1D5E7'),
    ('CrossEntropy\nLoss',                                           '#BAC8D3'),
]

edge_labels = [
    '', '', '', '', '', 'attention', '', '', '512-dim', 'logits'
]

# Skip connection indices (which nodes have internal residual skip)
skip_indices = [2, 3, 4, 5]  # Stage 1, 2, 3, 4

# ── Layout parameters ──
n       = len(nodes)
box_w   = 1.2          # width of vertical box
box_h   = 5.5          # height of vertical box (tall enough for rotated text)
gap     = 2.0          # gap between boxes → longer arrows
step    = box_w + gap  # centre-to-centre distance

fig_w   = step * n + 3
fig_h   = box_h + 6.0  # extra room for skip arcs below + title above

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(-2, step * n + 1)
ax.set_ylim(-4.5, box_h + 3)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Helper ──
def cx_of(i):
    return i * step + box_w / 2

cy = box_h / 2  # vertical centre for all boxes

# ── Draw boxes and rotated text ──
for i, (label, color) in enumerate(nodes):
    cx = cx_of(i)

    rect = mpatches.FancyBboxPatch(
        (cx - box_w / 2, cy - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor='#444444', linewidth=1.5
    )
    ax.add_patch(rect)

    # Text rotated 90° (reads bottom → top)
    ax.text(cx, cy, label,
            ha='center', va='center',
            fontsize=8, fontfamily='sans-serif', fontweight='bold',
            color='#222222', linespacing=1.5,
            rotation=90)

# ── Draw main flow arrows ──
for i in range(n - 1):
    x_start = cx_of(i) + box_w / 2       # right edge of box i
    x_end   = cx_of(i + 1) - box_w / 2   # left edge of box i+1

    ax.annotate(
        '', xy=(x_end, cy), xytext=(x_start, cy),
        arrowprops=dict(arrowstyle='->', color='#444444', lw=2.0,
                        shrinkA=4, shrinkB=4)
    )

    # Edge label above arrow
    if edge_labels[i]:
        ax.text((x_start + x_end) / 2, cy + 0.40, edge_labels[i],
                ha='center', va='bottom', fontsize=7.5, fontstyle='italic',
                color='#666666')

# ── Draw skip connections (curved arcs below each Stage box) ──
for idx in skip_indices:
    cx = cx_of(idx)
    y_bot = cy - box_h / 2  # bottom edge of the box

    # Start & end points on the bottom of the box, inset slightly
    x_left  = cx - box_w / 2 + 0.05
    x_right = cx + box_w / 2 - 0.05

    # Draw a smooth arc below the box
    arc_depth = 2.0  # how far below the box the arc goes
    t = np.linspace(0, 1, 60)
    arc_x = x_left + (x_right - x_left) * t
    arc_y = y_bot - arc_depth * np.sin(np.pi * t)

    ax.plot(arc_x, arc_y, color='#D94040', lw=2.0, linestyle='-', zorder=3)

    # Arrowhead at the right end
    ax.annotate(
        '', xy=(x_right, y_bot), xytext=(arc_x[-3], arc_y[-3]),
        arrowprops=dict(arrowstyle='->', color='#D94040', lw=2.0,
                        shrinkA=0, shrinkB=0),
        zorder=4
    )

    # "skip" label at the bottom of the arc
    ax.text(cx, y_bot - arc_depth - 0.25, 'skip connection',
            ha='center', va='top', fontsize=7, fontstyle='italic',
            color='#D94040', fontweight='bold')

# ── Title ──
ax.set_title('IR-50 + CoordAtt  —  Architecture Diagram',
             fontsize=18, fontweight='bold', pad=25, color='#222222')

plt.tight_layout()
plt.savefig('IR50CoordAtt.png', dpi=200, bbox_inches='tight', facecolor='white')
print('Diagram saved to: IR50CoordAtt.png')
plt.show()