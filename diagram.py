import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(22, 5))
ax.set_xlim(-0.5, 11.5)
ax.set_ylim(-1, 2)
ax.axis('off')
fig.patch.set_facecolor('white')

nodes = [
    ('Input\n3 × 112 × 112',                                '#F2F2F2'),
    ('Input Layer\nConv3×3 → BN → PReLU\n64 × 112 × 112',   '#F8CECC'),
    ('Stage 1\n3× BasicBlockIR\n64 × 56 × 56',              '#DAE8FC'),
    ('Stage 2\n4× BasicBlockIR\n128 × 28 × 28',             '#DAE8FC'),
    ('Stage 3\n14× BasicBlockIR\n256 × 14 × 14',            '#DAE8FC'),
    ('Stage 4\n3× BasicBlockIR\n512 × 7 × 7',               '#DAE8FC'),
    ('CoordAtt\n512 × 7 × 7',                               '#D5E8D4'),
    ('Output Layer\nBN → Dropout(0.4)\nFlatten → Linear\n(25088→512) → BN1d',  '#F8CECC'),
    ('Embedding\n512-dim',                                   '#FFE6CC'),
    ('AdaFace / ArcFace\nMargin Layer\nN classes',           '#E1D5E7'),
    ('CrossEntropy\nLoss',                                   '#BAC8D3'),
]

edge_labels = [
    '', '', '', '', '', 'attention', '', '', '512-dim', 'logits'
]

box_w = 0.85
box_h = 1.2

for i, (label, color) in enumerate(nodes):
    x = i
    y = 0.5

    is_ellipse = (i == len(nodes) - 1)

    if is_ellipse:
        ellipse = mpatches.FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor='#666666', linewidth=1.2,
            mutation_scale=0.3
        )
        ax.add_patch(ellipse)
    else:
        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor='#666666', linewidth=1.2
        )
        ax.add_patch(rect)

    ax.text(x, y, label, ha='center', va='center',
            fontsize=7, fontfamily='sans-serif', fontweight='bold',
            color='#333333', linespacing=1.3)

for i in range(len(nodes) - 1):
    x_start = i + box_w / 2
    x_end = (i + 1) - box_w / 2
    y = 0.5

    ax.annotate('', xy=(x_end, y), xytext=(x_start, y),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

    if edge_labels[i]:
        ax.text((x_start + x_end) / 2, y + 0.18, edge_labels[i],
                ha='center', va='bottom', fontsize=6, fontstyle='italic',
                color='#888888')

ax.set_title('IR-50 + CoordAtt  —  Architecture Diagram', fontsize=14,
             fontweight='bold', pad=15, color='#222222')

plt.tight_layout()
plt.savefig('IR50CoordAtt.png', dpi=200, bbox_inches='tight', facecolor='white')
print('Diagram saved to: IR50CoordAtt.png')
plt.show()