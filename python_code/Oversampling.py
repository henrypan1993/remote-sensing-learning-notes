import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

pixel_height = 0.8
ifov_width = 1.0
gap = 0.1

# --- Left: Proper sampling (no overlap, total width normal) ---
x_start = 0
for i in range(5):
    ax[0].add_patch(plt.Rectangle((x_start, 0), ifov_width, pixel_height,
                                  facecolor='skyblue', edgecolor='k', alpha=0.7))
    x_start += ifov_width + gap

ax[0].set_title("Proper Sampling\n(Total Width Normal)")
ax[0].set_xlim(-0.5, 7)
ax[0].set_ylim(-0.5, 1.5)
ax[0].set_aspect('equal', adjustable='box')
ax[0].axis('off')

# --- Right: Oversampling (more pixels, total width larger) ---
x_start = 0
overlap_shift = 0.5  # less than IFOV width, causing overlap
for i in range(12):  # more pixels, so total width expands
    ax[1].add_patch(plt.Rectangle((x_start, 0), ifov_width, pixel_height,
                                  facecolor='orange', edgecolor='k', alpha=0.7))
    x_start += overlap_shift

ax[1].set_title("Oversampling\n(Total Width Too Broad)")
ax[1].set_xlim(-0.5, 9)  # wider axis to show increase
ax[1].set_ylim(-0.5, 1.5)
ax[1].set_aspect('equal', adjustable='box')
ax[1].axis('off')

plt.tight_layout()
plt.show()
