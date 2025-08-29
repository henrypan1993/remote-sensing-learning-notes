# Integrated script to reproduce the target image (two panels with FOV & grid)
import matplotlib.pyplot as plt
import numpy as np

# --- Helper ---
def draw_ground_grid(ax_obj, x_min, x_max, y, step=1, height=0.5):
    for gx in np.arange(x_min, x_max + step, step):
        ax_obj.plot([gx, gx], [y, y + height], linestyle=':', color='gray', alpha=0.5)

# --- Figure ---
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Common params
platform_y = 5.0
pixel_height = 0.6

# ------------------------------
# Left: Actual Ground Coverage with FOV & Grid
# ------------------------------
# Sensor platform
ax[0].plot([0], [platform_y], 'ko', label='Sensor platform')

# FOV ray lines (dashed)
ax[0].plot([0, -0.5], [platform_y, 0], '--', color='k', alpha=0.7)
ax[0].plot([0, 0.5], [platform_y, 0], '--', color='k', alpha=0.7)
ax[0].plot([0, 2.2], [platform_y, 0], '--', color='k', alpha=0.7)
ax[0].plot([0, 4.2], [platform_y, 0], '--', color='k', alpha=0.7)

# Ground grid
draw_ground_grid(ax[0], -2, 6, 0, step=1, height=pixel_height)

# Ground pixels: center vs edge
ax[0].add_patch(plt.Rectangle((-0.5, 0), 1.0, pixel_height, color='skyblue', alpha=0.8, label='Center pixel'))
ax[0].add_patch(plt.Rectangle((2.2, 0), 2.0, pixel_height, color='orange', alpha=0.8, label='Edge pixel'))

# Layout
ax[0].set_title("Actual Ground Coverage with FOV & Grid")
ax[0].set_xlim(-2, 6)
ax[0].set_ylim(-0.8, platform_y + 0.7)
ax[0].set_aspect('equal', adjustable='box')
ax[0].axis('off')
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 0.88), frameon=True)

# ------------------------------
# Right: Projected to Uniform Scale & Grid
# ------------------------------
draw_ground_grid(ax[1], -2, 3, 0, step=1, height=pixel_height)

ax[1].add_patch(plt.Rectangle((-0.5, 0), 1.0, pixel_height, color='skyblue', alpha=0.8, label='Center pixel'))
ax[1].add_patch(plt.Rectangle((0.7, 0), 1.0, pixel_height, color='orange', alpha=0.8, label='Edge pixel (compressed)'))

ax[1].set_title("Projected to Uniform Scale & Grid")
ax[1].set_xlim(-2, 3)
ax[1].set_ylim(-0.8, platform_y + 0.7)
ax[1].set_aspect('equal', adjustable='box')
ax[1].axis('off')
ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 0.88), frameon=True)

plt.tight_layout()

# Save to disk
out_path = "/mnt/data/fov_projection.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
out_path
