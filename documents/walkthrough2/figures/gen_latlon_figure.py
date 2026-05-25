"""
Generate a figure that explains latitude and longitude.
Earth is drawn as a slightly oblate ellipse, with one latitude
parallel and one longitude meridian highlighted.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D

out = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
})

fig, ax = plt.subplots(figsize=(7.6, 6.0))

a = 1.04   # equatorial radius
b = 1.00   # polar radius

# ---- Earth as a filled ellipse with a soft fill -------------------
earth = Ellipse((0, 0), width=2 * a, height=2 * b,
                facecolor='#dbe8f4', edgecolor='#1f4e79', lw=2, zorder=1)
ax.add_patch(earth)

# ---- Faint background parallels (latitude lines) ------------------
for lat in [-60, -45, -30, -15, 15, 30, 45, 60]:
    phi = np.deg2rad(lat)
    y = b * np.sin(phi)
    half_w = a * np.cos(phi)
    ax.plot([-half_w, half_w], [y, y],
            color='#7898b8', lw=0.6, alpha=0.55, zorder=2)

# Equator (slightly stronger)
ax.plot([-a, a], [0, 0], color='#1f4e79', lw=1.0, alpha=0.85, zorder=2)
ax.text(a + 0.04, 0, 'equator', color='#1f4e79', fontsize=12,
        va='center', ha='left', fontweight='bold')

# ---- Faint background meridians (longitude lines) -----------------
def meridian_xy(lon_deg, n=200):
    """Project a meridian (constant longitude) onto the 2D ellipse face.
    For visualization only: orthographic projection of a line of constant
    longitude on a sphere onto the page.
    """
    lon = np.deg2rad(lon_deg)
    phi = np.linspace(-np.pi / 2, np.pi / 2, n)
    x = a * np.cos(phi) * np.sin(lon)
    y = b * np.sin(phi)
    return x, y

for lon in [-60, -30, 30, 60]:
    mx, my = meridian_xy(lon)
    ax.plot(mx, my, color='#7898b8', lw=0.6, alpha=0.55, zorder=2)

# Prime meridian (slightly stronger)
mx, my = meridian_xy(0)
ax.plot(mx, my, color='#1f4e79', lw=1.0, alpha=0.85, zorder=2)

# ---- Highlight ONE latitude line ---------------------------------
LAT_HI = 40
phi_hi = np.deg2rad(LAT_HI)
y_hi = b * np.sin(phi_hi)
half_w = a * np.cos(phi_hi)
ax.plot([-half_w, half_w], [y_hi, y_hi],
        color='#d62728', lw=2.6, zorder=4)

# Label the highlighted latitude
ax.annotate(rf'latitude $\varphi = {LAT_HI}^\circ$',
            xy=(half_w * 0.85, y_hi),
            xytext=(1.55, y_hi + 0.55),
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2),
            color='#d62728', fontsize=14, ha='left')

# ---- Highlight ONE longitude line --------------------------------
LON_HI = 60
mx_hi, my_hi = meridian_xy(LON_HI)
ax.plot(mx_hi, my_hi, color='#2ca02c', lw=2.6, zorder=4)

# Label the highlighted longitude
ax.annotate(rf'longitude $\lambda = {LON_HI}^\circ$',
            xy=(mx_hi[len(mx_hi) // 3], my_hi[len(my_hi) // 3]),
            xytext=(-2.55, 0.85),
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.2),
            color='#2ca02c', fontsize=14, ha='left')

# ---- Pole markers and small reference axis -----------------------
ax.plot([0], [b], 'o', color='#1f4e79', markersize=6, zorder=5)
ax.plot([0], [-b], 'o', color='#1f4e79', markersize=6, zorder=5)
ax.text(0.06, b + 0.10, 'north pole', color='#1f4e79', fontsize=12,
        ha='left', fontweight='bold')
ax.text(0.06, -b - 0.16, 'south pole', color='#1f4e79', fontsize=12,
        ha='left', fontweight='bold')

# ---- A point at (LAT_HI, LON_HI) intersection --------------------
phi = np.deg2rad(LAT_HI)
lon = np.deg2rad(LON_HI)
px = a * np.cos(phi) * np.sin(lon)
py = b * np.sin(phi)
ax.plot([px], [py], 'o', color='black', markersize=8, zorder=6)
ax.annotate(rf'a point at $(\varphi, \lambda) = ({LAT_HI}^\circ,\, {LON_HI}^\circ)$',
            xy=(px, py),
            xytext=(px + 0.55, py - 0.55),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
            color='black', fontsize=15, ha='left')

# ---- Cosmetic --------------------------------------------------
ax.set_xlim(-2.8, 2.9)
ax.set_ylim(-1.65, 1.65)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Latitude and longitude on the Earth',
             fontsize=15, pad=18)

plt.tight_layout()
plt.savefig(os.path.join(out, 'latlon_intro.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(out, 'latlon_intro.png'), dpi=160, bbox_inches='tight')
plt.close()
print('Wrote', os.path.join(out, 'latlon_intro.pdf'))
