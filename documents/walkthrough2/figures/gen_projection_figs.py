"""
Generate three figures for the Planar projection section:
  - proj_globe.pdf      : globe + tangent plane intuition
  - proj_distortion.pdf : meridian convergence & cos(phi) correction
  - proj_slc_box.pdf    : SLC bounding box, lat/lon -> meters mapping
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

import os
out = os.path.dirname(os.path.abspath(__file__))

# Larger default fonts so figures stay legible after being scaled down
# to a single column in the two-column layout.
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
})

PHI0 = 40.75
LAM0 = -111.90
R = 6_371_000.0


# ============================================================
# Figure 1: meridian convergence (why we need the cos(phi) factor)
# ============================================================
fig, ax = plt.subplots(figsize=(7.0, 4.5))

lats = np.array([0, 15, 30, 45, 60, 75])
xs_eq = np.linspace(-3, 3, 7)
ys = lats

# Draw a wireframe of the sphere as projected onto a "flat" lat-lon plane,
# but show how the spacing of meridians converges with latitude.
for lat in lats:
    width = np.cos(np.deg2rad(lat))
    for x in xs_eq:
        ax.plot([x * width, x * width], [lat - 1.4, lat + 1.4], color='#bbbbbb', lw=0.5)
    # parallel
    ax.plot([-3, 3], [lat, lat], color='#888888', lw=0.6)
    # one degree of longitude at this latitude (in km)
    km = np.cos(np.deg2rad(lat)) * 111.32
    ax.text(3.25, lat, rf'$1^\circ$ lon $\approx$ {km:.1f} km',
            fontsize=13, va='center')

# Highlight latitude 40.75 (SLC)
ax.axhline(40.75, color='#d62728', lw=1.6, alpha=0.9)
ax.text(-5.6, 40.75, 'SLC\n$\\varphi_0=40.75^\\circ$', color='#d62728',
        fontsize=13, ha='left', va='center', fontweight='bold')

ax.set_xlim(-6.0, 5.5)
ax.set_ylim(-3, 80)
ax.set_xlabel('relative longitude (degrees)')
ax.set_ylabel('latitude (degrees)')
ax.grid(False)
for spine in ('top', 'right'):
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(out, 'proj_distortion.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(out, 'proj_distortion.png'), dpi=160, bbox_inches='tight')
plt.close()


# ============================================================
# Figure 2: tangent plane at (phi0, lam0)
# ============================================================
fig, ax = plt.subplots(figsize=(7.0, 4.6))

# Draw the curved earth as a circle (cross section), with a tangent plane
theta = np.linspace(np.pi/3, 2*np.pi/3, 200)
ex = np.cos(theta)
ey = np.sin(theta)
ax.plot(ex, ey, color='#1f77b4', lw=1.4, alpha=0.55, label='Earth surface')

# Tangent point
t0 = np.pi/2
px = np.cos(t0)
py = np.sin(t0)

# Tangent plane (a horizontal line at y=1)
ax.plot([-0.85, 0.85], [1.0, 1.0], color='#2ca02c', lw=2.0, label='tangent plane')
ax.plot([-0.85, -0.85], [0.99, 1.01], color='#2ca02c', lw=1.5)
ax.plot([ 0.85,  0.85], [0.99, 1.01], color='#2ca02c', lw=1.5)

# A point on Earth a bit east of the tangent point
t_pt = np.pi/2 - 0.42
qx = np.cos(t_pt)
qy = np.sin(t_pt)

# Highlight the great-circle arc segment from tangent point to Earth point.
seg_t = np.linspace(t0, t_pt, 80)
seg_x = np.cos(seg_t)
seg_y = np.sin(seg_t)
ax.plot(seg_x, seg_y, color='#1f77b4', lw=3.4, solid_capstyle='round',
        zorder=3, label='great-circle distance')

# Endpoints, drawn after the highlighted segment so they sit on top.
ax.plot([qx], [qy], 'o', color='#7f7f7f', markersize=7, zorder=5)
ax.plot([px], [py], 'o', color='#d62728', markersize=8, zorder=5)
ax.annotate(r'tangent point  $(\varphi_0, \lambda_0)$',
            xy=(px, py), xytext=(0.6, 1.18),
            arrowprops=dict(arrowstyle='->', color='#d62728'),
            color='#d62728', fontsize=14)

# Project that point straight up onto the tangent plane
ax.plot([qx, qx], [qy, 1.0], '--', color='#7f7f7f', lw=0.9)
ax.plot([qx], [1.0], 'o', color='#9467bd', markersize=6, zorder=5)

# Label the highlighted arc.
mid_t = (t0 + t_pt) / 2.0
arc_mx = np.cos(mid_t)
arc_my = np.sin(mid_t)
ax.annotate('great-circle\ndistance on Earth',
            xy=(qx, qy),
            xytext=(arc_mx + 0.27, arc_my - 0.30),
            arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.2),
            color='#1f77b4', fontsize=13, ha='left')
ax.annotate(r'projected $(x, y)$ in meters',
            xy=(qx, 1.0), xytext=(-0.3, 1.32),
            arrowprops=dict(arrowstyle='->', color='#9467bd'),
            color='#9467bd', fontsize=13, ha='center')

ax.text(-0.92, 0.62, 'curved Earth',
        color='#1f77b4', fontsize=14, fontweight='bold')
ax.text(-0.85, 1.07, 'flat working plane',
        color='#2ca02c', fontsize=14, fontweight='bold')

ax.set_xlim(-1.05, 1.25)
ax.set_ylim(0.40, 1.45)
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(out, 'proj_globe.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(out, 'proj_globe.png'), dpi=160, bbox_inches='tight')
plt.close()


# ============================================================
# Figure 3: SLC bounding box - lat/lon vs meters
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(6.0, 9.5))

# Bounding box of SLC network (approximate)
slc_minlat, slc_maxlat = 40.55, 40.95
slc_minlon, slc_maxlon = -112.20, -111.60

# Sample some (lat, lon) points: corners + center + a few interior points
demo_pts_ll = np.array([
    (slc_minlat, slc_minlon),
    (slc_maxlat, slc_minlon),
    (slc_maxlat, slc_maxlon),
    (slc_minlat, slc_maxlon),
    (40.75, -111.90),       # center
    (40.65, -112.0),
    (40.85, -111.75),
    (40.78, -111.85),
])

def project(lat, lon):
    x = (lon - LAM0) * np.cos(np.deg2rad(PHI0)) * R * np.pi/180
    y = (lat - PHI0) * R * np.pi/180
    return x, y

xs, ys = project(demo_pts_ll[:, 0], demo_pts_ll[:, 1])

# left: lat/lon
ax = axes[0]
rect = Rectangle((slc_minlon, slc_minlat),
                 slc_maxlon - slc_minlon, slc_maxlat - slc_minlat,
                 fill=False, edgecolor='#1f77b4', lw=1.6)
ax.add_patch(rect)
for i, (lat, lon) in enumerate(demo_pts_ll):
    color = '#d62728' if (lat, lon) == (40.75, -111.90) else '#1f77b4'
    ax.plot(lon, lat, 'o', color=color, markersize=8 if color == '#d62728' else 6)
ax.plot(LAM0, PHI0, 'o', color='#d62728', markersize=10)
ax.annotate(rf'$(\varphi_0, \lambda_0) = ({PHI0:.2f}^\circ, {LAM0:.2f}^\circ)$',
            xy=(LAM0, PHI0), xytext=(LAM0 + 0.06, PHI0 - 0.08),
            color='#d62728', fontsize=12,
            arrowprops=dict(arrowstyle='->', color='#d62728'))
ax.set_xlabel('longitude $\\lambda$ (degrees)')
ax.set_ylabel('latitude $\\varphi$ (degrees)')
ax.set_title('Before: degrees on the globe')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xticks([-112.2, -112.0, -111.8, -111.6])

# right: meters
ax = axes[1]
xs_corners, ys_corners = project(np.array([slc_minlat, slc_maxlat, slc_maxlat, slc_minlat, slc_minlat]),
                                 np.array([slc_minlon, slc_minlon, slc_maxlon, slc_maxlon, slc_minlon]))
ax.plot(xs_corners / 1000, ys_corners / 1000, color='#2ca02c', lw=1.6)
for i, (x, y) in enumerate(zip(xs / 1000, ys / 1000)):
    color = '#d62728' if (demo_pts_ll[i, 0], demo_pts_ll[i, 1]) == (40.75, -111.90) else '#2ca02c'
    ax.plot(x, y, 'o', color=color, markersize=8 if color == '#d62728' else 6)
ax.plot(0, 0, 'o', color='#d62728', markersize=10)
ax.annotate(r'$(0, 0)$  origin = tangent point',
            xy=(0, 0), xytext=(5, -8),
            color='#d62728', fontsize=12,
            arrowprops=dict(arrowstyle='->', color='#d62728'))
ax.set_xlabel('$x$ (kilometers, eastward)')
ax.set_ylabel('$y$ (kilometers, northward)')
ax.set_title('After: meters in the working plane')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out, 'proj_slc_box.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(out, 'proj_slc_box.png'), dpi=160, bbox_inches='tight')
plt.close()

print('Wrote:')
for f in ['proj_distortion', 'proj_globe', 'proj_slc_box']:
    print('  ', os.path.join(out, f + '.pdf'))
