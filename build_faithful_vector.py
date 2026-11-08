import numpy as np
from PIL import Image
from scipy import ndimage

img_path = "/Users/frank/Work/13_money/02_开发/fassett-farm-vinyl/assets/FassettFarmNursery_BW-jpeg.jpg"
gray = np.array(Image.open(img_path).convert("L"))
h, w = gray.shape
ink = gray < 190

struct8 = np.ones((3, 3), dtype=bool)
y3, x3 = np.ogrid[-3:4, -3:4]
disk3 = (x3*x3 + y3*y3) <= 9
y2, x2 = np.ogrid[-2:3, -2:3]
disk2 = (x2*x2 + y2*y2) <= 4
y6, x6 = np.ogrid[-6:7, -6:7]
disk6 = (x6*x6 + y6*y6) <= 36

# Bridge the dotted dome of the middle Echinacea coneflower (y: 1270..1420, x: 580..750)
cone_patch = ink[1270:1420, 580:750]
cone_closed = ndimage.binary_dilation(cone_patch, structure=disk3)
ink[1270:1420, 580:750] = cone_closed

# Protect typography & coneflower cones
protect_mask = np.zeros_like(ink)
protect_mask[:780, :] = True
protect_mask[1930:, 920:1880] = True
protect_mask[1020:1620, 260:780] = True  # All 3 Echinacea flower heads

# Remove floating grass ticks / meadow dashes / sky specks (< 2400 px outside protected regions)
lbl, _ = ndimage.label(ink, structure=struct8)
sz = np.bincount(lbl.ravel())

clean_ink = np.zeros_like(ink)
for i in range(1, len(sz)):
    if sz[i] >= 2400:
        clean_ink[lbl == i] = True
    elif sz[i] >= 80:
        if np.any(protect_mask & (lbl == i)):
            clean_ink[lbl == i] = True

# In the middle coneflower cone & lower-left coneflower cone, remove tiny 1-2px stipple dots inside the dome while keeping the dome arc!
# Wait: let's see what components are inside 1020:1620, 260:780 with size < 350 (tiny stipple dots inside cones)
for i in range(1, len(sz)):
    if sz[i] < 350 and np.any(protect_mask[1020:1620, 260:780] & (lbl[1020:1620, 260:780] == i)):
        clean_ink[lbl == i] = False

# Simplify dense internal webbing in Yarrow florets & Tree crown by merging small white cells into clean open shapes
white = ~clean_ink
lbl_w, _ = ndimage.label(white, structure=struct8)
sz_w = np.bincount(lbl_w.ravel())

dense_region = np.zeros_like(ink)
dense_region[860:1520, 1620:2650] = True  # Yarrow heads + fern leaves
dense_region[850:1320, 1020:1460] = True  # Tree canopy

med_w = (sz_w >= 140) & (sz_w < 1800)
med_w[0] = False
med_w_mask = med_w[lbl_w] & dense_region

outer_bg = (lbl_w == 1)
outer_border_guard = ndimage.binary_dilation(outer_bg, structure=np.ones((15, 15), dtype=bool))

carve_white = ndimage.binary_dilation(med_w_mask, structure=disk3) & (~outer_border_guard)
simplified_ink = clean_ink & (~carve_white)

# Thicken all strokes uniformly so minimum vinyl path width is bold and plotter-safe
thick_ink = ndimage.binary_dilation(simplified_ink, structure=disk2)

# Clean isolated crumbs (< 850 px outside typography)
lbl3, _ = ndimage.label(thick_ink, structure=struct8)
sz3 = np.bincount(lbl3.ravel())
keep3 = (sz3 >= 850) | ((sz3 >= 350) & np.isin(np.arange(len(sz3)), np.unique(lbl3[protect_mask])))
keep3[0] = False
thick_ink = keep3[lbl3]

# Fill tiny remaining pinholes (< 350 px)
w3 = ~thick_ink
lbl_w3, _ = ndimage.label(w3, structure=struct8)
sz_w3 = np.bincount(lbl_w3.ravel())
fill_w3 = (sz_w3 > 0) & (sz_w3 < 350)
fill_w3[0] = False
thick_ink = thick_ink | fill_w3[lbl_w3]

# Gaussian smooth for crisp vector curves
smooth = ndimage.gaussian_filter(thick_ink.astype(np.float32), sigma=2.0) > 0.45

out_bw = np.where(smooth, 0, 255).astype(np.uint8)
Image.fromarray(out_bw).save("/tmp/fassett_faithful_v3.png")
print("Saved /tmp/fassett_faithful_v3.png")
