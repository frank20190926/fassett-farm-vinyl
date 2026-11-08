import numpy as np
from PIL import Image
from scipy import ndimage

img_path = "/Users/frank/Work/13_money/02_开发/fassett-farm-vinyl/assets/FassettFarmNursery_BW-jpeg.jpg"
gray = np.array(Image.open(img_path).convert("L"))
h, w = gray.shape
ink = gray < 185

struct8 = np.ones((3, 3), dtype=bool)
y3, x3 = np.ogrid[-3:4, -3:4]
disk3 = (x3*x3 + y3*y3) <= 9
y2, x2 = np.ogrid[-2:3, -2:3]
disk2 = (x2*x2 + y2*y2) <= 4

# 1. Protect typography regions (top arch y < 780, bottom arch y > 1950)
type_mask = np.zeros_like(ink)
type_mask[:780, :] = True
type_mask[1930:, :] = True
# Protect left stems & leaves that go down below 1930 (x < 920)
type_mask[1930:, :920] = False
type_mask[1930:, 1880:] = False

# 2. Remove all floating black specks/ticks in the illustration (< 2200 px), while keeping all letters (> 300 px) in type_mask
lbl, _ = ndimage.label(ink, structure=struct8)
sz = np.bincount(lbl.ravel())

clean_ink = np.zeros_like(ink)
for i in range(1, len(sz)):
    if sz[i] >= 2400:
        clean_ink[lbl == i] = True
    elif sz[i] >= 350:
        # keep if in typography region
        if np.any(type_mask & (lbl == i)):
            clean_ink[lbl == i] = True

# 3. For tiny enclosed white holes:
# Instead of turning tiny white petals into black blobs in the yarrow & tree canopy,
# what if we OPEN (convert to white) the tiny cramped black dividers inside the yarrow/tree so adjacent tiny white holes merge into larger, clean plotter-safe white petals!
# Wait: how do we detect where tiny white holes are clustered?
white = ~clean_ink
lbl_w, _ = ndimage.label(white, structure=struct8)
sz_w = np.bincount(lbl_w.ravel())

# Tiny white holes (< 650 px) are weeding traps.
# In the yarrow heads (x > 1620, 850 < y < 1520) and tree crown (1020 < x < 1460, 860 < y < 1300),
# if we dilate the white holes slightly to merge micro-petals into bold clean florets, and only fill isolated tiny pinholes (< 180 px):
tiny_w = (sz_w > 0) & (sz_w < 550)
tiny_w[0] = False
tiny_w_mask = tiny_w[lbl_w]

# Region of Yarrow florets & Tree crown where detail is overly dense
dense_region = np.zeros_like(ink)
dense_region[860:1520, 1620:2650] = True  # Yarrow heads + fern leaves
dense_region[850:1320, 1020:1460] = True  # Tree canopy

# In dense_region, dilate medium white spaces (250..1800 px) by 3px so thin internal black dividers vanish and merge into clean bold white florets, while keeping the outer contour!
med_w = (sz_w >= 180) & (sz_w < 1600)
med_w[0] = False
med_w_mask = med_w[lbl_w] & dense_region

# Keep the outer silhouette boundary of the dense regions intact!
# Outer background is label 1 in lbl_w
outer_bg = (lbl_w == 1)
outer_border_guard = ndimage.binary_dilation(outer_bg, structure=np.ones((15, 15), dtype=bool))

carve_white = ndimage.binary_dilation(med_w_mask, structure=disk3) & (~outer_border_guard)
simplified_ink = clean_ink & (~carve_white)

# Now thicken all remaining strokes uniformly by disk2 (radius 2.5px) so every line is bold and cuttable!
thick_ink = ndimage.binary_dilation(simplified_ink, structure=disk2)

# Remove any newly isolated tiny black crumbs (< 900 px outside type_mask)
lbl3, _ = ndimage.label(thick_ink, structure=struct8)
sz3 = np.bincount(lbl3.ravel())
keep3 = (sz3 >= 950) | ((sz3 >= 350) & np.isin(np.arange(len(sz3)), np.unique(lbl3[type_mask])))
keep3[0] = False
thick_ink = keep3[lbl3]

# Fill only truly small remaining pinholes (< 320 px)
w3 = ~thick_ink
lbl_w3, _ = ndimage.label(w3, structure=struct8)
sz_w3 = np.bincount(lbl_w3.ravel())
fill_w3 = (sz_w3 > 0) & (sz_w3 < 320)
fill_w3[0] = False
thick_ink = thick_ink | fill_w3[lbl_w3]

# Gaussian smooth for crisp vector curves
smooth = ndimage.gaussian_filter(thick_ink.astype(np.float32), sigma=2.0) > 0.46

# Final cleanup of any tiny islands/holes
lbl4, _ = ndimage.label(smooth, structure=struct8)
sz4 = np.bincount(lbl4.ravel())
k4 = (sz4 >= 800) | ((sz4 >= 350) & np.isin(np.arange(len(sz4)), np.unique(lbl4[type_mask])))
k4[0] = False
smooth = k4[lbl4]

w4 = ~smooth
lbl_w4, _ = ndimage.label(w4, structure=struct8)
sz_w4 = np.bincount(lbl_w4.ravel())
fw4 = (sz_w4 > 0) & (sz_w4 < 320)
fw4[0] = False
smooth = smooth | fw4[lbl_w4]

print(f"Final ink components: {ndimage.label(smooth, structure=struct8)[1]}, Final white components: {ndimage.label(~smooth, structure=struct8)[1]}")
out_bw = np.where(smooth, 0, 255).astype(np.uint8)
Image.fromarray(out_bw).save("/tmp/fassett_clean_v2.png")
