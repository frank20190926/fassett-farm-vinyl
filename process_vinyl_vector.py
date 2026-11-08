import numpy as np
from PIL import Image
from scipy import ndimage

img_path = "/Users/frank/Work/13_money/02_开发/fassett-farm-vinyl/assets/FassettFarmNursery_BW-jpeg.jpg"
gray = np.array(Image.open(img_path).convert("L"))
h, w = gray.shape
print(f"Loaded {w}x{h}")

# Binary mask of ink (True = black ink, False = white background)
ink = gray < 180

# 1. Remove tiny isolated black specks (floating grass ticks, dust, tiny dashes in meadow/sky)
struct8 = np.ones((3, 3), dtype=bool)
labeled_ink, num_ink = ndimage.label(ink, structure=struct8)
sizes_ink = np.bincount(labeled_ink.ravel())
# Remove components with area < 550 px
keep_ink = sizes_ink >= 550
keep_ink[0] = False
clean_ink = keep_ink[labeled_ink]
print(f"Removed {int(np.sum((sizes_ink > 0) & (sizes_ink < 550)))} tiny floating black ticks/specks")

# 2. Thicken ultra-thin strokes using circular disk structuring element
y, x = np.ogrid[-3:4, -3:4]
disk3 = (x*x + y*y) <= 9
thick_ink = ndimage.binary_dilation(clean_ink, structure=disk3)

# 3. Remove tiny enclosed white pinholes (weeding traps inside tree foliage, coneflower cones, and yarrow florets)
white = ~thick_ink
labeled_w, num_w = ndimage.label(white, structure=struct8)
sizes_w = np.bincount(labeled_w.ravel())
# Any white pinhole < 900 px gets filled into solid ink
fill_w = (sizes_w > 0) & (sizes_w < 900)
fill_w[0] = False
thick_ink = thick_ink | fill_w[labeled_w]
print(f"Consolidated {int(np.sum(fill_w))} microscopic white pinholes (weeding traps)")

# 4. Smooth contours with Gaussian blur + threshold for crisp plotter curves
smooth = ndimage.gaussian_filter(thick_ink.astype(np.float32), sigma=2.2) > 0.45

# Final pass to remove any tiny islands or pinholes after smoothing
lbl2, _ = ndimage.label(smooth, structure=struct8)
sz2 = np.bincount(lbl2.ravel())
k2 = sz2 >= 600
k2[0] = False
smooth = k2[lbl2]

w2 = ~smooth
lbl_w2, _ = ndimage.label(w2, structure=struct8)
sz_w2 = np.bincount(lbl_w2.ravel())
fw2 = (sz_w2 > 0) & (sz_w2 < 900)
fw2[0] = False
smooth = smooth | fw2[lbl_w2]

out_bw = np.where(smooth, 0, 255).astype(np.uint8)
Image.fromarray(out_bw).save("/tmp/fassett_simplified_bw.png")
print("Saved /tmp/fassett_simplified_bw.png")
