import numpy as np
from PIL import Image
from scipy import ndimage
import math

proj_dir = "/Users/frank/Work/13_money/02_开发/fassett-farm-vinyl"
src_master = "/Users/frank/.gemini/antigravity/brain/80f48f5c-03f1-44aa-ac8e-ce2aa8d2e90f/fassett_vinyl_simplified_master_1794120926206.jpg"
cp_img = Image.open(src_master).convert("L")
# Resize to 1600x1600 for clean high-precision contour extraction
W, H = 1600, 1600
cp_img = cp_img.resize((W, H), Image.Resampling.LANCZOS)
gray = np.array(cp_img)
ink = gray < 165

# Clean any stray < 40px specks or holes
struct8 = np.ones((3, 3), dtype=bool)
lbl, _ = ndimage.label(ink, structure=struct8)
sz = np.bincount(lbl.ravel())
k = sz >= 45
k[0] = False
ink = k[lbl]

w_mask = ~ink
lbl_w, _ = ndimage.label(w_mask, structure=struct8)
sz_w = np.bincount(lbl_w.ravel())
fw = (sz_w > 0) & (sz_w < 45)
fw[0] = False
ink = ink | fw[lbl_w]

# Separate ink into 2 clean Spot Color layers (zero overlap, clean registration!):
# Wait: how should the 2-color version separate cleanly into single layers for easy weeding and application?
# Look at our truck door mockup (fassett_truck_door_mockup_1794120950905.jpg):
# - Layer 1 (Primary Structure & Frame): Arched "FASSETT FARM" & "NURSERY" text, Double Circle Ring, Mount Monadnock horizon ridge, hedgerow line, and ground slope.
# - Layer 2 (Botanical Accent — Gold / Sage): Left 3 Echinacea coneflowers + leaves, Right 3 Yarrow flower clusters + fern leaves, and the central Tree crown/trunk!
# Wait: what if we ALSO provide a 2-color version where the flower petals/umbels/tree have a solid color fill + dark outline, AND a 2-color pure cut-vinyl version where Layer 1 is the structural ring/text/mountain and Layer 2 is the botanical flora + tree?
# Let's build BOTH!
# First, let's segment the pixels of `ink` into `layer1_mask` (Typography + Double Ring + Mountain + Hedgerow/Ground) vs `layer2_mask` (Left Coneflowers + Right Yarrow + Central Tree)!
yy, xx = np.mgrid[0:H, 0:W]
cx, cy = 800, 807  # Center of double circle ring
r = np.hypot(xx - cx, yy - cy)

# Let's identify botanical regions:
# 1. Left Echinacea flowers & leaves:
# Coneflowers are on lower-left: x < 660, y > 710, excluding the double ring where it doesn't touch flowers
left_flora = (xx < 655) & (yy > 715) & (yy < 1475)
# Wait: trim `left_flora` so it doesn't grab the "N" of NURSERY (NURSERY starts around x >= 445, y >= 1340)
nursery_box = (xx >= 440) & (xx <= 1145) & (yy >= 1320)
left_flora = left_flora & (~nursery_box)

# 2. Right Yarrow clusters & fern leaves:
# Yarrow is on lower-right: x > 985, y > 645, y < 1400
right_flora = (xx > 985) & (yy > 645) & (yy < 1400) & (~nursery_box)
# Also include the inner yarrow cluster that extends to x > 985..1580

# 3. Central Tree crown (above the trunk/hedgerow):
central_tree = (xx >= 575) & (xx <= 855) & (yy >= 610) & (yy <= 940)

layer2_region = left_flora | right_flora | central_tree
# Smooth the transition slightly and snap to connected white/ink features
layer2_ink = ink & layer2_region
layer1_ink = ink & (~layer2_region)

# Also create a solid Spot Fill layer for the interior of the Coneflower petals, Coneflower cones, and Yarrow florets!
# In `w_mask` (the white enclosed cells), any enclosed white cell inside `left_flora` or `right_flora` (with area between 50 and 28,000 px) is an interior petal/floret/leaf cell!
enclosed_floral_cells = np.zeros_like(ink)
for i in range(2, len(sz_w)):
    if 45 <= sz_w[i] <= 32000:
        cell = (lbl_w == i)
        # Check if centroid or majority of cell is inside left_flora or right_flora
        if np.mean(layer2_region[cell]) > 0.6:
            enclosed_floral_cells |= cell

# Dilate `enclosed_floral_cells` by 2px so it tucks cleanly under the dark line art (trap/bleed for 2-layer vinyl application)
y2, x2 = np.ogrid[-2:3, -2:3]
disk2 = (x2*x2 + y2*y2) <= 4
floral_fill_plate = ndimage.binary_dilation(enclosed_floral_cells, structure=disk2)

print("Segmented Layer 1 (Structure/Type), Layer 2 (Botanical Linework), and Floral Backing Plate!")

# Save high-res PNG previews of all variations so we can inspect them and embed clean vector paths
# 1. 1-Color Knockout Positive (Black on White)
img_1c_pos = np.where(ink, 20, 255).astype(np.uint8)
Image.fromarray(img_1c_pos).save(f"{proj_dir}/exports/01_Fassett_Farm_1Color_Positive_Black.png")

# 2. 1-Color Knockout Reversed (White on Dark Charcoal #161E19)
rgba_rev = np.zeros((H, W, 4), dtype=np.uint8)
rgba_rev[:, :, 0] = 255
rgba_rev[:, :, 1] = 255
rgba_rev[:, :, 2] = 255
rgba_rev[:, :, 3] = np.where(ink, 255, 0)
Image.fromarray(rgba_rev).save(f"{proj_dir}/exports/02_Fassett_Farm_1Color_Knockout_White_Transparent.png")

# 3. 2-Color Spot Version A (Monadnock Forest Green #1B4332 + Botanical Harvest Gold #D99B26 on White)
rgb_2c_light = np.full((H, W, 3), 255, dtype=np.uint8)
# Backing gold fill in petals/florets OR 2-color direct vinyl cut:
# Wait: for vinyl cutting, a 2-color vinyl decal has TWO solid vinyl colors (Spot 1 + Spot 2) with NO 3rd color!
# Let's make sure the 2-color version uses strictly TWO vinyl colors:
# Color 1 (Monadnock Forest Green #1B4332): Frame, Typography, Mountain, Hedgerow, Ground
# Color 2 (Harvest Gold #D48C28 / Terracotta Gold): Coneflowers, Yarrow, Central Tree (OR Color 1 = Forest Green all linework + Color 2 = Gold petal/cone fills!)
# Wait: in a 2-color vinyl decal on a white truck, the truck body is White, Vinyl Roll 1 is Forest Green (#1B4332), and Vinyl Roll 2 is Harvest Gold (#D99B26)!
# And on a dark green/black truck, the truck body is Dark, Vinyl Roll 1 is Warm Ivory Cream (#F4EFE6), and Vinyl Roll 2 is Harvest Gold (#E09F3E) — JUST LIKE OUR TRUCK MOCKUP (`fassett_truck_door_mockup_1794120950905.jpg`)!
# Look at `fassett_truck_door_mockup_1794120950905.jpg`:
# In the truck mockup:
# - The Typography, Double Ring, Mountain, and Ground line are Warm Ivory Cream (#F4EFE6)
# - The Coneflowers, Yarrow, and Central Tree are Harvest Gold (#E09F3E), with Cream outlines on the Coneflower petals!
# Let's match that exact 2-color separation so the vector files and the truck mockup are 100% consistent!

# Let's create:
# (A) 2-Color Dark Bodywork Palette (Warm Cream #F4EFE6 + Harvest Gold #E09F3E on Dark Forest Panel #132A1F)
rgb_2c_dark = np.zeros((H, W, 3), dtype=np.uint8)
rgb_2c_dark[:, :] = [19, 42, 31]  # #132A1F dark truck green background
# Put Harvest Gold (#E09F3E -> [224, 159, 62]) on floral_fill_plate AND layer2_ink
gold_mask = layer2_ink | enclosed_floral_cells
# Wait: if both `layer2_ink` and `enclosed_floral_cells` are Gold, they would merge!
# Instead:
# In the 2-color separated vinyl cut:
# - Spot Color 1 (Forest Green #1B4332 on light bodywork, or Cream #F4EFE6 on dark bodywork): `layer1_ink`
# - Spot Color 2 (Harvest Gold #D48C28 / #E09F3E): `layer2_ink`!
# Because `layer1_ink` and `layer2_ink` are both subsets of `ink` (`layer1_ink | layer2_ink == ink`), every petal and floret stays 100% crisp and open, and the two vinyl colors (`layer1_ink` and `layer2_ink`) never overlap or merge into blobs!
rgb_2c_dark[layer1_ink] = [244, 239, 230]  # #F4EFE6 Warm Cream
rgb_2c_dark[layer2_ink] = [224, 159, 62]   # #E09F3E Harvest Gold
Image.fromarray(rgb_2c_dark).save(f"{proj_dir}/exports/04_Fassett_Farm_2Color_Dark_Bodywork.png")

# (B) 2-Color Light Bodywork Palette (Monadnock Forest Green #1B4332 + Harvest Gold #C87D20 on White/Cream)
rgb_2c_light[layer1_ink] = [27, 67, 50]    # #1B4332 Forest Green
rgb_2c_light[layer2_ink] = [200, 125, 32]  # #C87D20 Harvest Amber Gold
Image.fromarray(rgb_2c_light).save(f"{proj_dir}/exports/03_Fassett_Farm_2Color_Light_Bodywork.png")

print("Saved PNG previews of 1-color and 2-color spot separations!")
