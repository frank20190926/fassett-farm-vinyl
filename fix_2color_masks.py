import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

proj_dir = "/Users/frank/Work/13_money/02_开发/fassett-farm-vinyl"
src_master = "/Users/frank/.gemini/antigravity/brain/80f48f5c-03f1-44aa-ac8e-ce2aa8d2e90f/fassett_vinyl_simplified_master_1794120926206.jpg"
W, H = 1600, 1600
cp_img = Image.open(src_master).convert("L").resize((W, H), Image.Resampling.LANCZOS)
gray = np.array(cp_img)
ink = gray < 165

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

mask_img = Image.new("L", (W, H), 0)
draw = ImageDraw.Draw(mask_img)

# Polygon 1: Left 3 Echinacea Coneflowers + stems + leaves
poly_coneflowers = [
    (20, 710), (248, 715), (320, 765), (370, 860),
    (388, 895), (538, 955), (552, 1028), (548, 1145),
    (495, 1148), (430, 1150), (465, 1190), (658, 1195),
    (658, 1255), (560, 1278), (470, 1270), (492, 1345),
    (455, 1358), (395, 1300), (330, 1410), (235, 1490),
    (20, 1490)
]
draw.polygon(poly_coneflowers, fill=255)

# Polygon 2: Right 3 Yarrow clusters + stems + fern leaves
poly_yarrow = [
    (1235, 640), (1590, 640), (1590, 1150),
    (1500, 1250), (1320, 1400), (1135, 1400),
    (1148, 1235), (1162, 1175), (1020, 1110), (988, 1005),
    (992, 910), (1065, 835), (1165, 805), (1228, 760)
]
draw.polygon(poly_yarrow, fill=255)

# Polygon 3: Central Tree Canopy (including small right branch foliage)
poly_tree_crown = [
    (575, 740), (630, 610), (795, 610), (850, 690),
    (845, 775), (792, 815), (750, 818), (675, 808), (580, 770)
]
draw.polygon(poly_tree_crown, fill=255)

botanical_poly = np.array(mask_img) > 127

yy, xx = np.mgrid[0:H, 0:W]
cx, cy = 799.0, 804.0
rr = np.hypot(xx - cx, yy - cy)

# Only guard the bottom-right ring (x between 800 and 1150, y > 1175)
ring_guard_br = (xx >= 800) & (xx < 1150) & (yy > 1175)
# Only guard the bottom-left ring where the coneflower leaves don't cover it
ring_guard_bl = (rr >= 510) & (rr <= 562) & (xx > 475) & (xx < 800) & (yy > 1255)

layer2_region = botanical_poly & (~ring_guard_br) & (~ring_guard_bl)

layer2_ink = ink & layer2_region
layer1_ink = ink & (~layer2_region)

# Save fixed PNGs to exports/
rgb_2c_light = np.full((H, W, 3), 255, dtype=np.uint8)
rgb_2c_light[layer1_ink] = [27, 67, 50]    # #1B4332 Monadnock Forest Green
rgb_2c_light[layer2_ink] = [200, 125, 32]  # #C87D20 Botanical Harvest Gold
Image.fromarray(rgb_2c_light).save(f"{proj_dir}/exports/03_Fassett_Farm_2Color_Light_Bodywork.png")
Image.fromarray(rgb_2c_light).save("/tmp/fassett_2c_light_v2.png")

rgb_2c_dark = np.zeros((H, W, 3), dtype=np.uint8)
rgb_2c_dark[:, :] = [19, 42, 31]           # #132A1F Dark Truck Panel
rgb_2c_dark[layer1_ink] = [244, 239, 230]  # #F4EFE6 Warm Ivory Cream
rgb_2c_dark[layer2_ink] = [224, 159, 62]   # #E09F3E Botanical Harvest Gold
Image.fromarray(rgb_2c_dark).save(f"{proj_dir}/exports/04_Fassett_Farm_2Color_Dark_Bodywork.png")
Image.fromarray(rgb_2c_dark).save("/tmp/fassett_2c_dark_v2.png")

# Also save individual Spot Layer 1 Plate and Spot Layer 2 Plate for the separation proof!
plate1 = np.full((H, W, 3), 255, dtype=np.uint8)
plate1[layer1_ink] = [27, 67, 50]
Image.fromarray(plate1).save(f"{proj_dir}/exports/05_Spot_Layer_1_Base_Structure.png")

plate2 = np.full((H, W, 3), 255, dtype=np.uint8)
plate2[layer2_ink] = [200, 125, 32]
Image.fromarray(plate2).save(f"{proj_dir}/exports/06_Spot_Layer_2_Botanical_Accent.png")

print("Saved v2 2-color previews and separated spot plates!")
