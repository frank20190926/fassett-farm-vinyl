import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math

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
    (388, 895), (538, 955), (552, 1028), (548, 1152),
    (505, 1158), (430, 1155), (465, 1188), (658, 1192),
    (658, 1265), (560, 1280), (470, 1270), (492, 1345),
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
    (845, 775), (810, 825), (805, 902), (735, 902),
    (735, 815), (675, 808), (580, 770)
]
draw.polygon(poly_tree_crown, fill=255)

botanical_poly = np.array(mask_img) > 127

yy, xx = np.mgrid[0:H, 0:W]
cx, cy = 799.0, 804.0
rr = np.hypot(xx - cx, yy - cy)

ring_guard_br = (xx >= 800) & (xx < 1150) & (yy > 1175)
ring_guard_bl = (rr >= 510) & (rr <= 562) & (xx > 475) & (xx < 800) & (yy > 1266)

layer2_region = botanical_poly & (~ring_guard_br) & (~ring_guard_bl)

layer2_ink = ink & layer2_region
layer1_ink = ink & (~layer2_region)

# Save updated PNGs
rgb_2c_light = np.full((H, W, 3), 255, dtype=np.uint8)
rgb_2c_light[layer1_ink] = [27, 67, 50]    # #1B4332 Monadnock Forest Green
rgb_2c_light[layer2_ink] = [200, 125, 32]  # #C87D20 Botanical Harvest Gold
Image.fromarray(rgb_2c_light).save(f"{proj_dir}/exports/03_Fassett_Farm_2Color_Light_Bodywork.png")

rgb_2c_dark = np.zeros((H, W, 3), dtype=np.uint8)
rgb_2c_dark[:, :] = [19, 42, 31]           # #132A1F Dark Truck Panel
rgb_2c_dark[layer1_ink] = [244, 239, 230]  # #F4EFE6 Warm Ivory Cream
rgb_2c_dark[layer2_ink] = [224, 159, 62]   # #E09F3E Botanical Harvest Gold
Image.fromarray(rgb_2c_dark).save(f"{proj_dir}/exports/04_Fassett_Farm_2Color_Dark_Bodywork.png")

plate1 = np.full((H, W, 3), 255, dtype=np.uint8)
plate1[layer1_ink] = [27, 67, 50]
Image.fromarray(plate1).save(f"{proj_dir}/exports/05_Spot_Layer_1_Base_Structure.png")

plate2 = np.full((H, W, 3), 255, dtype=np.uint8)
plate2[layer2_ink] = [200, 125, 32]
Image.fromarray(plate2).save(f"{proj_dir}/exports/06_Spot_Layer_2_Botanical_Accent.png")

Image.open("/tmp/fassett_faithful_v3.png").save(f"{proj_dir}/exports/07_Fassett_Farm_Faithful_Original_Serif_BW.png")

def rdp(pts, eps):
    if len(pts) < 3:
        return pts
    dmax = 0.0
    idx = 0
    x1, y1 = pts[0]
    x2, y2 = pts[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy)
    for i in range(1, len(pts) - 1):
        px, py = pts[i]
        if norm == 0:
            d = math.hypot(px - x1, py - y1)
        else:
            d = abs(dy * px - dx * py + x2 * y1 - y2 * x1) / norm
        if d > dmax:
            idx = i
            dmax = d
    if dmax > eps:
        r1 = rdp(pts[:idx+1], eps)
        r2 = rdp(pts[idx:], eps)
        return r1[:-1] + r2
    else:
        return [pts[0], pts[-1]]

def mask_to_bezier_svg_path(binary_mask, tolerance=1.7):
    fig, ax = plt.subplots()
    cs = ax.contour(binary_mask.astype(np.float32), levels=[0.5])
    plt.close(fig)
    subpaths = []
    node_pts = []
    segs = cs.allsegs[0]
    for c in segs:
        if len(c) < 8:
            continue
        pts_raw = [(float(p[0]), float(p[1])) for p in c]
        simp = rdp(pts_raw, tolerance)
        pts = simp[:-1] if len(simp) > 3 and math.hypot(simp[0][0]-simp[-1][0], simp[0][1]-simp[-1][1]) < 2.0 else simp
        n = len(pts)
        if n < 3:
            continue
        for p in pts[::3]:
            node_pts.append(p)
        d = [f"M {pts[0][0]:.1f},{pts[0][1]:.1f}"]
        for i in range(n):
            p0 = pts[(i - 1) % n]
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            p3 = pts[(i + 2) % n]
            t = 0.18
            cp1x = p1[0] + (p2[0] - p0[0]) * t
            cp1y = p1[1] + (p2[1] - p0[1]) * t
            cp2x = p2[0] - (p3[0] - p1[0]) * t
            cp2y = p2[1] - (p3[1] - p1[1]) * t
            d.append(f"C {cp1x:.1f},{cp1y:.1f} {cp2x:.1f},{cp2y:.1f} {p2[0]:.1f},{p2[1]:.1f}")
        d.append("Z")
        subpaths.append(" ".join(d))
    return " ".join(subpaths), node_pts

d_layer1, nodes1 = mask_to_bezier_svg_path(layer1_ink, tolerance=1.7)
d_layer2, nodes2 = mask_to_bezier_svg_path(layer2_ink, tolerance=1.7)
d_full, nodes_full = mask_to_bezier_svg_path(ink, tolerance=1.7)

print(f"Vectorized Layer 1 ({len(nodes1)} nodes), Layer 2 ({len(nodes2)} nodes), Full 1-Color ({len(nodes_full)} nodes)")

svg_2c_light = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">
  <title>Fassett Farm Nursery — 2-Color Spot Separated Vinyl Cut Vector (Light Bodywork)</title>
  <rect width="{W}" height="{H}" fill="#FFFFFF"/>
  <g id="Layer_1_Spot_Forest_Green_Oracal_651_060" fill="#1B4332" fill-rule="evenodd">
    <path d="{d_layer1}"/>
  </g>
  <g id="Layer_2_Spot_Harvest_Gold_Oracal_651_020" fill="#C87D20" fill-rule="evenodd">
    <path d="{d_layer2}"/>
  </g>
</svg>'''
open(f"{proj_dir}/exports/01_Fassett_Farm_2Color_Spot_Separated_Light.svg", "w").write(svg_2c_light)

svg_2c_dark = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">
  <title>Fassett Farm Nursery — 2-Color Spot Separated Vinyl Cut Vector (Dark Bodywork)</title>
  <rect width="{W}" height="{H}" fill="#132A1F"/>
  <g id="Layer_1_Spot_Warm_Cream_Oracal_651_023" fill="#F4EFE6" fill-rule="evenodd">
    <path d="{d_layer1}"/>
  </g>
  <g id="Layer_2_Spot_Harvest_Gold_Oracal_651_020" fill="#E09F3E" fill-rule="evenodd">
    <path d="{d_layer2}"/>
  </g>
</svg>'''
open(f"{proj_dir}/exports/02_Fassett_Farm_2Color_Spot_Separated_Dark.svg", "w").write(svg_2c_dark)

svg_1c_pos = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">
  <title>Fassett Farm Nursery — 1-Color Single-Pass Vinyl Cut Vector (Positive Black)</title>
  <rect width="{W}" height="{H}" fill="#FFFFFF"/>
  <g id="Single_Pass_Knockout_Black" fill="#141816" fill-rule="evenodd">
    <path d="{d_full}"/>
  </g>
</svg>'''
open(f"{proj_dir}/exports/03_Fassett_Farm_1Color_Knockout_Positive.svg", "w").write(svg_1c_pos)

svg_1c_rev = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">
  <title>Fassett Farm Nursery — 1-Color Single-Pass Vinyl Cut Vector (Reversed White Knock-Out)</title>
  <rect width="{W}" height="{H}" fill="#141816"/>
  <g id="Single_Pass_Knockout_White" fill="#FFFFFF" fill-rule="evenodd">
    <path d="{d_full}"/>
  </g>
</svg>'''
open(f"{proj_dir}/exports/04_Fassett_Farm_1Color_Knockout_Reversed_White.svg", "w").write(svg_1c_rev)

node_rects = "\n    ".join([f'<rect x="{x-3.5:.1f}" y="{y-3.5:.1f}" width="7.0" height="7.0" fill="#FFFFFF" stroke="#0066FF" stroke-width="1.8"/>' for (x, y) in (nodes1[::2] + nodes2[::2])])
svg_cad = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">
  <title>Fassett Farm Nursery — Plotter Cut Path Inspection (Ctrl+Y Outline Mode)</title>
  <rect width="{W}" height="{H}" fill="#F8FAFC"/>
  <g fill="none" stroke="#0066FF" stroke-width="2.4" fill-rule="evenodd">
    <path d="{d_layer1}"/>
  </g>
  <g fill="none" stroke="#D97706" stroke-width="2.4" fill-rule="evenodd">
    <path d="{d_layer2}"/>
  </g>
  <g id="Anchor_Points">
    {node_rects}
  </g>
</svg>'''
open(f"{proj_dir}/exports/05_Fassett_Farm_CAD_Plotter_Wireframe.svg", "w").write(svg_cad)
print("All SVG vector files generated successfully!")
