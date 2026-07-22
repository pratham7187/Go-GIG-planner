import cv2
import numpy as np

img = cv2.imread(r'uploads\3113555f-4326-48e6-83d4-ed5325aecbc3.jpeg', cv2.IMREAD_GRAYSCALE)
h, w = img.shape
print(f'Image: {w}x{h}')
print('Scanning slices from bottom:')
slice_h = max(15, h // 30)

for y in range(h - slice_h, h // 2, -slice_h):
    band = img[y:y + slice_h, :]
    above = img[max(0, y - slice_h):y, :]
    bstd = float(band.std())
    astd = float(above.std())
    bmean = float(band.mean())
    amean = float(above.mean())
    mark = ' ***BAND***' if (bstd < 35 and astd > bstd * 1.5 and abs(bmean - amean) > 20) else ''
    if not mark:
        # Also check: dark band with text (higher std than pure uniform, but lower than natural image)
        mark2 = ' **DARK_BAND**' if (bmean < 100 and bstd < astd * 0.7 and abs(bmean - amean) > 25) else ''
    else:
        mark2 = ''
    print(f'  y={y}: mean={bmean:.1f} std={bstd:.1f} | above: mean={amean:.1f} std={astd:.1f}{mark}{mark2}')

# Also check corners for watermark
print('\nCorner analysis:')
corner_h = max(30, h // 10)
corner_w = max(40, w // 8)
corners = {
    'bottom_right': img[h - corner_h:, w - corner_w:],
    'bottom_left': img[h - corner_h:, :corner_w],
}
for name, corner in corners.items():
    edges = cv2.Canny(corner, 50, 150)
    edge_density = float(edges.mean()) / 255.0
    cstd = float(corner.std())
    cmean = float(corner.mean())
    print(f'  {name}: mean={cmean:.1f} std={cstd:.1f} edge_density={edge_density:.3f}')

# Check overall image center for comparison
center = img[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
center_edges = cv2.Canny(center, 50, 150)
print(f'  center: mean={float(center.mean()):.1f} std={float(center.std()):.1f} edge_density={float(center_edges.mean()) / 255.0:.3f}')
