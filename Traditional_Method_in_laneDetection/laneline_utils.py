import edge_detection as ed
import numpy as np

def filter_colors(image, mask, form = "jpg"):
    color_mask = mask(image)
#     sky_mask = np.zeros_like(image[:, :, 0], dtype = bool)
#     sky_mask[:image.shape[0]//2, :] = True
    filtered_image = np.zeros_like(image[:, :, 0])
    filtered_image[color_mask] = 255
#     filtered_image = image.copy()
    
#     for channel in range(3):
#         filtered_image[:, :, channel][(~color_mask) | sky_mask] //= 3

    return filtered_image

def white_mask(image, form = "jpg"):
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    return ((red_channel >= 200) & (green_channel >= 200) & (blue_channel >= 200))

def yellow_mask(image, form = "jpg"):
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    return ((red_channel >= 150) & (green_channel >= 100) & (blue_channel <= 130) \
           & (green_channel - blue_channel >= 50)  & (red_channel - blue_channel >= 50)) \
           & ~ ((red_channel >= 200) & (green_channel >= 200) & (blue_channel >= 200)) 

def line_detect(image, threshold = 60, neighborhood_size = 10):
    accumulator, theta_vals, rho_vals = ed.hough_transform(image)
    peaks = ed.find_hough_peaks(
        accumulator, theta_vals, rho_vals, 
        threshold = threshold, 
        neighborhood_size = neighborhood_size
    )
    return peaks

def region_of_interest(edges, vertices):
    pass

import cv2
def find_image_lines(
    image, 
    form = "jpg", 
    blur_kernel_size = 5,
    blur_sigma = 1,
    dev_kernel_size = 9,
    dev_sigma = 0.4,
    raise_contrast_ratio = False
):
    img_yellow = filter_colors(image, mask = yellow_mask, form = form)
    img_white = filter_colors(image, mask = white_mask, form = form)
    
    img_yed = ed.edge_detect(
        img_yellow, 
        is_gray = True,
        blur_kernel_size = blur_kernel_size,
        blur_sigma = blur_sigma,
        dev_kernel_size = dev_kernel_size,
        dev_sigma = dev_sigma,
        lower_threshold = 5, 
        upper_threshold = 12,
        form = form,
        raise_contrast_ratio = raise_contrast_ratio
    )
    
    img_wed = ed.edge_detect(
        img_white, 
        is_gray = True,
        blur_kernel_size = blur_kernel_size,
        blur_sigma = blur_sigma,
        dev_kernel_size = dev_kernel_size,
        dev_sigma = dev_sigma,
        lower_threshold = 5, 
        upper_threshold = 12,
        form = form,
        raise_contrast_ratio = raise_contrast_ratio
    )
#     img_yed = cv2.Canny(img_yellow, 50, 150)
#     img_wed = cv2.Canny(img_white, 50, 150)
    yellow_peaks = line_detect(img_yed, threshold = 60)
    white_peaks = line_detect(img_wed, threshold = 60)
    
    return white_peaks, yellow_peaks, img_wed, img_yed
    
#     image = cv2.Canny(image, 50, 150)
#     peaks = line_detect(image, threshold = 60)
#     return peaks, image
    
def breakdown_lines(edges, lines, nsize = 10):
    break_lines = []
    nrow, ncol = edges.shape
    
    for rho, theta in lines:
        st = np.sin(theta)
        ct = np.cos(theta)
        flg = 0
        start = (0, rho / st)
        end = (0, rho / st)
        for tmpr in range(2 * nrow // 5, nrow):
            tmpc = int((rho - tmpr * ct) / (st + 0.0001))
            if not (0 <= tmpc < ncol):
                continue
            neighborhood = edges[max(0, tmpr - nsize // 2): min(nrow, tmpr + nsize // 2 + 1),
                                 max(0, tmpc - nsize // 2): min(ncol, tmpc + nsize // 2 + 1)]
            if np.any(neighborhood != 0):
                if flg == 0:
                    start = (tmpr, tmpc)
                    flg = 1
            else:
                if flg == 1:
                    end = (tmpr, tmpc)
                    flg = 0
                    break_lines.append((start, end))
        if start[0] <= end[0]:
            break_lines.append((start, end))
            
    return break_lines

def slope(line):
    start, end = line
    x1, y1 = start
    x2, y2 = end
    if x1 == x2:
        return 114514
    return (y2 - y1) / (x2 - x1)
    

def find_lane_lines(
    img, 
    form = "jpg", 
    slope_thresh = 2,
    blur_kernel_size = 5,
    blur_sigma = 1,
    dev_kernel_size = 9,
    dev_sigma = 0.4,
    raise_contrast_ratio = False
):
#     scale = 255.0 if form == "png" else 1.0
    
#     wp, yp, wed, yed = find_image_lines(img, form = "jpg", raise_contrast_ratio = raise_contrast_ratio)
#     if form == "png":
#         img = (img * 255).astype(np.uint8)
#     img = ed.enhance_contrast(img, alpha = 1.2, scale = 1.0)
#     img = ed.enhance_saturation(img, saturation_factor = 1.2, scale = 1.0)
#     img = ed.enhance_saturation(img, saturation_factor = 2, scale = 1.0)
    wp, yp, wed, yed = find_image_lines(
        img, 
        form = form, 
        blur_kernel_size = blur_kernel_size,
        blur_sigma = blur_sigma,
        dev_kernel_size = dev_kernel_size,
        dev_sigma = dev_sigma,
        raise_contrast_ratio = raise_contrast_ratio)
    wl = breakdown_lines(wed, wp, nsize = 10)
    yl = breakdown_lines(yed, yp, nsize = 10)
    
    return [line for line in wl if abs(slope(line)) < slope_thresh], \
           [line for line in yl if abs(slope(line)) < slope_thresh]