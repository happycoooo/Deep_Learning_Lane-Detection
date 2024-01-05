import numpy as np
import pandas as pd
from scipy.ndimage import convolve, convolve1d
from skimage import exposure

def rgb_to_gray(image, scale = 1):
    """Generate a gray scale image from an RGB(A) image."""
    return (np.dot(image[..., :3], [0.299, 0.587, 0.114]) * scale).astype(np.uint8)

def histogram_equalization(image):
    """To raise the image's contrast ratio"""
    equalized_image = exposure.equalize_hist(image)
    return (equalized_image * 255).astype(np.uint8)

def sharpen(image):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])

    sharpened_image = convolve(image, kernel)
    sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
    return sharpened_image


def gaussian_kernel(size, sigma = 1.0):
    """Generate a Gaussian kernel with standard deviation = sigma within the interval [-size, size]."""
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * np.exp(- ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def gaussian_blur(image, kernel_size = 5, sigma = 1):
    """Apply Guassian Blurring to the image."""
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve(image, kernel)

def gaussian_derivative_kernel(size = 5, sigma = 1, order = 1):
    """Generate a 1D Gaussian derivative kernel."""
    if size % 2 == 0:
        size += 1
    kernel = np.fromfunction(
        lambda x: (x - (size - 1) / 2) * np.exp(-((x - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size,)
    )
    kernel = (-1) ** order * kernel / (np.sqrt(2 * np.pi) * sigma ** 3)
    return kernel

def apply_gradient_operator(image, kernel = gaussian_derivative_kernel()):
    """Apply the gradient operator to the image using the provided kernels."""
    gradient_x = convolve1d(image, kernel, axis = -1, mode = 'reflect')
    gradient_y = convolve1d(image, kernel, axis = -2, mode = 'reflect')
    return gradient_x, gradient_y

def gaussian_derivative_kernel_2d(size=5, sigma=1, order=(1, 0)):
    """Generate a 2D Gaussian derivative kernel."""
    if size % 2 == 0:
        size += 1

    kernel_x = np.fromfunction(
        lambda x, y: (x - (size - 1) / 2) * np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )

    kernel_y = np.transpose(kernel_x)

    kernel_x = (-1) ** order[0] * kernel_x / (2 * np.pi * sigma ** 4)
    kernel_y = (-1) ** order[1] * kernel_y / (2 * np.pi * sigma ** 4)

    return kernel_x, kernel_y

def apply_gradient_operator_2d(image, kernel=gaussian_derivative_kernel_2d()):
    """Apply the gradient operator to the image using the provided 2D kernels."""
    gradient_x = convolve(image, kernel[0], mode='reflect')
    gradient_y = convolve(image, kernel[1], mode='reflect')
    return gradient_x, gradient_y

def sobel_operator(image):
    """Apply the Sobel operator to the image."""
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]]) / 4
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]]) / 4

    gradient_x = convolve(image, sobel_x, mode='reflect')
    gradient_y = convolve(image, sobel_y, mode='reflect')
    
    return gradient_x, gradient_y

def non_max_suppression(gradient_x, gradient_y):
    """Apply non-max suppression (NMS) to the image using its gradient images."""
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
#     print(np.max(gradient_magnitude))
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    rows, cols = gradient_magnitude.shape
    result = np.zeros_like(gradient_magnitude, dtype=np.uint8)

    for i in range(1, rows - 2):
        j = np.arange(1, cols - 2)
        mag = gradient_magnitude[i, j]
        angle = gradient_direction[i, j] % np.pi

        t = np.abs(np.tan(angle))
        mask = (-1 < t) & (t < 1)

        x1, y1 = i - 1, (j[mask] - t[mask]).astype(np.int16)
        x2, y2 = i - 1, (j[mask] - t[mask] + 1).astype(np.int16)
        x3, y3 = i + 1, (j[mask] + t[mask] + 1).astype(np.int16)
        x4, y4 = i + 1, (j[mask] + t[mask]).astype(np.int16)

        c1 = (1 - t[mask]) * gradient_magnitude[x1, y1] + t[mask] * gradient_magnitude[x2, y2]
        c2 = (1 - t[mask]) * gradient_magnitude[x3, y3] + t[mask] * gradient_magnitude[x4, y4]
        mask2 = np.zeros_like(mask, dtype = bool)
        mask2[mask] = (mag[mask] >= c1) & (mag[mask] >= c2)
        result[i, j[mask2]] = mag[mask2]
        
        mask = (-1 >= t) | (t >= 1)
        
        t = 1 / t
        x1, y1 = (i - t[mask]).astype(np.int16), j[mask] - 1
        x2, y2 = (i - t[mask] + 1).astype(np.int16), j[mask] - 1
        x3, y3 = (i + t[mask] + 1).astype(np.int16), j[mask] + 1
        x4, y4 = (i + t[mask]).astype(np.int16), j[mask] + 1

        c1 = (1 - t[mask]) * gradient_magnitude[x1, y1] + t[mask] * gradient_magnitude[x2, y2]
        c2 = (1 - t[mask]) * gradient_magnitude[x3, y3] + t[mask] * gradient_magnitude[x4, y4]

        mask2 = np.zeros_like(mask, dtype = bool)
        mask2[mask] = (mag[mask] >= c1) & (mag[mask] >= c2)
        result[i, j[mask2]] = mag[mask2]

    return result

def thinning_double_threshold(img, t1 = 1, t2 = 13):
    """Applying double threshold thinning algorithm to the image."""
    result = np.zeros_like(img)
    strong_edges = img >= t2
    weak_edges = (img >= t1) & (img < t2)

    result[strong_edges] = 255
    result[~(strong_edges | weak_edges)] = 0  

    flg = 1
    p = 1
    while(flg == 1):
        p += 1
        flg = 0
#         print(np.sum(weak_edges))
        for i, j in np.argwhere(weak_edges):
            if np.any(strong_edges[i - 2 : i + 3, j - 2 : j + 3]):
                result[i, j] = 255
                strong_edges[i, j] = True
                weak_edges[i, j] = False
                flg = 1
            elif not np.any(strong_edges[i - 1 : i + 2, j - 1 : j + 2]):
                weak_edges[i, j] = False
                    
    return result

def get_neighbors(img, i, j):
    """Retrieve the 8 neighbours of the pixel img[i, j]."""
    rows, cols = img.shape
    neighbors = []

    for x in range(max(0, i - 1), min(i + 2, rows)):
        for y in range(max(0, j - 1), min(j + 2, cols)):
            if x != i or y != j:
                neighbors.append(img[x, y])

    return np.array(neighbors)[np.array([1, 2, 4, 7, 6, 5, 3, 0])]

def thinning_zhangsuen(img, t1 = 0, t2 = 0):
    """Apply Zhang-Suen Thinning algorithm （张太怡-孫靖夷细化算法） to the image."""
    # 效果一般，速度贼慢
    result = np.zeros_like(img)
    result[img > t2] = 1
    
    rows, cols = result.shape
    
    to_be_processed = np.column_stack(np.where(img > t2))

    p = 0
    while p < 20:
        print(len(to_be_processed))
        p += 1
        to_delete = []
        
        # Iteration 1
        for point in to_be_processed:
            i, j = point
            neighbors = get_neighbors(img, i, j)
            
            # The indices of neighbors:
            # P9 P2 P3
            # P8 P1 P4
            # P7 P6 P5
            # Criterion 1A: 2 <= #(Neighbor == 1) >= 6
            # Criterion 1B: #(0 -> 1 when iterating using indices above) == 1
            # Criterion 1C: P2 * P4 * P6 == 0
            # Criterion 1D: P4 * P6 * P8 == 0
            if          (2 <= np.sum(neighbors) <= 6) \
                    and (np.sum(neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1 for i in range(0, 8)) == 1) \
                    and (neighbors[1] * neighbors[3] * neighbors[5] == 0) \
                    and (neighbors[3] * neighbors[5] * neighbors[7] == 0):
                to_delete.append(point)
                
        for point in to_delete:
            result[point[0], point[1]] = 0    
            
        to_be_processed = np.delete(to_be_processed, to_delete, axis=0)
        
        if len(to_delete) == 0:
            break
            
        to_delete = []
        
        # Iteration 2
        for point in to_be_processed:
            i, j = point
            neighbors = get_neighbors(result, i, j)
            
            # Criterion 2A: 2 <= #(Neighbor == 1) >= 6
            # Criterion 2B: #(0 -> 1 when iterating using indices above) == 1
            # Criterion 2C: P2 * P4 * P8 == 0
            # Criterion 2D: P2 * P6 * P8 == 0
            if          (2 <= np.sum(neighbors) <= 6) \
                    and (np.sum(neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1 for i in range(0, 8)) == 1) \
                    and (neighbors[1] * neighbors[3] * neighbors[7] == 0) \
                    and (neighbors[1] * neighbors[5] * neighbors[7] == 0):
                to_delete.append(point)
                
        for point in to_delete:
            result[point[0], point[1]] = 0    
            
        to_be_processed = np.delete(to_be_processed, to_delete, axis=0)
        
        if len(to_delete) == 0:
            break
    result[result > 0] = 255
    return result

def calculate_angle_image(grad_x, grad_y):
    """Return the image whose elements are arctan(grad_y / grad_x)."""
    return np.arctan2(grad_y, grad_x)

def edge_detect(
    img, 
    is_gray = False, 
    blur_kernel_size = 9, 
    blur_sigma = 1,
    dev_kernel_size = 9,
    dev_sigma = 0.4,
    lower_threshold = 5, 
    upper_threshold = 10, 
    sigma = 0.4, 
    form = "png",
    raise_contrast_ratio = False
):
    scale = 255 if form == "png" else 1
    if not is_gray:
        img = rgb_to_gray(img, scale)
    
    if raise_contrast_ratio:
        img = histogram_equalization(img)
    
    img = gaussian_blur(img, kernel_size = blur_kernel_size, sigma = blur_sigma)
    
    gx, gy = apply_gradient_operator_2d(
        img, 
        kernel = gaussian_derivative_kernel_2d(size = dev_kernel_size, sigma = dev_sigma)
    )

    nms = non_max_suppression(gx, gy)
    
    res = thinning_double_threshold(nms, lower_threshold, upper_threshold)

    return res


def hough_transform(edge_image, theta_res = 1, rho_res = 1, apply_mask = True):
    height, width = edge_image.shape
    diag_len = int(np.sqrt(height ** 2 + width ** 2))
    max_rho = diag_len
    theta_vals = np.deg2rad(np.arange(-90, 90, theta_res))
    rho_vals = np.arange(-max_rho, max_rho, rho_res)
    sin_vals = np.sin(theta_vals)
    cos_vals = np.cos(theta_vals)

    accumulator = np.zeros((len(rho_vals), len(theta_vals)), dtype=np.int32)

    edge_points = np.column_stack(np.where(edge_image > 0))
    for point in edge_points:
        x, y = point
        if apply_mask and x < height // 2:
            continue
        for theta_index, (sin_theta, cos_theta) in enumerate(zip(sin_vals, cos_vals)):
            rho = int(x * cos_theta + y * sin_theta)
            rho_index = np.argmin(np.abs(rho_vals - rho))
            accumulator[rho_index, theta_index] += 1

    return accumulator, theta_vals, rho_vals

def find_hough_peaks(accumulator, theta_vals, rho_vals, threshold = 100, neighborhood_size = 20):
    """
    To find peak values in the parameter space of Hough transformation.

    Parameters:
    - accumulator: Hough 变换的累加器
    - threshold: 阈值，用于确定峰值
    - neighborhood_size: 领域大小，用于确定峰值位置

    Return:
    - peaks: 峰值的坐标列表 [(rho_index, theta_index), ...]
        每个元素能够确定一条直线 x cos(θ) + y sin(θ) = ρ
    """
    row, col = np.where(accumulator > threshold)

    peaks = list(zip(row, col))

    filtered_peaks = []
    for peak in peaks:
        row, col = peak
        is_maximal = True

        for p_row, p_col in filtered_peaks:
            if (
                row - neighborhood_size < p_row < row + neighborhood_size
                and col - neighborhood_size < p_col < col + neighborhood_size
                and accumulator[row, col] < accumulator[p_row, p_col]
            ):
                is_maximal = False
                break

        if is_maximal:
            filtered_peaks.append((row, col))
    filtered_peaks = sorted(filtered_peaks, key=lambda x: accumulator[x[0], x[1]], reverse=True)
#     filtered_peaks = sorted(
#         filtered_peaks, 
#         key=lambda x: abs(row * np.cos(x[1]) + col / 2 * np.sin(x[1]) - x[0]), 
#         reverse=False)
    peaks = []
    for peak in filtered_peaks:
        flg = 1
        for p in peaks:
            if abs(p[0] - peak[0]) / 10 + abs(p[1] - peak[1]) <= neighborhood_size:
                flg = 0
                break
        if flg == 1:
            peaks.append(peak)
            
    return [(rho_vals[rho], theta_vals[theta]) for rho, theta in peaks]

def enhance_contrast(image, alpha = 1.5, beta = 0, scale = 255):
    # Apply contrast enhancement using the formula: new_pixel = alpha * pixel + beta
    enhanced_image = np.clip(alpha * (image * scale) + beta, 0, 255).astype(np.uint8)
    return enhanced_image

def enhance_saturation(img, saturation_factor = 1.5, scale = 255.0):

    # Normalization
    img_float = img.astype(float) / 255.0 * scale

    r, g, b = img_float[:,:,0], img_float[:,:,1], img_float[:,:,2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Calculate saturation
    saturation = np.where(gray < 0.5, (gray * (1 + saturation_factor)), ((1 - gray) * saturation_factor + gray))

    # Create a new image with adjusted saturation
    enhanced_image = np.stack([r * saturation, g * saturation, b * saturation], axis=-1)
    
    enhanced_image = np.clip(enhanced_image * 255.0, 0, 255).astype(np.uint8)
    return enhanced_image


# def gaussian_blur_color(image, kernel_size, sigma):
#     """Apply Gaussian blur to a three-channel image."""
#     result = np.zeros_like(image, dtype=np.float32)
#     for i in range(image.shape[2]):
#         result[..., i] = gaussian_blur(image[..., i], kernel_size, sigma)
#     return result

# def compute_all_channels_gradients(image, kernel):
#     """Compute gradients for all channels and combine them."""
#     all_gradients_x = np.zeros_like(image, dtype=np.float32)
#     all_gradients_y = np.zeros_like(image, dtype=np.float32)

#     for i in range(image.shape[-1]):
#         channel = image[..., i]
#         gradients_x, gradients_y = apply_gradient_operator(channel, kernel)
#         all_gradients_x[..., i] = gradients_x
#         all_gradients_y[..., i] = gradients_y

#     return all_gradients_x, all_gradients_y

# def non_max_suppression(gradient_x, gradient_y):
#     gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
#     gradient_direction = np.arctan2(gradient_y, gradient_x)
#     rows, cols = gradient_magnitude.shape
#     result = np.ones_like(gradient_magnitude)
#     print("Done 1")
#     angle_quantized = np.round(gradient_direction / (np.pi/4)) % 4

#     for i in range(1, rows-2):
#         if i % 400 == 0:
#             print(f"Done {i}")
#         for j in range(1, cols-2):
#             mag = gradient_magnitude[i, j]
#             direction = angle_quantized[i, j]

#             if direction == 0 and (mag >= gradient_magnitude[i, j-1]) and (mag >= gradient_magnitude[i, j+1]):
#                 result[i, j] = 0
#             elif direction == 1 and (mag >= gradient_magnitude[i-1, j+1]) and (mag >= gradient_magnitude[i+1, j-1]):
#                 result[i, j] = 0
#             elif direction == 2 and (mag >= gradient_magnitude[i-1, j]) and (mag >= gradient_magnitude[i+1, j]):
#                 result[i, j] = 0
#             elif direction == 3 and (mag >= gradient_magnitude[i-1, j-1]) and (mag >= gradient_magnitude[i+1, j+1]):
#                 result[i, j] = 0

#     return result

# def nms(gradient_magnitude, gradient_direction):
#     rows, cols = gradient_magnitude.shape
#     result = np.zeros_like(gradient_magnitude)
#     print("Done 1")
#     angle_quantized = np.round(gradient_direction / (np.pi/4)) % 4

#     for i in range(1, rows-2):
#         if i % 400 == 0:
#             print(f"Done {i}")
#         for j in range(1, cols-2):
#             mag = gradient_magnitude[i, j]
#             direction = angle_quantized[i, j]

#             if direction == 0 and (mag >= gradient_magnitude[i, j-1]) and (mag >= gradient_magnitude[i, j+1]):
#                 result[i, j] = mag
#             elif direction == 1 and (mag >= gradient_magnitude[i-1, j+1]) and (mag >= gradient_magnitude[i+1, j-1]):
#                 result[i, j] = mag
#             elif direction == 2 and (mag >= gradient_magnitude[i-1, j]) and (mag >= gradient_magnitude[i+1, j]):
#                 result[i, j] = mag
#             elif direction == 3 and (mag >= gradient_magnitude[i-1, j-1]) and (mag >= gradient_magnitude[i+1, j+1]):
#                 result[i, j] = mag

#     return result

