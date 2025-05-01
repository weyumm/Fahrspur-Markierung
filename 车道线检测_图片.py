import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置 matplotlib 支持中文和更大图像尺寸
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['figure.figsize'] = (18, 12)

# 1. 加载图像
img = cv2.imread('Euro Truck Simulator 2 2025-04-09 22-25-50 - frame at 1m7s.jpg')
if img is None:
    print("无法加载图片，请检查图片路径。")
else:
    # 转为 RGB 格式用于 matplotlib 显示
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. 定义透视变换区域（源点）
    src_pts = np.float32([
        [755, 500],   # 左上角
        [855, 500],   # 右上角
        [1150, 800],  # 右下角
        [420, 800]    # 左下角
    ])

    # 3. 目标区域的点（目标图尺寸：宽600，高1200）
    width, height = 600, 1200
    dst_pts = np.float32([
        [0, 0],             # 左上角
        [width - 1, 0],     # 右上角
        [width - 1, height - 1],  # 右下角
        [0, height - 1]     # 左下角
    ])

    # 4. 计算透视变换矩阵并做变换
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    bird_eye = cv2.warpPerspective(img, M, (width, height))
    bird_eye_rgb = cv2.cvtColor(bird_eye, cv2.COLOR_BGR2RGB)

    # 5. 在鸟瞰图上提取白色区域（可根据实际情况调整 HSV 阈值）
    hsv = cv2.cvtColor(bird_eye, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 140])
    upper_white = np.array([180, 60, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # 6. 定义滑动窗口检测车道像素点的函数（针对虚线，将阈值调整为30）
    def seekLane(binary_image, lane_peak_x, sliding_window_num=9, margin=100, min_pixels_threshold=30):
        """
        输入：
            binary_image: 二值化图像（例如白色区域掩码）
            lane_peak_x: 车道起始的峰值 x 坐标（直方图得到）
            sliding_window_num: 将图像划分的窗口数量（自下而上）
            margin: 窗口宽度一半
            min_pixels_threshold: 窗口内重新定位中心的最少像素数
        输出：
            lane_x, lane_y: 车道线上所有白色像素的 x 和 y 坐标
        """
        h, w = binary_image.shape
        nonzero = binary_image.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        
        window_center = lane_peak_x
        sliding_window_height = h // sliding_window_num
        lane_pixel_indexes = []
        
        for i in range(sliding_window_num):
            window_top = h - (i + 1) * sliding_window_height
            window_bottom = h - i * sliding_window_height
            window_left = int(window_center - margin)
            window_right = int(window_center + margin)
            
            window_pixel_indices = ((nonzero_y >= window_top) & (nonzero_y < window_bottom) &
                                    (nonzero_x >= window_left) & (nonzero_x < window_right)).nonzero()[0]
            lane_pixel_indexes.append(window_pixel_indices)
            
            # 如果当前窗口内像素数大于设定阈值，则更新窗口中心（取质心）
            if len(window_pixel_indices) > min_pixels_threshold:
                window_center = int(np.mean(nonzero_x[window_pixel_indices]))
        
        if len(lane_pixel_indexes) > 0:
            lane_pixel_indexes = np.concatenate(lane_pixel_indexes)
        else:
            lane_pixel_indexes = np.array([])
        
        lane_x = nonzero_x[lane_pixel_indexes]
        lane_y = nonzero_y[lane_pixel_indexes]
        
        return lane_x, lane_y

    # 7. 利用直方图确定左右车道线的初始峰值
    histogram = np.sum(white_mask[white_mask.shape[0] // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    left_lane_peak = np.argmax(histogram[:midpoint])
    right_lane_peak = np.argmax(histogram[midpoint:]) + midpoint

    # 分别检测左右车道线像素
    left_lane_x, left_lane_y = seekLane(white_mask, left_lane_peak, sliding_window_num=9, margin=100, min_pixels_threshold=30)
    right_lane_x, right_lane_y = seekLane(white_mask, right_lane_peak, sliding_window_num=9, margin=100, min_pixels_threshold=30)

    # 8. 对左右车道线分别进行二次多项式拟合（以 y 作为自变量，x 为因变量）
    plot_y = np.linspace(0, height - 1, height)
    left_fit_x, right_fit_x = None, None
    if len(left_lane_y) > 0 and len(left_lane_x) > 0:
        left_fit = np.polyfit(left_lane_y, left_lane_x, 2)
        left_fit_poly = np.poly1d(left_fit)
        left_fit_x = left_fit_poly(plot_y)
    if len(right_lane_y) > 0 and len(right_lane_x) > 0:
        right_fit = np.polyfit(right_lane_y, right_lane_x, 2)
        right_fit_poly = np.poly1d(right_fit)
        right_fit_x = right_fit_poly(plot_y)

    # 9. 将6张图展示在一个2x3的图中
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 子图1：原图
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("原图")
    axes[0, 0].axis('off')

    # 子图2：鸟瞰图
    axes[0, 1].imshow(bird_eye_rgb)
    axes[0, 1].set_title("鸟瞰图")
    axes[0, 1].axis('off')

    # 子图3：白色区域掩码
    axes[0, 2].imshow(white_mask, cmap='gray')
    axes[0, 2].set_title("白色区域掩码")
    axes[0, 2].axis('off')

    # 子图4：横向白色像素累计直方图
    axes[1, 0].plot(histogram, color='black')
    axes[1, 0].set_title("横向白色像素累计直方图")
    axes[1, 0].set_xlabel("列坐标（x轴）")
    axes[1, 0].set_ylabel("像素数量")
    axes[1, 0].grid(True)

    # 子图5：车道像素点检测（左右）
    axes[1, 1].imshow(bird_eye_rgb)
    axes[1, 1].scatter(left_lane_x, left_lane_y, s=10, color='yellow', label='左侧车道')
    axes[1, 1].scatter(right_lane_x, right_lane_y, s=10, color='cyan', label='右侧车道')
    axes[1, 1].set_title("车道像素点检测")
    axes[1, 1].legend()
    axes[1, 1].axis('off')

    # 子图6：车道拟合结果
    axes[1, 2].imshow(bird_eye_rgb)
    if left_fit_x is not None:
        axes[1, 2].plot(left_fit_x, plot_y, color='red', linewidth=2, label='左侧拟合曲线')
    if right_fit_x is not None:
        axes[1, 2].plot(right_fit_x, plot_y, color='blue', linewidth=2, label='右侧拟合曲线')
    axes[1, 2].set_title("车道拟合结果")
    axes[1, 2].legend()
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()
