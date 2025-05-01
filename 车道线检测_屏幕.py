import cv2
import numpy as np
import time

class LaneDetector:
    def __init__(self, car_marker_path=None):
        """
        初始化车道检测器
        :param car_marker_path: 车辆标记图片路径（可选，用于后续扩展）
        """
        self.car_marker_path = car_marker_path
        if car_marker_path is not None:
            self.car_marker = cv2.imread(car_marker_path)
        else:
            self.car_marker = None

        # 定义透视变换参数（根据实际相机标定设定）
        # 源点：车道区域在原图中的4个顶点
        self.src_pts = np.float32([
            [755, 500],  # 左上角
            [835, 500],  # 右上角
            [1150, 800], # 右下角
            [420, 800]   # 左下角
        ])
        # 目标图尺寸：此处设定为宽600，高1200，可根据需要调整
        self.dst_width = 600
        self.dst_height = 1200
        self.dst_pts = np.float32([
            [0, 0],
            [self.dst_width - 1, 0],
            [self.dst_width - 1, self.dst_height - 1],
            [0, self.dst_height - 1]
        ])
        # 计算透视变换矩阵
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)

        # 定义像素到真实世界的转换因子（单位：米/像素），需根据实际相机标定调整
        self.ym_per_pix = 30.0 / 720.0  # y方向转换因子
        self.xm_per_pix = 3.7 / 700.0   # x方向转换因子

    def process_frame(self, frame):
        """
        对单帧图像进行车道检测，计算车道曲率和车辆偏移
        :param frame: 原始BGR图像
        :return: 输出图像（绘制了车道拟合曲线及文字信息），以及车辆横向偏移值（单位：米）
        """
        # 1. 透视变换获得鸟瞰图
        bird_eye = cv2.warpPerspective(frame, self.M, (self.dst_width, self.dst_height))
        
        # 2. HSV颜色空间提取白色区域（阈值可调，根据道路情况）
        hsv = cv2.cvtColor(bird_eye, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 90])
        upper_white = np.array([180, 60, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # 3. 利用下半部分直方图确定左右车道初始位置
        histogram = np.sum(white_mask[self.dst_height//2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        left_lane_peak = np.argmax(histogram[:midpoint])
        right_lane_peak = np.argmax(histogram[midpoint:]) + midpoint

        # 4. 利用滑动窗口方法检测左右车道像素点
        left_lane_x, left_lane_y = self.seekLane(white_mask, left_lane_peak)
        right_lane_x, right_lane_y = self.seekLane(white_mask, right_lane_peak)
        
        # 5. 对左右车道像素点做二次多项式拟合（以y为自变量，x为因变量）
        plot_y = np.linspace(0, self.dst_height - 1, self.dst_height)
        left_fit, right_fit = None, None
        left_fit_x, right_fit_x = None, None
        if len(left_lane_y) > 0 and len(left_lane_x) > 0:
            left_fit = np.polyfit(left_lane_y, left_lane_x, 2)
            left_fit_poly = np.poly1d(left_fit)
            left_fit_x = left_fit_poly(plot_y)
        if len(right_lane_y) > 0 and len(right_lane_x) > 0:
            right_fit = np.polyfit(right_lane_y, right_lane_x, 2)
            right_fit_poly = np.poly1d(right_fit)
            right_fit_x = right_fit_poly(plot_y)
        
        # 6. 计算车道曲率（转换到真实世界坐标计算）
        left_curverad, right_curverad = None, None
        y_eval = self.dst_height * self.ym_per_pix  # 取图像底部对应真实世界的y值
        if left_fit is not None:
            left_fit_cr = np.polyfit(left_lane_y * self.ym_per_pix, left_lane_x * self.xm_per_pix, 2)
            left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1])**2)**1.5) / np.abs(2 * left_fit_cr[0])
        if right_fit is not None:
            right_fit_cr = np.polyfit(right_lane_y * self.ym_per_pix, right_lane_x * self.xm_per_pix, 2)
            right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1])**2)**1.5) / np.abs(2 * right_fit_cr[0])
        
        # 7. 计算车辆偏移量（假设车辆位于鸟瞰图的中心位置）
        vehicle_offset = None
        if left_fit is not None and right_fit is not None:
            left_bottom_x = np.poly1d(left_fit)(self.dst_height - 1)
            right_bottom_x = np.poly1d(right_fit)(self.dst_height - 1)
            lane_center = (left_bottom_x + right_bottom_x) / 2.0
            car_center = self.dst_width / 2.0
            vehicle_offset = (car_center - lane_center) * self.xm_per_pix
        
        # 8. 在鸟瞰图上绘制检测结果
        out_img = bird_eye.copy()
        # 绘制拟合曲线
        if left_fit_x is not None:
            pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))]).astype(np.int32)
            cv2.polylines(out_img, [pts_left], isClosed=False, color=(0, 0, 255), thickness=5)
        if right_fit_x is not None:
            pts_right = np.array([np.transpose(np.vstack([right_fit_x, plot_y]))]).astype(np.int32)
            cv2.polylines(out_img, [pts_right], isClosed=False, color=(255, 0, 0), thickness=5)
        
        # 在输出图上添加曲率和车辆偏移信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        if left_curverad is not None:
            cv2.putText(out_img, "Left Curvature: {:.2f} m".format(left_curverad),
                        (30, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if right_curverad is not None:
            cv2.putText(out_img, "Right Curvature: {:.2f} m".format(right_curverad),
                        (30, 60), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if vehicle_offset is not None:
            direction = "left" if vehicle_offset < 0 else "right"
            cv2.putText(out_img, "Vehicle Offset: {:.2f} m ({})".format(np.abs(vehicle_offset), direction),
                        (30, 90), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        return out_img, vehicle_offset

    def seekLane(self, binary_image, lane_peak_x, sliding_window_num=9, margin=100, min_pixels_threshold=30):
        """
        滑动窗口检测车道像素点，返回该车道所有白色像素的x和y坐标
        :param binary_image: 二值化图像（如白色区域掩码）
        :param lane_peak_x: 初始车道x方向峰值坐标
        :param sliding_window_num: 分段窗口数
        :param margin: 窗口半宽
        :param min_pixels_threshold: 每窗口更新中心所需最小像素数
        :return: lane_x, lane_y
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
            if len(window_pixel_indices) > min_pixels_threshold:
                window_center = int(np.mean(nonzero_x[window_pixel_indices]))
        if len(lane_pixel_indexes) > 0:
            lane_pixel_indexes = np.concatenate(lane_pixel_indexes)
        else:
            lane_pixel_indexes = np.array([])
        lane_x = nonzero_x[lane_pixel_indexes]
        lane_y = nonzero_y[lane_pixel_indexes]
        return lane_x, lane_y


def main():
    import cv2
    import time

    video_path = r"D:\同济汽院大二下课程资料\水课\6，汽车竞赛\1，车道线检测代码\Euro Truck Simulator 2 2025-04-11 22-10-29.mp4"
    lane_detector = LaneDetector(car_marker_path='car.jpg')
    window_name = "Result"
    cv2.namedWindow(window_name)
    detection_on = True  # 检测开关：True 启动，False 停止

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if detection_on:
            output_image, offset = lane_detector.process_frame(frame)
            print("Offset: {:.6f}".format(offset))
        else:
            output_image = frame

        cv2.imshow(window_name, output_image)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('z'):
            detection_on = not detection_on

        # 保持大约30 FPS
        time.sleep(1 / 30)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
