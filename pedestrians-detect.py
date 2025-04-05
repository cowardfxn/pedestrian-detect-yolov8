# -*- coding: utf-8 -*-
import cv2
import torch
import json
import datetime
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class TimeWindowCounter:
    def __init__(self, interval_seconds=60):
        self.interval = datetime.timedelta(seconds=interval_seconds)
        self.counted_ids = set()
        self.time_windows = defaultdict(set)
        self.first_detection_time = {}

    def update(self, track_id, current_time):
        if track_id not in self.first_detection_time:
            self.first_detection_time[track_id] = current_time
            window_start = current_time.replace(
                minute=(current_time.minute // self.interval.seconds // 60)
                * self.interval.seconds
                // 60,
                second=0,
                microsecond=0,
            )
            self.time_windows[window_start].add(track_id)
            self.counted_ids.add(track_id)

    def get_stats(self):
        return {
            "total": len(self.counted_ids),
            "time_windows": {
                window.isoformat(): len(ids)
                for window, ids in sorted(self.time_windows.items())
            },
        }


class PeopleCounter:
    """
    默认使用最新YOLOv8 Nano模型
    """

    def __init__(self, model_path="./yolov8/yolov8n.pt"):
        # 初始化模型
        self.detector = YOLO(model_path)
        self.tracker = DeepSort(
            # 目标丢失后保持跟踪的最大帧数
            max_age=30,
            # 新目标确认所需的连续检测次数
            n_init=3,
            # 轻量化ReID模型
            embedder="mobilenet",
            # 特征匹配阈值
            max_cosine_distance=0.4,
        )
        # 已计数的行人ID集合
        self.counted_ids = set()
        # 总人数统计
        self.total_count = 0

    def process_frame(self, frame):
        # YOLOv8检测行人（仅person类）
        results = self.detector.predict(
            source=frame,
            # 0对应COCO的person类
            classes=[0],
            # 置信度阈值
            conf=0.5,
            imgsz=640,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # 转换检测结果为DeepSORT格式
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                detections.append((box, conf, None))  # (xyxy, conf, cls)

        # DeepSORT跟踪
        tracked_objects = self.tracker.update_tracks(detections, frame=frame)

        # 绘制结果并计数
        for obj in tracked_objects:
            if not obj.is_confirmed():
                continue
            track_id = obj.track_id
            bbox = obj.to_ltrb(orig=True)  # 获取边界框坐标

            # 绘制边界框和ID
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID:{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )

            # 去重计数逻辑
            if track_id not in self.counted_ids:
                self.total_count += 1
                self.counted_ids.add(track_id)

        # 显示统计信息
        cv2.putText(
            frame,
            f"Total: {self.total_count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3,
        )
        return frame


class PedestrianCounter:
    """
    默认使用最新YOLOv8 Nano模型
    """

    def __init__(self, model_path="./yolov8/yolov8n.pt", time_interval=60):
        # 初始化模型
        self.detector = YOLO(model_path)
        self.tracker = DeepSort(
            # 目标丢失后保持跟踪的最大帧数
            max_age=30,
            # 新目标确认所需的连续检测次数
            n_init=3,
            # 轻量化ReID模型
            embedder="mobilenet",
            # 特征匹配阈值
            max_cosine_distance=0.4,
        )
        # 已计数的行人ID集合
        self.counted_ids = set()
        # 总人数统计
        self.total_count = 0

        self.counter = TimeWindowCounter(time_interval)
        self.start_time = datetime.datetime.now()
        self.interval = datetime.timedelta(seconds=time_interval)
        self.time_windows = defaultdict(set)
        self.first_detection = {}

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 目标检测与跟踪
            results = self.detector.predict(frame, classes=[0], conf=0.5)
            detections = [(box.xyxy[0], box.conf[0], None) for box in results[0].boxes]
            tracked_objs = self.tracker.update_tracks(detections, frame=frame)

            current_time = datetime.datetime.now()
            # 更新统计信息
            for obj in tracked_objs:
                if obj.is_confirmed():
                    self._update_counter(obj.track_id, current_time)

        return self._generate_report(duration)

    def _update_counter(self, track_id, timestamp):
        """核心统计逻辑"""
        if track_id not in self.first_detection:
            # 新目标首次出现时间记录
            self.first_detection[track_id] = timestamp
            self.counted_ids.add(track_id)

            # 计算所属时间窗口
            window_start = timestamp.replace(
                second=0,
                microsecond=0,
                minute=(timestamp.minute // self.interval.seconds // 60)
                * self.interval.seconds
                // 60,
            )
            self.time_windows[window_start].add(track_id)

    def _generate_report(self, total_duration):
        """生成结构化统计报告"""
        time_stats = {
            win.isoformat(): len(ids) for win, ids in sorted(self.time_windows.items())
        }

        return json.dumps(
            {
                "total_unique_pedestrians": len(self.counted_ids),
                "time_window_statistics": time_stats,
                "metadata": {
                    "processing_time": datetime.datetime.now().isoformat(),
                    "video_duration": total_duration,
                    "detection_model": "yolov8n",
                },
            },
            indent=2,
        )


def mark_pedestrians(video_path):
    """标记视频中的行人并显示"""
    counter = PeopleCounter("./yolov8/yolov8l.pt")
    cap = cv2.VideoCapture(video_path)

    # 视频输出设置
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = counter.process_frame(frame)
        out.write(processed_frame)
        cv2.imshow("People Counter", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Total people counted: {counter.total_count}")


def count_peds(video_path):
    """计数行人并返回JSON结果"""
    counter = PedestrianCounter("./yolov8/yolov8l.pt", time_interval=5)
    result_json = counter.process_video("input.mp4")
    print(result_json)


if __name__ == "__main__":
    # 支持本地文件/摄像头/RTSP流
    video_path = "input2.mp4"

    # count_peds(video_path)

    mark_pedestrians(video_path)
