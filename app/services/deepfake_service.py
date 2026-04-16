from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class FaceResult:
    x1: int
    y1: int
    x2: int
    y2: int
    det_conf: float
    label: int
    confidence: float
    prob_real: float
    prob_fake: float


@dataclass
class AnalyzeResult:
    input_type: str
    faces_count: int
    verdict: str
    summary_text: str
    output_path: Path | None


@dataclass
class VideoFacePrediction:
    x1: int
    y1: int
    x2: int
    y2: int
    det_conf: float
    label: int
    confidence: float
    prob_real: float
    prob_fake: float


@dataclass
class VideoFrameAnalysis:
    frame_idx: int
    timestamp_sec: float
    predictions: list[VideoFacePrediction]
    frame_fake_score: float


@dataclass
class VideoAggregate:
    verdict: str
    overall_fake_score: float
    mean_fake_score: float
    peak_fake_score: float
    fake_frame_ratio: float
    suspicious_streak_sec: float
    fake_frames: int
    real_frames: int


def generate_anchors(input_size: int = 256) -> np.ndarray:
    strides = [16, 32]
    anchors_per_cell = [2, 6]
    anchors = []

    for stride, num_anchors in zip(strides, anchors_per_cell):
        grid_size = input_size // stride
        for y in range(grid_size):
            for x in range(grid_size):
                cx = (x + 0.5) / grid_size
                cy = (y + 0.5) / grid_size
                for _ in range(num_anchors):
                    anchors.append([cx, cy])

    return np.array(anchors, dtype=np.float32)


class FaceDetector:
    def __init__(self, model_path: str | os.PathLike, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = int(input_shape[2]) if input_shape[2] else 256
        self.anchors = generate_anchors(self.input_size)

    def preprocess(self, image: np.ndarray):
        h, w = image.shape[:2]
        scale = self.input_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))
        padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)

        pad_y = (self.input_size - new_h) // 2
        pad_x = (self.input_size - new_w) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        blob = padded.astype(np.float32) / 127.5 - 1.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        return blob, h, w, scale, pad_x, pad_y

    def decode_boxes(self, raw_boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        boxes = np.zeros_like(raw_boxes[:, :4])
        boxes[:, 0] = raw_boxes[:, 0] / self.input_size + anchors[:, 0]
        boxes[:, 1] = raw_boxes[:, 1] / self.input_size + anchors[:, 1]
        boxes[:, 2] = raw_boxes[:, 2] / self.input_size
        boxes[:, 3] = raw_boxes[:, 3] / self.input_size

        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        return np.stack([x1, y1, x2, y2], axis=1)

    def nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.3) -> list[int]:
        if len(boxes) == 0:
            return []

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self, image: np.ndarray, box_scale: float = 2.0) -> list[list[float]]:
        h, w = image.shape[:2]
        input_tensor, _, _, scale, pad_x, pad_y = self.preprocess(image)

        outputs = self.session.run(None, {self.input_name: input_tensor})

        raw_boxes = np.concatenate([outputs[0][0], outputs[1][0]], axis=0)
        raw_scores = np.concatenate([outputs[2][0], outputs[3][0]], axis=0)

        raw_flat = np.clip(raw_scores[:, 0], -50, 50)
        scores = 1.0 / (1.0 + np.exp(-raw_flat))
        boxes = self.decode_boxes(raw_boxes, self.anchors)

        mask = scores >= self.confidence_threshold
        if not np.any(mask):
            return []

        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]

        keep = self.nms(filtered_boxes, filtered_scores)
        filtered_boxes = filtered_boxes[keep]
        filtered_scores = np.array(filtered_scores)[keep]

        result = []

        for i in range(len(filtered_boxes)):
            box = filtered_boxes[i].copy()

            box[0] = (box[0] * self.input_size - pad_x) / scale
            box[1] = (box[1] * self.input_size - pad_y) / scale
            box[2] = (box[2] * self.input_size - pad_x) / scale
            box[3] = (box[3] * self.input_size - pad_y) / scale

            if box_scale != 1.0:
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                bw = (box[2] - box[0]) * box_scale
                bh = (box[3] - box[1]) * box_scale
                box[0] = cx - bw / 2
                box[1] = cy - bh / 2
                box[2] = cx + bw / 2
                box[3] = cy + bh / 2

            x1 = int(np.clip(box[0], 0, w))
            y1 = int(np.clip(box[1], 0, h))
            x2 = int(np.clip(box[2], 0, w))
            y2 = int(np.clip(box[3], 0, h))

            if x2 > x1 and y2 > y1:
                result.append([x1, y1, x2, y2, float(filtered_scores[i])])

        return result


class DeepfakeClassifier:
    def __init__(self, model_path: str | os.PathLike, data_path: str | os.PathLike | None = None):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD
        self.img_size = 256

        if data_path and os.path.exists(data_path):
            with open(data_path, "rb") as f:
                data = f.read()
                if len(data) >= 28:
                    self.mean = np.frombuffer(data[0:12], dtype=np.float32)
                    self.std = np.frombuffer(data[12:24], dtype=np.float32)
                    self.img_size = int(np.frombuffer(data[24:28], dtype=np.int32)[0])

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        h, w = face.shape[:2]
        scale = self.img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        face = cv2.resize(face, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2

        face = cv2.copyMakeBorder(
            face,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        face = face.astype(np.float32) / 255.0
        face = (face - self.mean) / self.std
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)
        return face

    def classify(self, face_bgr: np.ndarray):
        input_tensor = self.preprocess(face_bgr)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        logits = outputs[0][0]

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        weights = np.array([0.7, 0.3], dtype=np.float32)
        weighted = probs * weights
        weighted = weighted / np.sum(weighted)

        prob_real = float(weighted[0] * 100)
        prob_fake = float(weighted[1] * 100)
        label = int(np.argmax(weighted))
        confidence = float(np.max(weighted) * 100)

        return label, confidence, prob_real, prob_fake


class DeepfakeService:
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

    def __init__(
        self,
        face_model_path: Path,
        deepfake_model_path: Path,
        deepfake_data_path: Path | None,
        results_dir: Path,
        frame_skip: int = 5,
    ) -> None:
        if not face_model_path.exists():
            raise FileNotFoundError(f"Face model not found: {face_model_path}")

        if not deepfake_model_path.exists():
            raise FileNotFoundError(f"Deepfake model not found: {deepfake_model_path}")

        self.face_detector = FaceDetector(face_model_path)
        self.deepfake_classifier = DeepfakeClassifier(deepfake_model_path, deepfake_data_path)
        self.results_dir = results_dir
        self.frame_skip = frame_skip

    def analyze(self, input_path: Path) -> AnalyzeResult:
        ext = input_path.suffix.lower()

        if ext in self.IMAGE_EXTS:
            return self._analyze_image(input_path)
        if ext in self.VIDEO_EXTS:
            return self._analyze_video(input_path)

        raise ValueError(f"Неподдерживаемый формат файла: {ext}")

    def _analyze_image(self, image_path: Path) -> AnalyzeResult:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Не удалось открыть изображение")

        faces = self.face_detector.detect(image)

        if not faces:
            return AnalyzeResult(
                input_type="image",
                faces_count=0,
                verdict="no_faces",
                summary_text="Лица не найдены.",
                output_path=None,
            )

        results: list[FaceResult] = []

        for face in faces:
            x1, y1, x2, y2, det_conf = face
            face_crop = image[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            label, confidence, prob_real, prob_fake = self.deepfake_classifier.classify(face_crop)

            results.append(
                FaceResult(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    det_conf=det_conf,
                    label=label,
                    confidence=confidence,
                    prob_real=prob_real,
                    prob_fake=prob_fake,
                )
            )

        for item in results:
            color = (0, 255, 0) if item.label == 0 else (0, 0, 255)
            label_text = "REAL" if item.label == 0 else "FAKE"

            cv2.rectangle(image, (item.x1, item.y1), (item.x2, item.y2), color, 2)
            text = f"{label_text} {item.confidence:.1f}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top_y = max(item.y1 - th - 10, 0)
            cv2.rectangle(image, (item.x1, top_y), (item.x1 + tw, item.y1), color, -1)
            cv2.putText(
                image,
                text,
                (item.x1, max(item.y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        output_path = self.results_dir / f"{image_path.stem}_result.jpg"
        cv2.imwrite(str(output_path), image)

        fake_count = sum(1 for r in results if r.label == 1)
        real_count = sum(1 for r in results if r.label == 0)
        verdict = "fake" if fake_count > real_count else "real"

        lines = [
            f"Найдено лиц: {len(results)}",
            f"Real: {real_count}",
            f"Fake: {fake_count}",
            "",
        ]

        for idx, r in enumerate(results, start=1):
            label_text = "FAKE" if r.label == 1 else "REAL"
            lines.append(
                f"Лицо #{idx}: {label_text} | confidence={r.confidence:.1f}% | "
                f"real={r.prob_real:.1f}% | fake={r.prob_fake:.1f}%"
            )

        return AnalyzeResult(
            input_type="image",
            faces_count=len(results),
            verdict=verdict,
            summary_text="\n".join(lines),
            output_path=output_path,
        )

    def _analyze_video(self, video_path: Path) -> AnalyzeResult:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Не удалось открыть видео")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = self.results_dir / f"{video_path.stem}_result.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        start_time = time.time()

        analyzed_frames = self._extract_video_frames(cap, fps)
        frame_map = {analysis.frame_idx: analysis for analysis in analyzed_frames}

        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            writer.release()
            raise ValueError("Не удалось повторно открыть видео для сборки результата")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in frame_map:
                self._annotate_video_frame(frame, frame_map[frame_idx])

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        frames_with_faces = [frame for frame in analyzed_frames if frame.predictions]
        if not frames_with_faces:
            return AnalyzeResult(
                input_type="video",
                faces_count=0,
                verdict="no_faces",
                summary_text="На видео лица не найдены.",
                output_path=output_path,
            )

        aggregate = self._aggregate_video_predictions(frames_with_faces, fps)
        all_predictions = [prediction for frame in frames_with_faces for prediction in frame.predictions]
        avg_real = float(np.mean([prediction.prob_real for prediction in all_predictions]))
        avg_fake = float(np.mean([prediction.prob_fake for prediction in all_predictions]))
        elapsed = time.time() - start_time

        suspicious_frames = sorted(
            frames_with_faces,
            key=lambda frame: frame.frame_fake_score,
            reverse=True,
        )[:3]

        summary_lines = [
            f"Видео обработано за {elapsed:.1f} сек.",
            f"Кадров всего: {total_frames}",
            f"Кадров проанализировано: {len(analyzed_frames)} (каждый {self.frame_skip}-й)",
            f"Кадров с лицами: {len(frames_with_faces)}",
            f"Обнаружений лиц: {len(all_predictions)}",
            f"Итоговый fake score: {aggregate.overall_fake_score * 100:.1f}%",
            f"Средний fake score по кадрам: {aggregate.mean_fake_score * 100:.1f}%",
            f"Пиковый fake score кадра: {aggregate.peak_fake_score * 100:.1f}%",
            f"Подозрительных кадров: {aggregate.fake_frames}",
            f"Стабильно подозрительный фрагмент: {aggregate.suspicious_streak_sec:.2f} сек.",
            f"Средний real: {avg_real:.1f}%",
            f"Средний fake: {avg_fake:.1f}%",
        ]

        if suspicious_frames:
            summary_lines.append("")
            summary_lines.append("Самые подозрительные кадры:")
            for frame in suspicious_frames:
                summary_lines.append(
                    f"- кадр {frame.frame_idx} ({frame.timestamp_sec:.2f} сек): "
                    f"fake score {frame.frame_fake_score * 100:.1f}% | лиц {len(frame.predictions)}"
                )

        return AnalyzeResult(
            input_type="video",
            faces_count=len(all_predictions),
            verdict=aggregate.verdict,
            summary_text="\n".join(summary_lines),
            output_path=output_path,
        )

    def _extract_video_frames(self, cap: cv2.VideoCapture, fps: float) -> list[VideoFrameAnalysis]:
        analyzed_frames: list[VideoFrameAnalysis] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_skip == 0:
                analyzed_frames.append(self._analyze_video_frame(frame, frame_idx, fps))

            frame_idx += 1

        return analyzed_frames

    def _analyze_video_frame(self, frame: np.ndarray, frame_idx: int, fps: float) -> VideoFrameAnalysis:
        faces = self.face_detector.detect(frame)
        predictions: list[VideoFacePrediction] = []

        for face in faces:
            x1, y1, x2, y2, det_conf = face
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            label, confidence, prob_real, prob_fake = self.deepfake_classifier.classify(face_crop)
            predictions.append(
                VideoFacePrediction(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    det_conf=det_conf,
                    label=label,
                    confidence=confidence,
                    prob_real=prob_real,
                    prob_fake=prob_fake,
                )
            )

        frame_fake_score = self._score_video_frame(predictions)
        return VideoFrameAnalysis(
            frame_idx=frame_idx,
            timestamp_sec=frame_idx / fps if fps > 0 else 0.0,
            predictions=predictions,
            frame_fake_score=frame_fake_score,
        )

    def _score_video_frame(self, predictions: list[VideoFacePrediction]) -> float:
        if not predictions:
            return 0.0

        weighted_scores = []
        for prediction in predictions:
            model_conf = prediction.confidence / 100.0
            fake_prob = prediction.prob_fake / 100.0
            weighted_scores.append(fake_prob * (0.55 + 0.45 * model_conf) * prediction.det_conf)

        peak_score = max(weighted_scores)
        mean_score = float(np.mean(weighted_scores))
        return float(np.clip(0.65 * peak_score + 0.35 * mean_score, 0.0, 1.0))

    def _aggregate_video_predictions(
        self,
        frames_with_faces: list[VideoFrameAnalysis],
        fps: float,
    ) -> VideoAggregate:
        frame_scores = [frame.frame_fake_score for frame in frames_with_faces]
        mean_fake_score = float(np.mean(frame_scores))
        peak_fake_score = float(np.max(frame_scores))
        suspicious_threshold = 0.55

        fake_frames = sum(1 for score in frame_scores if score >= suspicious_threshold)
        real_frames = len(frame_scores) - fake_frames
        fake_frame_ratio = fake_frames / len(frame_scores)
        suspicious_streak = self._max_suspicious_streak(frames_with_faces, suspicious_threshold, fps)

        top_count = max(1, int(np.ceil(len(frame_scores) * 0.3)))
        top_mean = float(np.mean(sorted(frame_scores, reverse=True)[:top_count]))
        streak_score = min(suspicious_streak / 1.5, 1.0)

        overall_fake_score = float(
            np.clip(
                0.40 * mean_fake_score
                + 0.30 * top_mean
                + 0.20 * fake_frame_ratio
                + 0.10 * streak_score,
                0.0,
                1.0,
            )
        )

        strong_peak = peak_fake_score >= 0.85 and fake_frame_ratio >= 0.20
        consistent_fake = overall_fake_score >= 0.58 or (overall_fake_score >= 0.52 and suspicious_streak >= 1.0)
        verdict = "fake" if strong_peak or consistent_fake else "real"

        return VideoAggregate(
            verdict=verdict,
            overall_fake_score=overall_fake_score,
            mean_fake_score=mean_fake_score,
            peak_fake_score=peak_fake_score,
            fake_frame_ratio=fake_frame_ratio,
            suspicious_streak_sec=suspicious_streak,
            fake_frames=fake_frames,
            real_frames=real_frames,
        )

    def _max_suspicious_streak(
        self,
        frames_with_faces: list[VideoFrameAnalysis],
        suspicious_threshold: float,
        fps: float,
    ) -> float:
        if not frames_with_faces or fps <= 0:
            return 0.0

        longest = 0
        current = 0
        expected_step = max(1, self.frame_skip)
        previous_idx: int | None = None

        for frame in frames_with_faces:
            is_consecutive = previous_idx is not None and frame.frame_idx - previous_idx == expected_step
            if frame.frame_fake_score >= suspicious_threshold:
                current = current + 1 if is_consecutive else 1
                longest = max(longest, current)
            else:
                current = 0

            previous_idx = frame.frame_idx

        return longest * expected_step / fps

    def _annotate_video_frame(self, frame: np.ndarray, analysis: VideoFrameAnalysis) -> None:
        for prediction in analysis.predictions:
            color = (0, 255, 0) if prediction.label == 0 else (0, 0, 255)
            label_text = "REAL" if prediction.label == 0 else "FAKE"

            cv2.rectangle(frame, (prediction.x1, prediction.y1), (prediction.x2, prediction.y2), color, 2)
            text = f"{label_text} {prediction.confidence:.1f}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top_y = max(prediction.y1 - th - 10, 0)
            cv2.rectangle(frame, (prediction.x1, top_y), (prediction.x1 + tw, prediction.y1), color, -1)
            cv2.putText(
                frame,
                text,
                (prediction.x1, max(prediction.y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        banner_text = f"Fake score: {analysis.frame_fake_score * 100:.1f}%"
        cv2.rectangle(frame, (10, 10), (310, 50), (20, 20, 20), -1)
        cv2.putText(
            frame,
            banner_text,
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
