import cv2
import numpy as np
import mss
import time
from ultralytics import YOLO


class KalmanTracker:
    """
    State estimator based on a Kalman filter.
    It smooths noisy observations and predicts target motion.
    """

    def __init__(self):
        # State vector: (x, y, vx, vy), Measurement: (x, y)
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], np.float32)

        # Noise covariance matrices (tuned empirically)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def update(self, x, y):
        """Update the filter with a new measurement."""
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)

    def predict(self):
        """Predict the next state."""
        prediction = self.kf.predict()
        return int(prediction[0][0]), int(prediction[1][0])


class PerceptionEngine:
    """
    Perception module integrating detection, filtering, and tracking.
    """

    def __init__(self, model_path):
        # Load TensorRT engine for accelerated inference
        self.model = YOLO(model_path, task='detect')
        self.tracker = KalmanTracker()

        # Hyperparameters
        self.conf_threshold = 0.45
        self.roi_size = 640
        self.last_target_pos = None
        self.lost_frames = 0
        self.max_lost_frames = 20

    def _feature_verification(self, roi):
        """
        Secondary feature verification using HSV color filtering.
        This reduces false positives from the detector.
        """
        if roi is None or roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Example: filter specific color range (e.g., UI element or label)
        mask = cv2.inRange(
            hsv,
            np.array([35, 43, 46]),
            np.array([85, 255, 255])
        )

        return cv2.countNonZero(mask) > 10

    def process_frame(self, frame, offset_x=0, offset_y=0):
        """
        Core perception pipeline:
        detection → verification → tracking → prediction
        """
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            device=0,
            half=True,
            verbose=False
        )

        detected_pos = None

        if results and len(results[0].boxes) > 0:
            # Select the highest-confidence bounding box
            box = results[0].boxes[0]
            c = box.xywh[0].cpu().numpy()
            b = box.xyxy[0].cpu().numpy().astype(int)

            # Crop region for secondary verification
            crop = frame[max(0, b[1]):b[3], max(0, b[0]):b[2]]

            if self._feature_verification(crop):
                # Map ROI coordinates to global screen coordinates
                detected_pos = (int(c[0]) + offset_x, int(c[1]) + offset_y)

                self.tracker.update(detected_pos[0], detected_pos[1])
                self.lost_frames = 0
            else:
                self.lost_frames += 1
        else:
            self.lost_frames += 1

        # Prediction when detection is temporarily lost
        if self.lost_frames <= self.max_lost_frames:
            self.last_target_pos = self.tracker.predict()
        else:
            self.last_target_pos = None

        return detected_pos, self.last_target_pos


def main():
    """
    Real-time perception loop with screen capture and visualization.
    """

    engine = PerceptionEngine(r'D:\04 Project\Object-Tracking-System\weights\best.engine')

    sct = mss.mss()
    monitor = sct.monitors[1]

    print("Perception Engine Started. Press 'q' to quit.")

    while True:
        start_time = time.time()

        # 1. Frame acquisition
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 2. Inference and tracking
        raw_pos, pred_pos = engine.process_frame(frame)

        # 3. Visualization
        if pred_pos:
            cv2.circle(frame, pred_pos, 10, (255, 0, 255), -1)
            cv2.line(frame, (pred_pos[0] - 20, pred_pos[1]),
                     (pred_pos[0] + 20, pred_pos[1]), (0, 255, 0), 2)
            cv2.line(frame, (pred_pos[0], pred_pos[1] - 20),
                     (pred_pos[0], pred_pos[1] + 20), (0, 255, 0), 2)

        # FPS display
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('AI Perception Debugger', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()