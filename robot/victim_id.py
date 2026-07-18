"""Victim perception, tracking, search actions, reporting, and debug display.

This module emits MissionAction requests for the mission runtime. It does not
write wheel speeds directly.
"""

from dataclasses import dataclass, field
from pathlib import Path
import atexit
import json
import math
import os
import subprocess
import sys
import threading
import time

import numpy as np

from config import LIVE_MAP_VIEWER_PATH, _env_float, _env_int, _env_flag
from parameters import LocalObstacleProjection as LocalObstacleProjectionParams
from parameters import Victim as VictimParams


def target_record_unavailable_for_robot(record, robot_id: str = "", allow_assigned: bool = False):
    if not isinstance(record, dict):
        return False
    return target_state_unavailable_for_robot(
        str(record.get("state", "") or ""),
        str(record.get("assigned_robot", "") or ""),
        robot_id,
        allow_assigned=allow_assigned,
    )


def target_state_unavailable_for_robot(
    state: str,
    assigned_robot: str,
    robot_id: str = "",
    allow_assigned: bool = False,
    reported: bool = False,
):
    if reported or state == "reported":
        return True
    if state in ("found", "report_ready"):
        return assigned_robot not in ("", robot_id)
    if state == "assigned":
        if allow_assigned:
            return False
        return bool(assigned_robot and assigned_robot != robot_id)
    return False


def _wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _tight_mask_dimensions_px(mask):
    rows, columns = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return 0, 0
    return (
        int(columns.max() - columns.min() + 1),
        int(rows.max() - rows.min() + 1),
    )


def _closest_detection_to_position(result, position):
    if result is None or position is None:
        return None
    detections = tuple(
        detection
        for detection in (getattr(result, "detections", ()) or ())
        if math.isfinite(detection.range_m)
    )
    if not detections:
        return None
    return min(
        detections,
        key=lambda detection: _distance(position, detection.world_position),
    )


@dataclass(frozen=True)
class VictimFrame:
    """Copied sensor frame safe for processing outside the Webots thread."""

    frame_id: int
    sim_time: float
    pose: tuple
    rgb: np.ndarray
    depth: np.ndarray


@dataclass
class VictimDetection:
    """One segmented victim with RGB-D local and world position estimates."""

    frame_id: int
    sim_time: float
    confidence: float
    box_xyxy: tuple
    mask: np.ndarray
    valid_depth_pixels: int
    valid_depth_fraction: float
    local_forward_m: float
    local_lateral_m: float
    local_height_m: float
    range_m: float
    world_position: tuple


@dataclass
class VictimVisualDetection:
    """One raw YOLO victim mask for diagnostics, even without valid depth."""

    frame_id: int
    sim_time: float
    confidence: float
    box_xyxy: tuple
    mask: np.ndarray
    valid_depth_pixels: int
    valid_depth_fraction: float
    range_m: float
    depth_status: str


@dataclass
class VictimFrameResult:
    """All detections produced from one submitted frame."""

    frame: VictimFrame
    detections: tuple = ()
    visual_detections: tuple = ()
    inference_s: float = 0.0
    error: str = ""


@dataclass
class VictimTrack:
    """Persistent identity joining a drone prior and repeated visual detections."""

    track_id: str
    prior_position: tuple = None
    visual_position: tuple = None
    uncertainty_m: float = 1.25
    confidence: float = 0.0
    observation_count: int = 0
    last_seen_time: float = -float("inf")
    seen_in_latest_frame: bool = False
    reportable_locks: list = field(default_factory=list)
    status: str = "UNSEARCHED"
    claimed_by: str = ""
    reported: bool = False
    coordinator_state: str = ""
    coordinator_assigned_robot: str = ""

    @property
    def position(self):
        return self.visual_position or self.prior_position

    @property
    def authoritative(self):
        return self.prior_position is not None

    @property
    def visual_lock(self):
        # Kept as a compatibility name for the mission state machine. There is
        # no multi-frame lock: one accepted YOLO+depth observation is enough.
        return self.visual_position is not None and self.seen_in_latest_frame


@dataclass
class MissionAction:
    """One navigation/report request emitted by VictimSearchController."""

    kind: str = "NONE"
    target_pose: tuple = None
    reason: str = ""
    track_id: str = ""
    confidence: float = 0.0
    strict_target: bool = False
    candidate_target_poses: tuple = ()


class VictimDetector:
    """
    Latest-frame-only YOLO segmentation worker.

    The main Webots loop copies RGB/depth/pose into submit(). The worker owns
    model inference and publishes only its newest result, preventing inference
    latency from blocking motor control or creating a stale-frame backlog.
    """

    def __init__(
        self,
        model_path,
        image_width,
        image_height,
        depth_min_m,
        depth_max_m,
        horizontal_fov,
        camera_forward_offset_m=LocalObstacleProjectionParams.CAMERA_FORWARD_OFFSET_M,
        camera_height_m=LocalObstacleProjectionParams.CAMERA_HEIGHT_M,
        enabled=True,
    ):
        self.model_path = Path(model_path)
        self.width = int(image_width)
        self.height = int(image_height)
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.horizontal_fov = float(horizontal_fov)
        self.camera_forward_offset_m = float(camera_forward_offset_m)
        self.camera_height_m = float(camera_height_m)
        self.enabled = bool(enabled) and _env_flag(
            "VICTIM_DETECTOR_ENABLED",
            VictimParams.DETECTOR_ENABLED,
        )
        self.image_size = _env_int(
            "VICTIM_MODEL_IMAGE_SIZE",
            VictimParams.MODEL_IMAGE_SIZE,
        )
        self.model_confidence = _env_float(
            "VICTIM_MODEL_CONFIDENCE",
            VictimParams.MODEL_CONFIDENCE,
        )
        self.max_range_m = _env_float("VICTIM_MAX_RANGE_M", VictimParams.MAX_RANGE_M)
        self.max_result_age_s = _env_float(
            "VICTIM_RESULT_MAX_AGE_S",
            VictimParams.RESULT_MAX_AGE_S,
        )
        self.min_depth_pixels = _env_int(
            "VICTIM_MIN_DEPTH_PIXELS",
            VictimParams.MIN_DEPTH_PIXELS,
        )
        self.min_depth_fraction = _env_float(
            "VICTIM_MIN_DEPTH_FRACTION",
            VictimParams.MIN_DEPTH_FRACTION,
        )
        self.mask_erode_px = max(
            0,
            _env_int("VICTIM_MASK_ERODE_PX", VictimParams.MASK_ERODE_PX),
        )
        self.transit_hz = max(
            0.1,
            _env_float("VICTIM_TRANSIT_HZ", VictimParams.TRANSIT_HZ),
        )
        self.scan_hz = max(
            0.1,
            _env_float("VICTIM_SCAN_HZ", VictimParams.SCAN_HZ),
        )
        self.device = os.environ.get("VICTIM_MODEL_DEVICE", VictimParams.MODEL_DEVICE)

        centre_x = 0.5 * (self.width - 1)
        centre_y = 0.5 * (self.height - 1)
        vertical_fov = 2.0 * math.atan(
            math.tan(self.horizontal_fov * 0.5) * (self.height / self.width)
        )
        columns = np.arange(self.width, dtype=np.float32)
        rows = np.arange(self.height, dtype=np.float32)
        self._lateral_factor = (
            (centre_x - columns)
            * math.tan(self.horizontal_fov * 0.5)
            / max(centre_x, 1.0)
        )
        self._vertical_factor = (
            (rows - centre_y)
            * math.tan(vertical_fov * 0.5)
            / max(centre_y, 1.0)
        )

        self.active = False
        self.scan_rate = False
        self.ready = False
        self.error_message = ""
        self._condition = threading.Condition()
        self._pending_frame = None
        self._latest_result = None
        self._last_result_id = -1
        self._last_submit_time = -float("inf")
        self._next_frame_id = 0
        self._stopping = False
        self._thread = None
        if self.enabled:
            self._thread = threading.Thread(
                target=self._worker,
                name="victim-detector",
                daemon=True,
            )
            self._thread.start()
            atexit.register(self.stop)

    @property
    def desired_interval_s(self):
        return 1.0 / (self.scan_hz if self.scan_rate else self.transit_hz)

    def should_submit(self, sim_time):
        return (
            self.active
            and sim_time - self._last_submit_time >= self.desired_interval_s
        )

    def set_active(self, active, scan_rate=False):
        self.active = bool(active) and self.enabled
        self.scan_rate = bool(scan_rate)

    def submit(self, sim_time, pose, rgb, depth):
        if not self.active or rgb is None or depth is None:
            return False
        if sim_time - self._last_submit_time < self.desired_interval_s:
            return False
        frame = VictimFrame(
            frame_id=self._next_frame_id,
            sim_time=float(sim_time),
            pose=(float(pose.x), float(pose.y), float(pose.yaw)),
            rgb=np.asarray(rgb, dtype=np.uint8).copy(),
            depth=np.asarray(depth, dtype=np.float32).copy(),
        )
        self._next_frame_id += 1
        self._last_submit_time = float(sim_time)
        with self._condition:
            self._pending_frame = frame
            self._condition.notify()
        return True

    def poll(self):
        with self._condition:
            result = self._latest_result
        if result is None or result.frame.frame_id == self._last_result_id:
            return None
        self._last_result_id = result.frame.frame_id
        return result

    def stop(self):
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _worker(self):
        model = None
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(self.model_path)
            os.environ.setdefault("MPLCONFIGDIR", "/tmp/sar-matplotlib")
            os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/sar-ultralytics")
            from ultralytics import YOLO

            model = YOLO(str(self.model_path))
            if getattr(model, "task", "") != "segment":
                raise ValueError("victim checkpoint is not a segmentation model")
            self.ready = True
        except Exception as exc:
            self.error_message = f"{type(exc).__name__}: {exc}"
            self.enabled = False
            return

        while True:
            with self._condition:
                while self._pending_frame is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                frame = self._pending_frame
                self._pending_frame = None
            result = self._infer(model, frame)
            with self._condition:
                self._latest_result = result

    def _infer(self, model, frame):
        started = time.perf_counter()
        try:
            # Ultralytics treats ndarray input as OpenCV BGR.
            bgr = frame.rgb[:, :, ::-1]
            prediction = model.predict(
                bgr,
                imgsz=self.image_size,
                conf=self.model_confidence,
                iou=0.5,
                max_det=10,
                device=self.device,
                retina_masks=True,
                verbose=False,
            )[0]
            detections, visual_detections = self._detections_from_prediction(
                frame,
                prediction,
            )
            return VictimFrameResult(
                frame=frame,
                detections=tuple(detections),
                visual_detections=tuple(visual_detections),
                inference_s=time.perf_counter() - started,
            )
        except Exception as exc:
            return VictimFrameResult(
                frame=frame,
                inference_s=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _detections_from_prediction(self, frame, prediction):
        if prediction.boxes is None or prediction.masks is None:
            return (), ()
        try:
            import cv2
        except ImportError:
            cv2 = None

        boxes = prediction.boxes.xyxy.detach().cpu().numpy()
        confidences = prediction.boxes.conf.detach().cpu().numpy()
        masks = prediction.masks.data.detach().cpu().numpy()
        detections = []
        visual_detections = []
        for box, confidence, raw_mask in zip(boxes, confidences, masks):
            if raw_mask.shape != (self.height, self.width):
                if cv2 is not None:
                    raw_mask = cv2.resize(
                        raw_mask,
                        (self.width, self.height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    from PIL import Image

                    raw_mask = np.asarray(
                        Image.fromarray(raw_mask).resize(
                            (self.width, self.height),
                            Image.Resampling.NEAREST,
                        )
                    )
            mask = raw_mask >= 0.5
            if self.mask_erode_px > 0 and cv2 is not None:
                size = 2 * self.mask_erode_px + 1
                kernel = np.ones((size, size), dtype=np.uint8)
                mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            visual_detections.append(
                self._visual_detection_from_mask(
                    frame,
                    mask,
                    float(confidence),
                    tuple(float(value) for value in box),
                )
            )
            detection = self._project_mask(
                frame,
                mask,
                float(confidence),
                tuple(float(value) for value in box),
            )
            if detection is not None:
                detections.append(detection)
        return detections, visual_detections

    def _visual_detection_from_mask(self, frame, mask, confidence, box_xyxy):
        depth = frame.depth
        valid = self._valid_depth_mask(depth, mask)
        valid_pixels = int(np.count_nonzero(valid))
        mask_pixels = max(1, int(np.count_nonzero(mask)))
        valid_fraction = valid_pixels / mask_pixels
        if valid_pixels > 0:
            measured_range = float(np.median(depth[valid]))
        else:
            measured_range = float("inf")
        if valid_pixels < self.min_depth_pixels:
            depth_status = "depth pixels low"
        elif valid_fraction < self.min_depth_fraction:
            depth_status = "depth fraction low"
        else:
            depth_status = "depth ok"
        return VictimVisualDetection(
            frame_id=frame.frame_id,
            sim_time=frame.sim_time,
            confidence=confidence,
            box_xyxy=box_xyxy,
            mask=mask,
            valid_depth_pixels=valid_pixels,
            valid_depth_fraction=valid_fraction,
            range_m=measured_range,
            depth_status=depth_status,
        )

    def _project_mask(self, frame, mask, confidence, box_xyxy):
        depth = frame.depth
        valid = self._valid_depth_mask(depth, mask)
        valid_pixels = int(np.count_nonzero(valid))
        mask_pixels = max(1, int(np.count_nonzero(mask)))
        valid_fraction = valid_pixels / mask_pixels
        if (
            valid_pixels < self.min_depth_pixels
            or valid_fraction < self.min_depth_fraction
        ):
            return None

        rows, columns = np.nonzero(valid)
        ranges = depth[rows, columns]
        forward = self.camera_forward_offset_m + ranges
        lateral = ranges * self._lateral_factor[columns]
        height = self.camera_height_m - ranges * self._vertical_factor[rows]
        local_forward = float(np.median(forward))
        local_lateral = float(np.median(lateral))
        local_height = float(np.median(height))
        measured_range = math.hypot(local_forward, local_lateral)
        if measured_range > self.max_range_m:
            return None

        pose_x, pose_y, pose_yaw = frame.pose
        c = math.cos(pose_yaw)
        s = math.sin(pose_yaw)
        world = (
            pose_x + c * local_forward - s * local_lateral,
            pose_y + s * local_forward + c * local_lateral,
        )
        return VictimDetection(
            frame_id=frame.frame_id,
            sim_time=frame.sim_time,
            confidence=confidence,
            box_xyxy=box_xyxy,
            mask=mask,
            valid_depth_pixels=valid_pixels,
            valid_depth_fraction=valid_fraction,
            local_forward_m=local_forward,
            local_lateral_m=local_lateral,
            local_height_m=local_height,
            range_m=measured_range,
            world_position=world,
        )

    def _valid_depth_mask(self, depth, mask):
        return (
            mask
            & np.isfinite(depth)
            & (depth >= self.depth_min_m)
            & (depth <= min(self.depth_max_m, self.max_range_m))
        )


class VictimTracker:
    """
    Drone-prior-authoritative victim association and temporal lock filtering.

    P tracks come from the drone estimates and are the only identities that may
    drive mission behaviour or reporting. Multiple YOLO masks in one frame may
    be fragments of the same victim, so all unambiguous fragments assigned to
    one prior are fused into one observation for that frame.

    Unmatched or ambiguous detections remain short-lived V tracks for visual
    diagnostics. They can never increase the confirmed victim count.
    """

    def __init__(self, priors):
        self.prior_radius_m = _env_float(
            "VICTIM_PRIOR_ASSOCIATION_M",
            VictimParams.PRIOR_ASSOCIATION_M,
        )
        self.visual_radius_m = _env_float(
            "VICTIM_VISUAL_ASSOCIATION_M",
            VictimParams.VISUAL_ASSOCIATION_M,
        )
        self.fusion_alpha = _env_float(
            "VICTIM_TRACK_FUSION_ALPHA",
            VictimParams.TRACK_FUSION_ALPHA,
        )
        self.prior_ambiguity_margin_m = _env_float(
            "VICTIM_PRIOR_AMBIGUITY_MARGIN_M",
            VictimParams.PRIOR_AMBIGUITY_MARGIN_M,
        )
        self.temporary_ttl_s = _env_float(
            "VICTIM_TEMP_TRACK_TTL_S",
            VictimParams.TEMP_TRACK_TTL_S,
        )
        self.tracks = []
        for index, prior in enumerate(priors, start=1):
            self.tracks.append(
                VictimTrack(
                    track_id=f"P{index}",
                    prior_position=(float(prior[0]), float(prior[1])),
                    uncertainty_m=self.prior_radius_m,
                )
            )
        self._next_visual_id = 1
        self.last_frame_id = -1

    def update(self, result):
        if result is None or result.frame.frame_id <= self.last_frame_id:
            return ()
        self.last_frame_id = result.frame.frame_id
        self._expire_temporary_tracks(result.frame.sim_time)
        for track in self.tracks:
            track.seen_in_latest_frame = False

        prior_buckets = {}
        unmatched_detections = []
        for detection_index, detection in enumerate(result.detections):
            prior = self._select_authoritative_prior(detection)
            if prior is None:
                unmatched_detections.append((detection_index, detection))
                continue
            prior_buckets.setdefault(prior.track_id, []).append(detection)

        for prior_id, detections in prior_buckets.items():
            self._update_track_from_fragments(self.get(prior_id), detections)

        temporary_tracks = [
            track for track in self.tracks if not track.authoritative
        ]
        candidates = []
        for unmatched_index, (_, detection) in enumerate(unmatched_detections):
            for track_index, track in enumerate(temporary_tracks):
                distance = _distance(track.position, detection.world_position)
                if distance <= self.visual_radius_m:
                    candidates.append(
                        (
                            distance,
                            detection.frame_id,
                            track.track_id,
                            unmatched_index,
                            track_index,
                        )
                    )
        assigned_detections = set()
        assigned_tracks = set()
        for _, _, _, detection_index, track_index in sorted(candidates):
            if (
                detection_index in assigned_detections
                or track_index in assigned_tracks
            ):
                continue
            self._update_track(
                temporary_tracks[track_index],
                unmatched_detections[detection_index][1],
            )
            assigned_detections.add(detection_index)
            assigned_tracks.add(track_index)

        for detection_index, (_, detection) in enumerate(unmatched_detections):
            if detection_index in assigned_detections:
                continue
            track = VictimTrack(
                track_id=f"V{self._next_visual_id}",
                visual_position=detection.world_position,
                uncertainty_m=self.visual_radius_m,
            )
            self._next_visual_id += 1
            self.tracks.append(track)
            track.seen_in_latest_frame = True
            track.observation_count = 1
            track.last_seen_time = detection.sim_time
            track.confidence = detection.confidence

        return tuple(self.tracks)

    def _select_authoritative_prior(self, detection):
        continuity = []
        prior_candidates = []
        for track in self.tracks:
            if not track.authoritative:
                continue
            prior_distance = _distance(
                track.prior_position,
                detection.world_position,
            )
            if (
                track.visual_position is not None
                and _distance(
                    track.visual_position,
                    detection.world_position,
                ) <= self.visual_radius_m
            ):
                continuity.append(
                    (
                        _distance(
                            track.visual_position,
                            detection.world_position,
                        ),
                        track.track_id,
                        track,
                    )
                )
            if prior_distance <= self.prior_radius_m:
                prior_candidates.append(
                    (prior_distance, track.track_id, track)
                )

        candidates = continuity if continuity else prior_candidates
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        if (
            len(candidates) > 1
            and candidates[1][0] - candidates[0][0]
            < self.prior_ambiguity_margin_m
        ):
            return None
        return candidates[0][2]

    def _update_track_from_fragments(self, track, detections):
        if track is None or not detections:
            return
        weights = np.asarray(
            [
                max(0.05, detection.confidence)
                * max(1, detection.valid_depth_pixels)
                for detection in detections
            ],
            dtype=np.float64,
        )
        weights /= float(np.sum(weights))
        measured_position = (
            float(
                sum(
                    weight * detection.world_position[0]
                    for weight, detection in zip(weights, detections)
                )
            ),
            float(
                sum(
                    weight * detection.world_position[1]
                    for weight, detection in zip(weights, detections)
                )
            ),
        )
        measured_confidence = float(
            sum(
                weight * detection.confidence
                for weight, detection in zip(weights, detections)
            )
        )
        self._update_track_measurement(
            track,
            measured_position,
            measured_confidence,
            max(detection.sim_time for detection in detections),
        )

    def _update_track(self, track, detection):
        self._update_track_measurement(
            track,
            detection.world_position,
            detection.confidence,
            detection.sim_time,
        )

    def _update_track_measurement(
        self,
        track,
        measured_position,
        measured_confidence,
        sim_time,
    ):
        if track.visual_position is None:
            fused = measured_position
        else:
            alpha = max(0.05, min(1.0, self.fusion_alpha))
            fused = (
                (1.0 - alpha) * track.visual_position[0]
                + alpha * measured_position[0],
                (1.0 - alpha) * track.visual_position[1]
                + alpha * measured_position[1],
            )
        track.visual_position = fused
        track.uncertainty_m = max(
            0.15,
            min(track.uncertainty_m, self.visual_radius_m)
            * 0.85,
        )
        track.confidence = (
            measured_confidence
            if track.observation_count == 0
            else 0.7 * track.confidence + 0.3 * measured_confidence
        )
        track.observation_count += 1
        track.last_seen_time = sim_time
        track.seen_in_latest_frame = True
        if track.visual_lock and track.status not in ("FOUND", "REPORTED"):
            track.status = "LOCKED"

    def _expire_temporary_tracks(self, sim_time):
        self.tracks = [
            track
            for track in self.tracks
            if (
                track.authoritative
                or sim_time - track.last_seen_time <= self.temporary_ttl_s
            )
        ]

    def get(self, track_id):
        return next(
            (track for track in self.tracks if track.track_id == track_id),
            None,
        )

    def unsearched_priors(self, robot_id: str = ""):
        return tuple(
            track
            for track in self.tracks
            if (
                track.prior_position is not None
                and track.status in ("UNSEARCHED", "SEARCHING")
                and not self.target_unavailable_for_robot(track, robot_id)
            )
        )

    def apply_coordinator_targets(self, records, robot_id: str = ""):
        if not isinstance(records, dict):
            return
        for track in self.tracks:
            record = records.get(track.track_id)
            if not isinstance(record, dict):
                continue
            state = str(record.get("state", "") or "")
            assigned_robot = str(record.get("assigned_robot", "") or "")
            track.coordinator_state = state
            track.coordinator_assigned_robot = assigned_robot
            if state == "reported":
                track.reported = True
                track.status = "REPORTED"
            elif self.target_unavailable_for_robot(track, robot_id):
                if track.status not in ("REPORTED",):
                    track.status = "FOUND"
            elif (
                track.status == "FOUND"
                and not track.reported
                and state in ("unassigned", "route_failed", "exhausted", "")
            ):
                track.status = "UNSEARCHED"

    @staticmethod
    def target_unavailable_for_robot(track, robot_id: str = "", allow_assigned: bool = False):
        return target_state_unavailable_for_robot(
            str(getattr(track, "coordinator_state", "") or ""),
            str(getattr(track, "coordinator_assigned_robot", "") or ""),
            robot_id,
            allow_assigned=allow_assigned,
            reported=track.reported,
        )


class VictimSearchController:
    """Coordinator-directed victim search, approach, close-in, and report policy."""

    IDLE = "IDLE"
    NAVIGATE = "NAVIGATE"
    SEARCH_360 = "SEARCH_360"
    APPROACH = "APPROACH"
    CLOSE_IN = "CLOSE_IN"
    FOUND = "FOUND"
    FAILED = "FAILED"

    def __init__(self, robot_id, tracker, enabled=True):
        self.robot_id = robot_id
        self.tracker = tracker
        self.enabled = bool(enabled)
        self.debug_print = _env_flag(
            "VICTIM_DEBUG_PRINT",
            VictimParams.DEBUG_PRINT_ENABLED,
        )
        self.state = self.IDLE
        self.reason = ""
        self.selected_track_id = ""
        self.current_target_pose = None
        self.approach_pose = None
        self.approach_refresh_attempted = False
        requested_selection = os.environ.get(
            "VICTIM_PRIOR_SELECTION_MODE",
            VictimParams.PRIOR_SELECTION_MODE,
        ).strip().lower()
        self.prior_selection_mode = (
            "furthest" if requested_selection == "furthest" else "nearest"
        )
        self.search_timeout_s = _env_float(
            "VICTIM_SEARCH_TIMEOUT_S",
            VictimParams.SEARCH_TIMEOUT_S,
        )
        self.scan_accumulated_rad = 0.0
        self.scan_last_yaw = None
        self.required_approach_distance_m = _env_float(
            "VICTIM_REQUIRED_APPROACH_DISTANCE_M",
            VictimParams.REQUIRED_APPROACH_DISTANCE_M,
        )
        self.side_view_enabled = _env_flag(
            "VICTIM_SIDE_VIEW_ENABLED",
            VictimParams.SIDE_VIEW_ENABLED,
        )
        self.side_view_range_min_m = _env_float(
            "VICTIM_SIDE_VIEW_RANGE_MIN_M",
            VictimParams.SIDE_VIEW_RANGE_MIN_M,
        )
        self.side_view_range_max_m = _env_float(
            "VICTIM_SIDE_VIEW_RANGE_MAX_M",
            VictimParams.SIDE_VIEW_RANGE_MAX_M,
        )
        self.side_view_min_mask_width_px = _env_int(
            "VICTIM_SIDE_VIEW_MIN_MASK_WIDTH_PX",
            VictimParams.SIDE_VIEW_MIN_MASK_WIDTH_PX,
        )
        self.side_view_min_mask_height_px = _env_int(
            "VICTIM_SIDE_VIEW_MIN_MASK_HEIGHT_PX",
            VictimParams.SIDE_VIEW_MIN_MASK_HEIGHT_PX,
        )
        self.side_view_max_mask_height_px = _env_int(
            "VICTIM_SIDE_VIEW_MAX_MASK_HEIGHT_PX",
            VictimParams.SIDE_VIEW_MAX_MASK_HEIGHT_PX,
        )
        self.side_view_distance_m = _env_float(
            "VICTIM_SIDE_VIEW_DISTANCE_M",
            VictimParams.SIDE_VIEW_DISTANCE_M,
        )
        self.report_distance_m = _env_float(
            "VICTIM_REPORT_DISTANCE_M",
            VictimParams.REPORT_DISTANCE_M,
        )
        self.close_in_final_distance_m = _env_float(
            "VICTIM_CLOSE_IN_FINAL_DISTANCE_M",
            VictimParams.CLOSE_IN_FINAL_DISTANCE_M,
        )
        self.close_in_speed_cap_rad_s = max(
            0.0,
            _env_float(
                "VICTIM_CLOSE_IN_SPEED_CAP_RAD_S",
                VictimParams.CLOSE_IN_SPEED_CAP_RAD_S,
            ),
        )
        self.close_in_min_advance_m = _env_float(
            "VICTIM_CLOSE_IN_MIN_ADVANCE_M",
            VictimParams.CLOSE_IN_MIN_ADVANCE_M,
        )
        self.report_lock_max_age_s = _env_float(
            "VICTIM_REPORT_LOCK_MAX_AGE_S",
            VictimParams.REPORT_LOCK_MAX_AGE_S,
        )
        self.last_result_frame_id = -1
        self.search_started_time = 0.0
        self.encountered_track_id = ""
        self.dismissed_encounter_track_id = ""
        self.close_in_target_pose = None
        self.close_in_started = False
        self.close_in_committed_track_id = ""
        self.close_in_committed_confidence = 0.0
        self.close_in_committed_position = None
        self.report_confidence = 0.0
        self.report_status = ""
        self._pending_action = MissionAction()

    def _debug(self, message: str):
        if not self.debug_print:
            return
        print(
            f"[{self.robot_id}] VICTIM DEBUG "
            f"state={self.state} track={self.selected_track_id or '-'}: {message}"
        )

    @property
    def active(self):
        return self.state not in (
            self.IDLE,
            self.FOUND,
            self.FAILED,
        )

    @property
    def scan_rate_requested(self):
        return self.state == self.SEARCH_360

    @property
    def selected_track(self):
        return self.tracker.get(self.selected_track_id)

    def toggle_prior_selection_mode(self):
        self.prior_selection_mode = (
            "furthest"
            if self.prior_selection_mode == "nearest"
            else "nearest"
        )
        return self.prior_selection_mode

    def start(
        self,
        current_pose,
        sim_time,
        evaluate_route,
        target_track_id="",
    ):
        if not self.enabled:
            return self._fail("victim search is disabled")
        self.cancel(reset_state=False)
        self.search_started_time = sim_time
        self.close_in_target_pose = None
        self.close_in_started = False
        self.report_confidence = 0.0
        self.report_status = ""
        candidates = []
        prior_tracks = self.tracker.unsearched_priors(self.robot_id)
        if target_track_id:
            track = self.tracker.get(target_track_id)
            if (
                track is None
                or track.prior_position is None
                or track.reported
                or self.tracker.target_unavailable_for_robot(track, self.robot_id)
            ):
                return self._fail(
                    f"assigned victim prior {target_track_id} is unavailable"
                )
            prior_tracks = (track,)
        for track in prior_tracks:
            target = (
                track.prior_position[0],
                track.prior_position[1],
                current_pose.yaw,
            )
            success, cost = evaluate_route(target)
            if success:
                candidates.append((cost, track))
        if not candidates:
            return self._fail("no reachable unsearched victim prior")

        ordered_candidates = sorted(
            candidates,
            key=lambda item: (item[0], item[1].track_id),
            reverse=self.prior_selection_mode == "furthest",
        )
        selected = ordered_candidates[0][1]
        selected.status = "SEARCHING"
        selected.claimed_by = self.robot_id
        self.selected_track_id = selected.track_id
        self._debug(
            "selected prior "
            f"{selected.track_id} cost={ordered_candidates[0][0]:.2f} "
            f"candidates={len(ordered_candidates)}"
        )
        return self._start_prior_route(current_pose)

    def cancel(self, reset_state=True):
        if reset_state:
            self.state = self.IDLE
            self.reason = "cancelled"
        self.selected_track_id = ""
        self.current_target_pose = None
        self.approach_pose = None
        self.approach_refresh_attempted = False
        self.encountered_track_id = ""
        self.dismissed_encounter_track_id = ""
        self.close_in_target_pose = None
        self.close_in_started = False
        self._reset_close_in_commit()
        self.scan_accumulated_rad = 0.0
        self.scan_last_yaw = None
        self._pending_action = MissionAction()

    def on_result(self, result, current_pose, sim_time):
        if result is None or result.frame.frame_id <= self.last_result_frame_id:
            return
        self.last_result_frame_id = result.frame.frame_id
        self._update_reportable_locks(sim_time)
        if self.state == self.CLOSE_IN:
            return
        side_view_action = self._side_view_action(result, current_pose)
        if side_view_action is not None:
            self._pending_action = side_view_action
            return
        visible_tracks = tuple(
            track
            for track in self.tracker.tracks
            if track.visual_lock
            and not self.tracker.target_unavailable_for_robot(
                track,
                self.robot_id,
                allow_assigned=True,
            )
        )
        visible_ids = {track.track_id for track in visible_tracks}
        if self.dismissed_encounter_track_id not in visible_ids:
            self.dismissed_encounter_track_id = ""

        selected = self.selected_track
        if (
            selected is not None
            and selected.visual_lock
            and self.state in (self.NAVIGATE, self.SEARCH_360)
        ):
            self.encountered_track_id = ""
            self._pending_action = MissionAction(
                "APPROACH",
                track_id=self.selected_track_id,
                reason="assigned victim detected",
            )
            return

        if self.state not in (self.NAVIGATE, self.SEARCH_360):
            return
        candidates = [
            track
            for track in visible_tracks
            if track.track_id != self.selected_track_id
            and track.track_id != self.dismissed_encounter_track_id
        ]
        if not candidates or self.encountered_track_id:
            return
        encountered = min(
            candidates,
            key=lambda track: _distance(
                (current_pose.x, current_pose.y),
                track.position,
            ),
        )
        self.encountered_track_id = encountered.track_id
        self._pending_action = MissionAction(
            "ENCOUNTER",
            reason=f"encountered different victim {encountered.track_id}",
            track_id=encountered.track_id,
        )

    def deny_encounter(self, track_id):
        if track_id != self.encountered_track_id:
            return
        self.dismissed_encounter_track_id = track_id
        self.encountered_track_id = ""
        if self._pending_action.kind == "ENCOUNTER":
            self._pending_action = MissionAction()

    def update(
        self,
        current_pose,
        sim_time,
        route_done,
        route_failed,
        robot_stopped,
    ):
        if self._pending_action.kind == "ENCOUNTER":
            action = self._pending_action
            self._pending_action = MissionAction()
            return action
        if (
            self._pending_action.kind == "ROUTE"
            and self._pending_action.candidate_target_poses
        ):
            action = self._pending_action
            self._pending_action = MissionAction()
            if not route_failed:
                return action
        if self._pending_action.kind == "APPROACH":
            trigger_reason = self._pending_action.reason
            self._pending_action = MissionAction()
            return self._start_direct_approach(
                current_pose,
                sim_time,
                reason=trigger_reason,
            )
        if not self.active:
            return MissionAction()
        if (
            self.state in (self.NAVIGATE, self.SEARCH_360, self.APPROACH)
            and sim_time - self.search_started_time > self.search_timeout_s
        ):
            return self._fail("victim search time budget exhausted")
        if route_failed:
            self._debug(
                "route_failed received "
                f"route_done={route_done} robot_stopped={robot_stopped}"
            )
            if self.state == self.NAVIGATE:
                return self._fail("direct victim-prior route failed")
            if self.state == self.APPROACH:
                return self._begin_close_in_or_search(
                    current_pose,
                    sim_time,
                    "approach route failed after replanning; using close-in fallback",
                )
            if self.state == self.CLOSE_IN:
                return self._finish_close_in(
                    "close-in route failed",
                )

        track = self.selected_track
        if self.state == self.NAVIGATE:
            track = self.selected_track
            close_enough = route_done
            if not close_enough and track is not None and track.prior_position is not None:
                px, py = track.prior_position
                close_enough = math.hypot(current_pose.x - px, current_pose.y - py) <= 0.5
            if close_enough:
                return self._start_scan(current_pose, sim_time)
        if self.state == self.SEARCH_360:
            return self._update_scan(current_pose)
        if self.state == self.APPROACH:
            if route_done:
                return self._complete_approach(current_pose, sim_time)
        if self.state == self.CLOSE_IN:
            if route_done:
                return self._finish_close_in("close-in target reached")
            return MissionAction()
        return MissionAction()

    def mark_reported(self, confidence, dry_run, current_pose=None):
        track = self.selected_track
        if track is not None:
            track.reported = True
            track.status = "REPORTED" if not dry_run else "FOUND"
        self.report_confidence = confidence
        self.report_status = "DRY RUN" if dry_run else "REPORTED"
        self.state = self.FOUND
        self.reason = "victim report prepared" if dry_run else "victim reported"
        self.approach_refresh_attempted = False

    def snapshot(self):
        selected = self.selected_track
        reportable_lock = None
        if selected is not None and selected.reportable_locks:
            reportable_lock = max(
                selected.reportable_locks,
                key=lambda lock: (lock[1], lock[0]),
            )
        return {
            "state": self.state,
            "reason": self.reason,
            "prior_selection_mode": self.prior_selection_mode,
            "selected_track_id": self.selected_track_id,
            "prior": selected.prior_position if selected else None,
            "approach_target": self.approach_pose,
            "approach_refresh_attempted": self.approach_refresh_attempted,
            "encountered_track_id": self.encountered_track_id,
            "close_in_target": self.close_in_target_pose,
            "close_in_started": self.close_in_started,
            "report_status": self.report_status,
            "report_confidence": self.report_confidence,
            "reportable_confidence": reportable_lock[1] if reportable_lock else 0.0,
            "reportable_time": reportable_lock[0] if reportable_lock else None,
            "close_in_committed_confidence": self.close_in_committed_confidence,
            "tracks": tuple(
                {
                    "id": track.track_id,
                    "position": track.position,
                    "locked": track.visual_lock,
                    "authoritative": track.authoritative,
                    "status": track.status,
                    "uncertainty_m": track.uncertainty_m,
                    "confidence": track.confidence,
                    "reportable_locks": tuple(track.reportable_locks),
                }
                for track in self.tracker.tracks
            ),
        }

    def _start_prior_route(self, current_pose):
        track = self.selected_track
        if track is None or track.prior_position is None:
            return self._fail("selected prior has no position")
        prior_x, prior_y = track.prior_position
        yaw = math.atan2(prior_y - current_pose.y, prior_x - current_pose.x)
        target = (prior_x, prior_y, yaw)
        self.current_target_pose = target
        self.state = self.NAVIGATE
        self.reason = "routing directly to victim prior"
        self._debug(
            f"routing to prior target=({prior_x:.2f},{prior_y:.2f}) "
            f"yaw={math.degrees(yaw):.1f}deg"
        )
        return MissionAction(
            "ROUTE",
            target,
            self.reason,
            self.selected_track_id,
            strict_target=False,
        )

    def _start_scan(self, current_pose, sim_time):
        del sim_time
        self.scan_accumulated_rad = 0.0
        self.scan_last_yaw = current_pose.yaw
        self.state = self.SEARCH_360
        self.reason = "search spin 0°/360° for assigned victim"
        return MissionAction("SPIN", reason=self.reason, track_id=self.selected_track_id)

    def _update_scan(self, current_pose):
        delta = _wrap_angle(current_pose.yaw - self.scan_last_yaw)
        self.scan_accumulated_rad += abs(delta)
        self.scan_last_yaw = current_pose.yaw
        if self.scan_accumulated_rad < 2.0 * math.pi:
            self.reason = (
                f"search spin {math.degrees(self.scan_accumulated_rad):.0f}°/360°"
            )
            return MissionAction("SPIN", reason=self.reason, track_id=self.selected_track_id)
        return self._fail("360° search complete, assigned victim not detected")

    def _side_view_action(self, result, current_pose):
        if (
            not self.side_view_enabled
            or self.state != self.APPROACH
            or self.approach_refresh_attempted
        ):
            return None
        track = self.selected_track
        if track is None or not track.visual_lock or track.position is None:
            return None
        detection = _closest_detection_to_position(result, track.position)
        if detection is None:
            return None
        width_px, height_px = _tight_mask_dimensions_px(detection.mask)
        if not self._side_view_mask_matches(
            detection.range_m,
            width_px,
            height_px,
        ):
            return None

        candidates = self._side_view_target_poses(
            current_pose,
            track.position,
        )
        if not candidates:
            return None
        self.approach_refresh_attempted = True
        self.reason = (
            "wide low victim mask requires one side-view approach "
            f"(range={detection.range_m:.2f}m mask={width_px}x{height_px}px)"
        )
        self._debug(self.reason)
        return MissionAction(
            "ROUTE",
            target_pose=candidates[0],
            reason=self.reason,
            track_id=self.selected_track_id,
            candidate_target_poses=candidates,
        )

    def _side_view_mask_matches(self, range_m, width_px, height_px):
        return (
            self.side_view_range_min_m <= float(range_m) <= self.side_view_range_max_m
            and int(width_px) >= self.side_view_min_mask_width_px
            and self.side_view_min_mask_height_px
            <= int(height_px)
            <= self.side_view_max_mask_height_px
        )

    def _side_view_target_poses(self, current_pose, victim_position):
        victim_x, victim_y = victim_position
        outward_x = current_pose.x - victim_x
        outward_y = current_pose.y - victim_y
        distance = math.hypot(outward_x, outward_y)
        if distance <= 1e-6:
            return ()
        outward_x /= distance
        outward_y /= distance
        radius = self.side_view_distance_m
        side_directions = (
            (-outward_y, outward_x),
            (outward_y, -outward_x),
        )
        targets = []
        for side_x, side_y in side_directions:
            x = victim_x + radius * side_x
            y = victim_y + radius * side_y
            yaw = math.atan2(victim_y - y, victim_x - x)
            targets.append((x, y, yaw))
        return tuple(targets)

    def accept_approach_target(self, target_pose, reason=""):
        self.approach_pose = tuple(target_pose)
        self.current_target_pose = tuple(target_pose)
        if reason:
            self.reason = reason

    def _start_direct_approach(
        self,
        current_pose,
        sim_time,
        reason="",
    ):
        track = self.selected_track
        if track is None or track.position is None:
            return self._start_scan(current_pose, sim_time)
        self.approach_refresh_attempted = False
        victim = track.position
        distance = _distance((current_pose.x, current_pose.y), victim)
        if distance <= self.required_approach_distance_m:
            return self._begin_close_in_or_search(
                current_pose,
                sim_time,
                "victim already within required approach distance",
            )
        return self._start_approach_route(
            current_pose,
            victim,
            reason or f"approaching assigned victim {track.track_id}",
        )

    def _start_approach_route(self, current_pose, victim_position, reason):
        victim = victim_position
        angle = math.atan2(current_pose.y - victim[1], current_pose.x - victim[0])
        x = victim[0] + self.required_approach_distance_m * math.cos(angle)
        y = victim[1] + self.required_approach_distance_m * math.sin(angle)
        yaw = math.atan2(victim[1] - y, victim[0] - x)
        target = (x, y, yaw)
        self.approach_pose = target
        self.current_target_pose = target
        self.state = self.APPROACH
        self.reason = reason
        self._debug(
            f"approach target=({x:.2f},{y:.2f}) "
            f"distance={self.required_approach_distance_m:.2f}m"
        )
        return MissionAction("ROUTE", target, self.reason, self.selected_track_id)

    def _complete_approach(self, current_pose, sim_time):
        track = self.selected_track
        best_lock = self._best_reportable_lock(track, sim_time)
        newest_lock = self._newest_reportable_lock(track, sim_time)
        if best_lock is None or newest_lock is None:
            return self._start_scan(current_pose, sim_time)

        newest_position = newest_lock[2]
        distance = _distance((current_pose.x, current_pose.y), newest_position)
        if (
            not self.approach_refresh_attempted
            and distance > self.required_approach_distance_m
        ):
            self.approach_refresh_attempted = True
            reason = (
                "latest fused victim centre remains "
                f"{distance:.2f}m away; updating approach route once"
            )
            self._debug(reason)
            return self._start_approach_route(
                current_pose,
                newest_position,
                reason,
            )

        reason = (
            "updated approach route completed"
            if self.approach_refresh_attempted
            else "required approach distance reached"
        )
        return self._begin_close_in_or_search(current_pose, sim_time, reason)

    def accept_close_in_safety_stop(self, current_pose, sim_time):
        del current_pose, sim_time
        if self.state != self.CLOSE_IN:
            return MissionAction()
        return self._finish_close_in("close-in stopped by safety")

    def _begin_close_in_or_search(self, current_pose, sim_time, reason):
        track = self.selected_track
        best_lock = self._best_reportable_lock(track, sim_time)
        newest_lock = self._newest_reportable_lock(track, sim_time)
        if best_lock is None or newest_lock is None:
            return self._start_scan(current_pose, sim_time)
        self._commit_close_in_track(track, best_lock, newest_lock)
        self.state = self.CLOSE_IN
        self.close_in_started = True
        self.reason = reason
        return self._start_close_in_motion(current_pose)

    def _start_close_in_motion(self, current_pose):
        target_position = self.close_in_committed_position
        if target_position is None:
            return self._fail("close-in has no committed victim position")
        distance = _distance((current_pose.x, current_pose.y), target_position)
        if distance <= max(self.report_distance_m, self.close_in_final_distance_m):
            return self._committed_close_in_report("close-in distance reached")
        advance = distance - self.close_in_final_distance_m
        if advance < self.close_in_min_advance_m:
            return self._committed_close_in_report("close-in minimum advance reached")
        unit_x = (target_position[0] - current_pose.x) / distance
        unit_y = (target_position[1] - current_pose.y) / distance
        target_x = current_pose.x + unit_x * advance
        target_y = current_pose.y + unit_y * advance
        yaw = math.atan2(target_position[1] - target_y, target_position[0] - target_x)
        target = (target_x, target_y, yaw)
        self.close_in_target_pose = target
        self.current_target_pose = target
        self.reason = (
            f"continuous close-in movement {advance:.2f} m toward victim "
            f"(distance {distance:.2f} m)"
        )
        self._debug(
            f"committed close-in confidence={self.close_in_committed_confidence:.2f} "
            f"target=({target_x:.2f},{target_y:.2f})"
        )
        return MissionAction(
            "CLOSE_IN",
            target,
            self.reason,
            self.close_in_committed_track_id,
        )

    def _finish_close_in(self, reason):
        self._debug(f"finishing close-in: {reason}")
        action = self._committed_close_in_report(reason)
        if action.kind == "REPORT":
            return action
        return self._fail("close-in ended without a committed report lock")

    def _commit_close_in_track(self, track, confidence_lock, position_lock=None):
        if track is None or confidence_lock is None:
            return False
        if position_lock is None:
            position_lock = confidence_lock
        self.close_in_committed_track_id = track.track_id
        self.close_in_committed_confidence = confidence_lock[1]
        self.close_in_committed_position = position_lock[2]
        self.report_confidence = self.close_in_committed_confidence
        return True

    def _committed_close_in_report(self, reason):
        track_id = self.close_in_committed_track_id
        confidence = max(0.0, min(1.0, float(self.close_in_committed_confidence)))
        if not track_id:
            return MissionAction()
        if confidence <= 0.0:
            return MissionAction()
        track = self.tracker.get(track_id)
        if (
            track is not None
            and self.tracker.target_unavailable_for_robot(
                track,
                self.robot_id,
                allow_assigned=True,
            )
        ):
            self.reason = "victim already handled by coordinator"
            return MissionAction("HOLD", reason=self.reason, track_id=track_id)
        return self._report_track_id_action(track_id, confidence, reason)

    def _reset_close_in_commit(self):
        self.close_in_committed_track_id = ""
        self.close_in_committed_confidence = 0.0
        self.close_in_committed_position = None

    @staticmethod
    def _report_track_id_action(track_id, confidence, reason):
        return MissionAction(
            "REPORT",
            reason=reason,
            track_id=track_id,
            confidence=confidence,
        )

    def _update_reportable_locks(self, sim_time):
        min_confidence = self._report_min_confidence()
        for track in self.tracker.tracks:
            self._prune_reportable_locks(track, sim_time)
            if not track.seen_in_latest_frame or track.position is None:
                continue
            confidence = self._report_confidence(track)
            if confidence < min_confidence:
                continue
            position = (float(track.position[0]), float(track.position[1]))
            track.reportable_locks.append(
                (float(track.last_seen_time), confidence, position)
            )
            self._prune_reportable_locks(track, sim_time)

    def _best_reportable_lock(self, track, sim_time):
        if track is None:
            return None
        self._prune_reportable_locks(track, sim_time)
        if not track.reportable_locks:
            return None
        return max(track.reportable_locks, key=lambda lock: (lock[1], lock[0]))

    def _newest_reportable_lock(self, track, sim_time):
        if track is None:
            return None
        self._prune_reportable_locks(track, sim_time)
        if not track.reportable_locks:
            return None
        return max(track.reportable_locks, key=lambda lock: (lock[0], lock[1]))

    def _prune_reportable_locks(self, track, sim_time):
        locks = getattr(track, "reportable_locks", None)
        if locks is None:
            track.reportable_locks = []
            return
        max_age_s = max(0.0, float(self.report_lock_max_age_s))
        track.reportable_locks = [
            lock
            for lock in locks
            if sim_time - lock[0] <= max_age_s
        ]

    @staticmethod
    def _report_confidence(track):
        return max(0.0, min(1.0, float(track.confidence)))

    @staticmethod
    def _report_min_confidence():
        return _env_float(
            "VICTIM_REPORT_MIN_CONFIDENCE",
            VictimParams.REPORT_MIN_CONFIDENCE,
        )

    def _fail(self, reason):
        self._debug(f"FAILED: {reason}")
        track = self.selected_track
        if track is not None and track.status != "FOUND":
            track.status = "SEARCH_EXHAUSTED"
        self.state = self.FAILED
        self.reason = reason
        self.approach_refresh_attempted = False
        return MissionAction("FAIL", reason=reason, track_id=self.selected_track_id)


class VictimReporter:
    """Exactly-once channel-43 reporting with env-controlled dry-run mode."""

    def __init__(self, robot_id, emitter=None):
        self.robot_id = robot_id
        self.emitter = emitter
        self.enabled = _env_flag(
            "VICTIM_REPORT_ENABLED",
            VictimParams.REPORT_ENABLED,
        )
        self.reported_track_ids = set()
        self.last_payload = None
        self.last_status = ""

    def report(self, track_id, sim_time, world_pose, confidence):
        if not (
            isinstance(track_id, str)
            and track_id.startswith("P")
            and track_id[1:].isdigit()
        ):
            self.last_status = "NON-PRIOR SUPPRESSED"
            return False, True, self.last_payload
        if track_id in self.reported_track_ids:
            self.last_status = "DUPLICATE SUPPRESSED"
            return False, True, self.last_payload
        payload = {
            "timestamp": float(sim_time),
            "robot_id": self.robot_id,
            "position": [
                float(world_pose.x),
                float(world_pose.y),
                0.0,
            ],
            "victim_found": True,
            "victim_confidence": float(max(0.0, min(1.0, confidence))),
        }
        self.last_payload = payload
        dry_run = not self.enabled
        if dry_run:
            self.last_status = "DRY RUN"
            print(f"[{self.robot_id}] VICTIM REPORT DRY RUN: {json.dumps(payload)}")
        else:
            if self.emitter is None:
                self.last_status = "ERROR: emitter unavailable"
                return False, False, payload
            self.emitter.send(json.dumps(payload).encode())
            self.last_status = "SENT"
            print(f"[{self.robot_id}] VICTIM REPORT SENT: {json.dumps(payload)}")
        self.reported_track_ids.add(track_id)
        return True, dry_run, payload


class VictimDebugViewer:
    """Optional annotated RGB viewer for masks, positions, and track state."""

    def __init__(self, robot_id, output_dir, enabled=True):
        self.robot_id = robot_id
        viewer_requested = bool(enabled) and _env_flag(
            "VICTIM_VIEWER_ENABLED",
            robot_id == "robot1" and VictimParams.VIEWER_ENABLED_ROBOT1_DEFAULT,
        )
        self.enabled = bool(enabled) and _env_flag(
            "VICTIM_DEBUG_RENDER",
            viewer_requested,
        )
        self.viewer_enabled = bool(viewer_requested and self.enabled)
        self.output_path = Path(output_dir) / f"{robot_id}_victim_detection.png"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._viewer_process = None
        if self.enabled:
            # Replace any image left by an earlier controller run. Otherwise a
            # loading or failed detector looks like a frozen, valid detection.
            self.render_status("waiting for first inference")
        if self.viewer_enabled:
            self._start_viewer()

    def render(self, result, mission_snapshot, current_sim_time=None):
        if result is None:
            return
        if not self.enabled:
            return
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return
        image = result.frame.rgb.copy()
        visual_detections = getattr(result, "visual_detections", ()) or ()
        for detection in visual_detections:
            colour = np.array((255, 150, 0), dtype=np.uint8)
            mask = detection.mask
            image[mask] = (
                0.55 * image[mask].astype(np.float32)
                + 0.45 * colour
            ).astype(np.uint8)
        for detection in result.detections:
            colour = np.array((255, 30, 30), dtype=np.uint8)
            mask = detection.mask
            image[mask] = (
                0.45 * image[mask].astype(np.float32)
                + 0.55 * colour
            ).astype(np.uint8)
        header_height = 116
        canvas = Image.new(
            "RGB",
            (image.shape[1], image.shape[0] + header_height),
            (245, 245, 245),
        )
        canvas.paste(Image.fromarray(image), (0, header_height))
        draw = ImageDraw.Draw(canvas)
        for index, detection in enumerate(visual_detections, start=1):
            x1, y1, x2, y2 = detection.box_xyxy
            box_width_px, box_height_px = self._box_dimensions_px(
                detection.box_xyxy
            )
            y1 += header_height
            y2 += header_height
            draw.rectangle(
                (x1, y1, x2, y2),
                outline=(255, 150, 0),
                width=2,
            )
            range_text = (
                f"{detection.range_m:.2f}m"
                if math.isfinite(detection.range_m)
                else "depth?"
            )
            draw.text(
                (x1 + 4, y2 - 16),
                f"raw{index} {detection.confidence:.2f} {range_text} "
                f"box={box_width_px}x{box_height_px}px",
                fill=(255, 210, 0),
            )
        for index, detection in enumerate(result.detections, start=1):
            x1, y1, x2, y2 = detection.box_xyxy
            box_width_px, box_height_px = self._box_dimensions_px(
                detection.box_xyxy
            )
            y1 += header_height
            y2 += header_height
            draw.rectangle(
                (x1, y1, x2, y2),
                outline=(255, 30, 30),
                width=3,
            )
            draw.text(
                (x1 + 4, y1 + 4),
                f"d{index} {detection.confidence:.2f} "
                f"{detection.range_m:.2f}m "
                f"box={box_width_px}x{box_height_px}px",
                fill=(255, 255, 0),
            )
        draw.text(
            (10, 8),
            f"{self.robot_id} victim detector | state={mission_snapshot.get('state', '-')}",
            fill=(0, 0, 0),
        )
        draw.text(
            (10, 28),
            f"frame={result.frame.frame_id} rgbd={len(result.detections)} "
            f"raw={len(visual_detections)} "
            f"inference={result.inference_s:.3f}s "
            f"age={self._result_age_text(result, current_sim_time)} "
            f"error={result.error or '-'}",
            fill=(0, 0, 0),
        )
        y = 48
        for index, detection in enumerate(result.detections[:3], start=1):
            draw.text(
                (10, y),
                f"d{index}: conf={detection.confidence:.2f} "
                f"range={detection.range_m:.2f}m "
                f"world=({detection.world_position[0]:.2f},"
                f"{detection.world_position[1]:.2f}) "
                f"depth={detection.valid_depth_pixels}px/"
                f"{detection.valid_depth_fraction:.0%}",
                fill=(160, 0, 0),
            )
            y += 18
        if not result.detections and visual_detections:
            for index, detection in enumerate(visual_detections[:3], start=1):
                range_text = (
                    f"{detection.range_m:.2f}m"
                    if math.isfinite(detection.range_m)
                    else "inf/invalid"
                )
                draw.text(
                    (10, y),
                    f"raw{index}: conf={detection.confidence:.2f} "
                    f"range={range_text} "
                    f"depth={detection.valid_depth_pixels}px/"
                    f"{detection.valid_depth_fraction:.0%} "
                    f"{detection.depth_status}",
                    fill=(170, 90, 0),
                )
                y += 18
        close_in = mission_snapshot.get("close_in_target")
        close_text = "close-in=-"
        if close_in is not None:
            close_text = f"close-in=({close_in[0]:.2f},{close_in[1]:.2f})"
        reportable_time = mission_snapshot.get("reportable_time")
        reportable_age = "-"
        if reportable_time is not None and current_sim_time is not None:
            reportable_age = f"{max(0.0, current_sim_time - reportable_time):.1f}s"
        draw.text(
            (10, header_height - 16),
            f"{close_text} reportable="
            f"{mission_snapshot.get('reportable_confidence', 0.0):.2f}/"
            f"{reportable_age} committed="
            f"{mission_snapshot.get('close_in_committed_confidence', 0.0):.2f}",
            fill=(120, 0, 140),
        )
        tmp = self.output_path.with_suffix(".tmp.png")
        canvas.save(tmp)
        os.replace(tmp, self.output_path)

    def render_status(self, status, detail=""):
        if not self.enabled:
            return
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return
        canvas = Image.new("RGB", (640, 160), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (10, 10),
            f"{self.robot_id} victim detector",
            fill=(0, 0, 0),
        )
        draw.text((10, 36), str(status), fill=(160, 0, 0))
        if detail:
            draw.text((10, 62), str(detail)[:100], fill=(80, 80, 80))
        tmp = self.output_path.with_suffix(".tmp.png")
        canvas.save(tmp)
        os.replace(tmp, self.output_path)

    @staticmethod
    def _result_age_text(result, current_sim_time):
        if current_sim_time is None:
            return "-"
        age_s = max(0.0, float(current_sim_time) - result.frame.sim_time)
        return f"{age_s:.2f}s"

    @staticmethod
    def _box_dimensions_px(box_xyxy):
        x1, y1, x2, y2 = box_xyxy
        return (
            max(0, int(round(float(x2) - float(x1)))),
            max(0, int(round(float(y2) - float(y1)))),
        )

    def _start_viewer(self):
        script = LIVE_MAP_VIEWER_PATH
        if not script.exists():
            return
        try:
            self._viewer_process = subprocess.Popen(
                [
                    sys.executable,
                    str(script),
                    str(self.output_path),
                    f"{self.robot_id} victim detection",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            atexit.register(self.stop)
        except OSError:
            self.enabled = False

    def stop(self):
        if self._viewer_process and self._viewer_process.poll() is None:
            self._viewer_process.terminate()
