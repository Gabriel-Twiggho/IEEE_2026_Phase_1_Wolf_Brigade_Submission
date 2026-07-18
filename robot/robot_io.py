"""Webots IO boundary for the flat controller architecture."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class SensorSnapshot:
    """One timestep of raw sensor data copied out of Webots devices."""

    robot_id: str = ""
    sim_time_s: float = 0.0
    timestep_ms: int = 0
    wheel_angles: dict = None
    compass: tuple = None
    accelerometer: tuple = None
    gyro: tuple = None
    lidar_ranges: tuple = ()
    lidar_fov: float = 2.0 * math.pi
    ir_ranges: dict = None
    rgb_width: int = 0
    rgb_height: int = 0
    depth_width: int = 0
    depth_height: int = 0
    depth_min_range: float = 0.0
    depth_max_range: float = 0.0
    depth_fov: float = 0.0
    depth_range_image: tuple = ()


class RobotIO:
    """
    Thin adapter around Webots devices.

    It centralises device names, sensor snapshots, keyboard polling, and the
    final motor write path. Higher-level modules decide what the robot should
    do; this class only talks to Webots devices.
    """

    MOTOR_NAMES = {
        "fl": "fl_wheel_joint",
        "fr": "fr_wheel_joint",
        "rl": "rl_wheel_joint",
        "rr": "rr_wheel_joint",
    }
    ENCODER_NAMES = {
        "fl": "front left wheel motor sensor",
        "fr": "front right wheel motor sensor",
        "rl": "rear left wheel motor sensor",
        "rr": "rear right wheel motor sensor",
    }
    RANGE_SENSOR_NAMES = ("fl_range", "fr_range", "rl_range", "rr_range")

    def __init__(self, robot):
        self.robot = robot
        self.robot_id = robot.getName()
        self.timestep = int(robot.getBasicTimeStep())
        self.motors = {}
        self.encoders = {}
        self.range_sensors = {}
        self.compass = None
        self.accelerometer = None
        self.gyro = None
        self.lidar = None
        self.camera_rgb = None
        self.camera_depth = None
        self.team_emitter = None
        self.team_receiver = None
        self._init_devices()
        self._cache_static_metadata()

    def _init_devices(self):
        # Motors are acquired here so all wheel output goes through write_motors().
        for key, name in self.MOTOR_NAMES.items():
            try:
                motor = self.robot.getDevice(name)
                motor.setPosition(float("inf"))
                motor.setVelocity(0.0)
                self.motors[key] = motor
            except Exception as exc:
                print(f"[{self.robot_id}] Warning: motor {name} unavailable: {exc}")

        for key, name in self.ENCODER_NAMES.items():
            encoder = self.robot.getDevice(name)
            encoder.enable(self.timestep)
            self.encoders[key] = encoder

        self.compass = self.robot.getDevice("imu compass")
        self.compass.enable(self.timestep)
        self.accelerometer = self._optional_sensor("imu accelerometer")
        self.gyro = self._optional_sensor("imu gyro")
        self.lidar = self._optional_lidar()
        self.camera_rgb = self._optional_camera("camera rgb")
        self.camera_depth = self._optional_camera("camera depth")
        self.team_emitter = self._optional_device("robot to robot emitter")
        self.team_receiver = self._optional_device("robot to robot receiver")
        if self.team_receiver is not None:
            self.team_receiver.enable(self.timestep)

        for name in self.RANGE_SENSOR_NAMES:
            sensor = self.robot.getDevice(name)
            sensor.enable(self.timestep)
            self.range_sensors[name] = sensor

    def _optional_device(self, name):
        try:
            return self.robot.getDevice(name)
        except Exception as exc:
            print(f"[{self.robot_id}] Warning: {name} not found: {exc}")
            return None

    def _cache_static_metadata(self):
        self.lidar_fov = 2.0 * math.pi
        if self.lidar is not None:
            try:
                self.lidar_fov = float(self.lidar.getFov())
            except AttributeError:
                self.lidar_fov = 2.0 * math.pi

        self.rgb_width = self._dimension(self.camera_rgb, "getWidth")
        self.rgb_height = self._dimension(self.camera_rgb, "getHeight")
        self.depth_width = self._dimension(self.camera_depth, "getWidth")
        self.depth_height = self._dimension(self.camera_depth, "getHeight")
        self.depth_min_range = self._dimension(self.camera_depth, "getMinRange", 0.0)
        self.depth_max_range = self._dimension(self.camera_depth, "getMaxRange", 0.0)
        self.depth_fov = self._dimension(self.camera_depth, "getFov", 0.0)

    def _optional_sensor(self, name):
        try:
            sensor = self.robot.getDevice(name)
            sensor.enable(self.timestep)
            return sensor
        except Exception as exc:
            print(f"[{self.robot_id}] Warning: {name} not found: {exc}")
            return None

    def _optional_lidar(self):
        try:
            lidar = self.robot.getDevice("laser")
        except Exception as exc:
            print(f"[{self.robot_id}] Warning: lidar device not found: {exc}")
            return None
        lidar.enable(self.timestep)
        return lidar

    def _optional_camera(self, name):
        try:
            camera = self.robot.getDevice(name)
            camera.enable(self.timestep)
            return camera
        except Exception as exc:
            print(f"[{self.robot_id}] Warning: {name} not found: {exc}")
            return None

    def read_snapshot(
        self,
        sim_time_s: float = 0.0,
        include_depth: bool = True,
        include_lidar: bool = True,
        include_ir: bool = True,
        include_static_metadata: bool = False,
    ):
        wheel_angles = {
            key: sensor.getValue()
            for key, sensor in self.encoders.items()
        }
        ir_ranges = {}
        if include_ir:
            ir_ranges = {
                name: sensor.getValue()
                for name, sensor in self.range_sensors.items()
            }
        lidar_ranges = ()
        if include_lidar:
            lidar_ranges, _lidar_fov = self.read_lidar_ranges()

        depth_range_image = ()
        if include_depth and self.camera_depth is not None:
            try:
                depth_range_image = tuple(self.camera_depth.getRangeImage())
            except AttributeError:
                depth_range_image = ()

        return SensorSnapshot(
            robot_id=self.robot_id,
            sim_time_s=float(sim_time_s),
            timestep_ms=self.timestep,
            wheel_angles=wheel_angles,
            compass=self._values(self.compass),
            accelerometer=self._values(self.accelerometer),
            gyro=self._values(self.gyro),
            lidar_ranges=lidar_ranges,
            lidar_fov=self.lidar_fov,
            ir_ranges=ir_ranges,
            rgb_width=self.rgb_width if include_static_metadata else 0,
            rgb_height=self.rgb_height if include_static_metadata else 0,
            depth_width=self.depth_width if include_static_metadata else 0,
            depth_height=self.depth_height if include_static_metadata else 0,
            depth_min_range=self.depth_min_range if include_static_metadata else 0.0,
            depth_max_range=self.depth_max_range if include_static_metadata else 0.0,
            depth_fov=self.depth_fov if include_static_metadata else 0.0,
            depth_range_image=depth_range_image,
        )

    def enable_keyboard(self):
        keyboard = self.robot.getKeyboard()
        keyboard.enable(self.timestep)
        return keyboard

    def read_lidar_ranges(self):
        if self.lidar is None:
            return (), self.lidar_fov
        return tuple(self.lidar.getRangeImage()), self.lidar_fov

    def read_pressed_keys(self, keyboard):
        pressed = set()
        while True:
            key = keyboard.getKey()
            if key == -1:
                break
            pressed.add(key)
        return pressed

    def write_motors(self, left_rad_s: float, right_rad_s: float):
        for key in ("fl", "rl"):
            if key in self.motors:
                self.motors[key].setVelocity(left_rad_s)
        for key in ("fr", "rr"):
            if key in self.motors:
                self.motors[key].setVelocity(right_rad_s)

    def send_team_message(self, payload):
        if self.team_emitter is None:
            return False
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        elif isinstance(payload, bytearray):
            payload = bytes(payload)
        self.team_emitter.send(payload)
        return True

    def poll_team_messages(self):
        messages = []
        if self.team_receiver is None:
            return messages
        while self.team_receiver.getQueueLength() > 0:
            if hasattr(self.team_receiver, "getString"):
                messages.append(self.team_receiver.getString())
            else:
                messages.append(self.team_receiver.getData())
            self.team_receiver.nextPacket()
        return messages

    def read_rgb_image(self):
        if self.camera_rgb is None:
            return None
        raw = self.camera_rgb.getImage()
        if raw is None:
            return None
        try:
            import numpy as np

            image = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.rgb_height, self.rgb_width, 4)
            )
            # Webots camera bytes are BGRA; expose RGB to the controller.
            return image[:, :, :3][:, :, ::-1].copy()
        except Exception:
            return None

    def read_depth_image(self):
        if self.camera_depth is None:
            return None
        try:
            import numpy as np

            return np.array(
                self.camera_depth.getRangeImage(),
                dtype=np.float32,
            ).reshape((self.depth_height, self.depth_width))
        except Exception:
            return None

    def compass_yaw(self):
        values = self.compass.getValues()
        return math.atan2(values[0], values[1])

    @staticmethod
    def _values(device):
        if device is None:
            return None
        return tuple(device.getValues())

    @staticmethod
    def _dimension(device, method_name: str, default=0):
        if device is None:
            return default
        try:
            return getattr(device, method_name)()
        except AttributeError:
            return default
