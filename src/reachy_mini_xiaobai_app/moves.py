"""Background motion executor — consumes the motion queue and drives the robot."""

import logging
import math
import queue
import threading
import time

import numpy as np
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini_dances_library import RecordedMoves

log = logging.getLogger(__name__)


class MovementExecutor:
    """Runs in a background thread, pulling commands from a :class:`queue.Queue`
    and executing them on the robot.

    Supported command types (``cmd["type"]``):

    * ``goto``      — move head to *yaw* / *pitch* over *duration*
    * ``nod``       — nod *times* times
    * ``shake``     — shake head *times* times
    * ``emotion``   — play a named emotion from the dances library
    * ``antennas``  — wiggle antennas for *duration* seconds
    """

    def __init__(
        self,
        mini: ReachyMini,
        motion_queue: "queue.Queue[dict]",
        stop_event: threading.Event,
    ) -> None:
        self._mini = mini
        self._queue = motion_queue
        self._stop = stop_event
        self._recorded_moves = RecordedMoves("pollen-robotics/reachy-mini-dances-library")
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        log.info("MovementExecutor started")
        while not self._stop.is_set():
            try:
                cmd = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._dispatch(cmd)
            except Exception:
                log.exception("Error executing motion command %s", cmd)
            finally:
                self._queue.task_done()

    def _dispatch(self, cmd: dict) -> None:
        cmd_type = cmd.get("type")
        if cmd_type == "goto":
            self._do_goto(cmd)
        elif cmd_type == "nod":
            self._do_nod(cmd)
        elif cmd_type == "shake":
            self._do_shake(cmd)
        elif cmd_type == "emotion":
            self._do_emotion(cmd)
        elif cmd_type == "antennas":
            self._do_antennas(cmd)
        else:
            log.warning("Unknown motion command type: %s", cmd_type)

    def _do_goto(self, cmd: dict) -> None:
        yaw = math.radians(cmd.get("yaw", 0.0))
        pitch = math.radians(cmd.get("pitch", 0.0))
        duration = cmd.get("duration", 0.5)
        head_pose = create_head_pose(roll=0, pitch=pitch, yaw=yaw)
        log.info("goto yaw=%.1f° pitch=%.1f° dur=%.2fs", cmd.get("yaw", 0), cmd.get("pitch", 0), duration)
        self._mini.goto_target(head=head_pose, duration=duration)
        time.sleep(duration)

    def _do_nod(self, cmd: dict) -> None:
        times = cmd.get("times", 1)
        log.info("nod ×%d", times)
        nod_angle = math.radians(15)
        for _ in range(times):
            self._mini.goto_target(
                head=create_head_pose(roll=0, pitch=-nod_angle, yaw=0), duration=0.25
            )
            time.sleep(0.25)
            self._mini.goto_target(
                head=create_head_pose(roll=0, pitch=nod_angle, yaw=0), duration=0.25
            )
            time.sleep(0.25)
        # Return to neutral
        self._mini.goto_target(
            head=create_head_pose(roll=0, pitch=0, yaw=0), duration=0.2
        )
        time.sleep(0.2)

    def _do_shake(self, cmd: dict) -> None:
        times = cmd.get("times", 1)
        log.info("shake ×%d", times)
        shake_angle = math.radians(20)
        for _ in range(times):
            self._mini.goto_target(
                head=create_head_pose(roll=0, pitch=0, yaw=-shake_angle), duration=0.2
            )
            time.sleep(0.2)
            self._mini.goto_target(
                head=create_head_pose(roll=0, pitch=0, yaw=shake_angle), duration=0.2
            )
            time.sleep(0.2)
        # Return to neutral
        self._mini.goto_target(
            head=create_head_pose(roll=0, pitch=0, yaw=0), duration=0.2
        )
        time.sleep(0.2)

    def _do_emotion(self, cmd: dict) -> None:
        name = cmd.get("name", "happy")
        log.info("emotion: %s", name)
        move = self._recorded_moves.get(name)
        if move is not None:
            self._mini.play_move(move)
        else:
            log.warning("Unknown emotion '%s', skipping", name)

    def _do_antennas(self, cmd: dict) -> None:
        duration = cmd.get("duration", 1.0)
        log.info("antennas wiggle %.1fs", duration)
        end_time = time.time() + duration
        toggle = True
        while time.time() < end_time and not self._stop.is_set():
            if toggle:
                self._mini.goto_target(antennas=np.deg2rad([30, -30]), duration=0.25)
            else:
                self._mini.goto_target(antennas=np.deg2rad([-30, 30]), duration=0.25)
            toggle = not toggle
            time.sleep(0.25)
        # Return to neutral
        self._mini.goto_target(antennas=np.deg2rad([0, 0]), duration=0.2)
        time.sleep(0.2)
