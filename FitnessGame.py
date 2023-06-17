from dataclasses import dataclass, field
from random import randint
from typing import NamedTuple, Tuple
from time import time
import cv2
import numpy as np
import mediapipe as mp


@dataclass
class PoseDetector:
    static_image_mode: bool = False
    model_complexity: int = 1
    smooth_landmarks: bool = True
    enable_segmentation: bool = False
    smooth_segmentation: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    def __post_init__(self) -> None:
        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def process_image(self, image: np.array) -> Tuple[np.array, NamedTuple]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = self.pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image, results

    def get_landmarks(self, image: np.array, pose_results: NamedTuple, draw=True) -> list:
        landmarks = []

        if pose_results.pose_landmarks:
            h, w, _ = image.shape
            if draw:
                self.mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            for lm in pose_results.pose_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks.append([x, y, lm.z])

        return landmarks


@dataclass
class Obstacle:
    x1: int
    y1: int
    width1: int
    height1: int
    x2: int
    y2: int
    width2: int
    height2: int
    color: Tuple[int, int, int] = (255, 0, 255)
    __point_counted: bool = False

    def draw_obstacle(self, image: np.array) -> None:
        cv2.rectangle(image, (self.x1, self.y1), (self.x1 + self.width1, self.y1 + self.height1),
                      self.color, -1)
        cv2.rectangle(image, (self.x2, self.y2), (self.x2 + self.width2, self.y2 + self.height2),
                      self.color, -1)

    def move(self, x: int) -> None:
        self.x1 -= x
        self.x2 -= x

    def check_if_point(self, center: Tuple[int, int]) -> bool:
        area = np.array([
            [self.x1, self.y1 + self.height1],
            [self.x1 + self.width1, self.y1 + self.height1],
            [self.x2 + self.width2, self.y2 + self.height2],
            [self.x2, self.y2 + self.height2],
        ])

        result = cv2.pointPolygonTest(area, center, False)
        if result != -1:
            self.color = (0, 200, 0)

            if not self.__point_counted:
                self.__point_counted = True
                return True
        else:
            self.color = (255, 0, 255)
            return False

    def check_collision(self, center: Tuple[int, int]) -> bool:
        upper_area = np.array([
            [self.x1, self.y1],
            [self.x1 + self.width1, self.y1],
            [self.x1 + self.width1, self.y1 + self.height1],
            [self.x1, self.y1 + self.height1],
        ])

        lower_area = np.array([
            [self.x2, self.y2],
            [self.x2 + self.width2, self.y2],
            [self.x2 + self.width2, self.y2 + self.height2],
            [self.x2, self.y2 + self.height2],
        ])

        l_area_col = cv2.pointPolygonTest(lower_area, center, False)
        u_area_col = cv2.pointPolygonTest(upper_area, center, False)

        if l_area_col != -1 or u_area_col != -1:
            return True
        else:
            return False


@dataclass
class FitnessGame:
    hole_height_range: Tuple[int, int] = (60, 120)
    obstacle_height_range: Tuple[int, int] = (50, 350)
    __first_run: bool = field(default=True, init=False)
    __scores: int = field(default=0, init=False)
    __frame_count: int = field(default=0, init=False)
    __obstacles_list: list = field(default_factory=lambda: [], init=False)
    __countdown_screen: bool = field(default=False, init=False)
    __countdown_start_time: float = field(default=-1, init=False)
    __countdown_lock: bool = field(default=-False, init=False)
    __paused: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.__cap = cv2.VideoCapture(0)
        self.__pose_detector = PoseDetector()

    def reset_variables(self) -> None:
        self.__obstacles_list = []
        self.__frame_count = 0

    def countdown(self, image: np.array) -> bool:
        if self.__countdown_screen:
            self.__countdown_lock = True
            if self.__countdown_start_time == -1:
                self.__countdown_start_time = time()

            time_diff = int(time() - self.__countdown_start_time)
            cv2.putText(image, f"{time_diff+1}", (270, 270), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 200), 12)
            if time_diff >= 3:
                self.__countdown_screen = False
                self.__countdown_lock = False
                self.__countdown_start_time = -1
                return True
            return False
        return True

    def menu_screen(self, image: np.array) -> None:
        if not self.__first_run:
            cv2.putText(image, f"You scored: {self.__scores} points", (160, 200), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

        cv2.putText(image, "Press 's' to start", (190, 240), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
        self.instruction_text(image)

        self.reset_variables()

    @staticmethod
    def instruction_text(image: np.array) -> None:
        cv2.putText(image, "Keep push up position around 0.5m", (20, 300), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
        cv2.putText(image, "from camera", (200, 340), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

    def game(self) -> None:
        frame = None
        game_start = False
        speed = 3
        frame_spawn_obstacle = 160
        prev_score = 0
        next_level_thr = 15
        p_time = 0

        while self.__cap.isOpened():
            if not self.__paused:
                ret, frame = self.__cap.read()
                if not ret:
                    break

            if not game_start:
                self.menu_screen(frame)
            elif self.__paused:
                cv2.putText(frame, "Press 'p' to resume", (170, 250), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
                self.instruction_text(frame)
            elif not self.__paused and game_start:
                if self.countdown(frame):
                    cv2.putText(frame, f"Score: {self.__scores}", (15, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
                    self.__first_run = False
                    img_h, img_w, _ = frame.shape
                    frame, results = self.__pose_detector.process_image(frame)

                    lms = self.__pose_detector.get_landmarks(frame, results)
                    if lms:
                        height_diff = abs(lms[23][1] - lms[11][1])
                        # controller = lms[0]
                        # cv2.circle(frame, (controller[0], controller[1]), 5, (0, 200, 0), -1)
                        if height_diff < 190 and lms[26][2] > 0 and lms[25][2] > 0:
                            controller = lms[0]
                            cv2.circle(frame, (controller[0], controller[1]), 5, (0, 200, 0), -1)
                        else:
                            controller = None
                            self.instruction_text(frame)
                    else:
                        self.__paused = not self.__paused
                        controller = None
                        if self.__paused:
                            self.__countdown_screen = True

                    if self.__frame_count % frame_spawn_obstacle == 0:
                        obs_h1 = randint(self.obstacle_height_range[0], self.obstacle_height_range[1])
                        obs_h2 = img_h - obs_h1 - randint(self.hole_height_range[0], self.hole_height_range[1])
                        obs = Obstacle(x1=img_w, y1=0, width1=50, height1=obs_h1,
                                       x2=img_w, y2=img_h, width2=50, height2=-obs_h2)

                        self.__obstacles_list.append(obs)
                        self.__frame_count += 1  # to prevent creating new obstacles when pose conditions are not met

                    obstacles_list_copy = self.__obstacles_list.copy()

                    if controller is not None:
                        if speed > 8:
                            speed = 3
                            frame_spawn_obstacle = 150

                        if self.__scores % next_level_thr == 0 and prev_score != self.__scores:
                            speed += 1
                            frame_spawn_obstacle -= 15
                            prev_score = self.__scores

                        for obs in obstacles_list_copy:
                            obs.draw_obstacle(frame)

                            if obs.check_collision((controller[0], controller[1])):
                                game_start = False

                            score = obs.check_if_point((controller[0], controller[1]))
                            if score:
                                self.__scores += 1

                            if obs.x1 + obs.width1 < 0:
                                self.__obstacles_list.remove(obs)
                            obs.move(speed)

                        self.__frame_count += 1

            try:
                c_time = time()
                fps = int(1 / (c_time - p_time))
                p_time = c_time
                cv2.putText(frame, f"FPS: {fps}", (500, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 200, 200), 2)
            except ZeroDivisionError:
                pass

            cv2.imshow("res", frame)

            key = cv2.waitKey(10)
            if key == ord('q'):
                break
            elif key == ord('p'):
                if game_start and not self.__countdown_lock:
                    self.__paused = not self.__paused
                    if self.__paused:
                        self.__countdown_screen = True

            elif key == ord('s'):
                if not game_start:
                    self.__scores = 0
                    self.__countdown_screen = True
                game_start = True

        self.__cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    game = FitnessGame()
    game.game()
