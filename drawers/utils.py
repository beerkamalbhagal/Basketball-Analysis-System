import cv2
import numpy as np
import sys
sys.path.append("../")
from utils import get_center_of_bbox, get_bbox_width  

def draw_triangle(frame, bbox, color):
    y = int(bbox[1])
    x,_ = get_center_of_bbox(bbox)

    triangle_points = np.array([
        [int(x), int(y)],
        [int(x - 10), int(y - 20)],
        [int(x + 10), int(y - 20)]
    ])

    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, color, 2)

    return frame

def draw_ellipse(frame, bbox, color, track_id=None):

    y2 = int(bbox[3])

    x_center,_ = get_center_of_bbox(bbox)

    x_center = int(x_center)
    y2 = int(y2)

    width = int(get_bbox_width(bbox))


    cv2.ellipse(
        frame,
        (x_center, y2),
        (width, int(0.35 * width)),
        0,
        -45,
        235,
        color,
        2,
        cv2.LINE_4
    )


    rectangle_width = 40
    rectangle_height = 20

    x1_rect = x_center - rectangle_width // 2
    y1_rect = (y2 - rectangle_height // 2) + 15
    x2_rect = x_center + rectangle_width // 2
    y2_rect = (y2 + rectangle_height // 2) + 15


    if track_id is not None:

        cv2.rectangle(
            frame,
            (int(x1_rect), int(y1_rect)),
            (int(x2_rect), int(y2_rect)),
            color,
            cv2.FILLED
        )

        cv2.putText(
            frame,
            str(track_id),
            (int(x1_rect + 12), int(y1_rect + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,0),
            2
        )


    return frame
