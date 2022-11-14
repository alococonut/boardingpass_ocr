import os
import uuid
import cv2
from asgi_correlation_id.context import correlation_id



def resize_frame(frame, desired_width, desired_height):
    h, w = frame.shape[:2]
    if w < desired_width:
        # Interpolate up
        resized_frame = cv2.resize(frame, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)
    else:
        # Interpolate down
        resized_frame = cv2.resize(frame, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    return resized_frame




def save_debug(file_name, file):
    debug_path = os.path.join(os.getcwd(), 'data', 'debug')
    if not os.path.exists(debug_path):
        os.makedirs(debug_path)
    x_request_id = correlation_id.get() or uuid.uuid4().hex
    cv2.imwrite(os.path.join(debug_path, f'{file_name}_{x_request_id}.png'), file)
