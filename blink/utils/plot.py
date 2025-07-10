import cv2

def plot_points(image, points):
    ret_image = image.copy()
    for point in points:
        cv2.circle(ret_image, point, 1, (0, 255, 0), 1)

    return ret_image

