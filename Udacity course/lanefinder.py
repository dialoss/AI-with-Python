import cv2
import numpy as np
import math
from PIL import ImageGrab, Image, ImageEnhance
import matplotlib.pyplot as plt

n = int(input())

prev_ll = np.array([[0, 0, 0, 0]] * n)
prev_rl = np.array([[0, 0, 0, 0]] * n)
prev_dl = np.array([[0, 0, 0, 0]] * n)
prev_dr = np.array([[0, 0, 0, 0]] * n)

bottom = 720
top = int(bottom * (2 / 5) / n)


def make_coords(image, line_params, loop):
    slope, intercept = line_params
    y1 = bottom - top * loop
    y2 = bottom - top * (loop + 1)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def asi(image, lines):
    global prev_ll, prev_rl
    left_fit = []
    right_fit = []
    left_line = np.array([[0, 0, 0, 0]] * n)
    right_line = np.array([[0, 0, 0, 0]] * n)
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if abs(x1 - x2) != 0 and abs(y1 - y2) / abs(x1 - x2) > 0.2:
            params = np.polyfit((x1, x2), (y1, y2), 1)
            slope = params[0]
            intercept = params[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            elif slope > 0:
                right_fit.append((slope, intercept))
    left_fit.sort(key=lambda x: x[1])
    right_fit.sort(key=lambda x: x[1])
    x = len(left_fit) // n
    y = len(right_fit) // n
    lfa = []
    rfa = []
    for i in range(n):
        lf = np.average(left_fit[i * x:(i + 1) * x + 1], axis=0)
        rf = np.average(right_fit[i * y:(i + 1) * y + 1], axis=0)
        try:
            x = len(lf)
        except:
            lf = [0, 0]
        try:
            x = len(rf)
        except:
            rf = [0, 0]
        lfa.append(lf)
        rfa.append(rf)
    lfa = np.array(lfa)
    rfa = np.array(rfa)
    for i in range(n):
        try:
            left_line[i] = make_coords(image, lfa[i], i)
            prev_ll[i] = np.copy(left_line[i])
        except:
            left_line[i] = prev_ll[i]
        try:
            right_line[i] = make_coords(image, rfa[i], i)
            prev_rl[i] = np.copy(right_line[i])
        except:
            right_line[i] = prev_rl[i]
    exc = []
    for i in range(n):
        exc.append([left_line[i], right_line[i]])
    return np.array(exc)


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def display(image, lines):
    global prev_dl, prev_dr
    line_image = np.zeros_like(image)
    for j in range(len(lines)):
        for i in range(2):
            x1, y1, x2, y2 = lines[j][i].reshape(4)
            if abs(x1 - x2) != 0 and abs(y1 - y2) / abs(x1 - x2) > 0.3:
                if i == 0:
                    prev_dl[j] = lines[j][i]
                else:
                    prev_dr[j] = lines[j][i]
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
            elif i == 0:
                x1, y1, x2, y2 = prev_dl[j].reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
            else:
                x1, y1, x2, y2 = prev_dr[j].reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    # for j in range(len(lines)):
    #     for i in range(2):
    #         x1, y1, x2, y2 = lines[0].reshape(4)
    #         x3, y3, x4, y4 = lines[1].reshape(4)
    #         m1 = abs(x3 - x1) // 5
    #         m2 = abs(x4 - x2) // 5
    #         lx1 = x1 + 2 * m1
    #         rx1 = x3 - 2 * m1
    #         lx2 = x2 + 2 * m2
    #         rx2 = x4 - 2 * m2
    #         pts = np.array([[lx1, y1], [lx2, y2], [rx2, y4], [rx1, y3]])
    #         pts = pts.reshape((-1, 1, 2))
    #         cv2.fillPoly(line_image, [pts], (0, 140, 0))
    return line_image


def region(image):
    height = image.shape[0]
    triangle = np.array([[(200, 600), (1000, 600), (650, 300)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked = cv2.bitwise_and(image, mask)
    return masked


def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    processed_img = cv2.GaussianBlur(processed_img, (3, 3), 0)
    return processed_img


def main():
    window_name = "MyScreen"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 1200, 700)))
        new_screen = process_img(screen)
        cv2.imshow(window_name, new_screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


while True:
    image = np.array(ImageGrab.grab(bbox=(0, 40, 1200, 700)))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lane_image = np.copy(image)
    canny_img = canny(lane_image)
    cropped = region(canny_img)
    lines = cv2.HoughLinesP(cropped, 2, np.pi / 180, 100, np.array([]), minLineLength=50, maxLineGap=90)
    if lines is not None:
        averaged = asi(lane_image, lines)
        line_image = display(lane_image, averaged)
        final = cv2.addWeighted(image, 0.8, line_image, 1, 1)
        cv2.imshow('w', final)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
