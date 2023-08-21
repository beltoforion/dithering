import os
import cv2
import numpy as np
import imageio

from numba import jit
from enum import Enum

class Colors(Enum):
    Monochrome = 1
    ThreeColors = 2
    EightColors = 3


def dither_fs2(gray_img):
#    preprocessed_img = cv2.equalizeHist(gray_img)
#    preprocessed_img = cv2.normalize(preprocessed_img, None, 0, 255, cv2.NORM_MINMAX)
    preprocessed_img = gray_img
#    return dither_fs2_core(preprocessed_img), preprocessed_img
    return dither_burkes_core(preprocessed_img), preprocessed_img

@jit(nopython=True)
def dither_burkes_core(gray_img):
    height, width = gray_img.shape

    for y in range(height):
        for x in range(width):
            old_pixel = gray_img[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            gray_img[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            # Distribute the quantization error to the neighboring pixels
            if x + 1 < width:
                gray_img[y, x + 1] += quant_error * 8 / 32
            if x + 2 < width:
                gray_img[y, x + 2] += quant_error * 4 / 32
            if x - 2 >= 0 and y + 1 < height:
                gray_img[y + 1, x - 2] += quant_error * 2 / 32
            if x - 1 >= 0 and y + 1 < height:
                gray_img[y + 1, x - 1] += quant_error * 4 / 32
            if y + 1 < height:
                gray_img[y + 1, x] += quant_error * 8 / 32
            if x + 1 < width and y + 1 < height:
                gray_img[y + 1, x + 1] += quant_error * 4 / 32
            if x + 2 < width and y + 1 < height:
                gray_img[y + 1, x + 2] += quant_error * 2 / 32

    return np.clip(gray_img, 0, 255).astype(np.uint8)


@jit(nopython=True)
def dither_atkinson_core(gray_img):
    height, width = gray_img.shape

    for y in range(height):
        for x in range(width):
            old_pixel = gray_img[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            gray_img[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            # Distribute the quantization error to the neighboring pixels
            if x + 1 < width:
                gray_img[y, x + 1] += quant_error * 1 / 8
            if x + 2 < width:
                gray_img[y, x + 2] += quant_error * 1 / 8
            if x - 1 >= 0 and y + 1 < height:
                gray_img[y + 1, x - 1] += quant_error * 1 / 8
            if y + 1 < height:
                gray_img[y + 1, x] += quant_error * 1 / 8
            if x + 1 < width and y + 1 < height:
                gray_img[y + 1, x + 1] += quant_error * 1 / 8
            if y + 2 < height:
                gray_img[y + 2, x] += quant_error * 1 / 8

    return np.clip(gray_img, 0, 255).astype(np.uint8)


@jit(nopython=True)
def dither_stucki_core(gray_img):
    height, width = gray_img.shape

    for y in range(height):
        for x in range(width):
            old_pixel = gray_img[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            gray_img[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            # Distribute the quantization error to the neighboring pixels, with rounding
            if x + 1 < width:
                gray_img[y, x + 1] += quant_error * 8 / 42
            if x + 2 < width:
                gray_img[y, x + 2] += quant_error * 4 / 42
            if x - 2 >= 0 and y + 1 < height:
                gray_img[y + 1, x - 2] += quant_error * 2 / 42
            if x - 1 >= 0 and y + 1 < height:
                gray_img[y + 1, x - 1] += quant_error * 4 / 42
            if y + 1 < height:
                gray_img[y + 1, x] += quant_error * 8 / 42
            if x + 1 < width and y + 1 < height:
                gray_img[y + 1, x + 1] += quant_error * 4 / 42
            if x + 2 < width and y + 1 < height:
                gray_img[y + 1, x + 2] += quant_error * 2 / 42
            if x - 2 >= 0 and y + 2 < height:
                gray_img[y + 2, x - 2] += quant_error * 1 / 42
            if x - 1 >= 0 and y + 2 < height:
                gray_img[y + 2, x - 1] += quant_error * 2 / 42
            if y + 2 < height:
                gray_img[y + 2, x] += quant_error * 4 / 42
            if x + 1 < width and y + 2 < height:
                gray_img[y + 2, x + 1] += quant_error * 2 / 42
            if x + 2 < width and y + 2 < height:
                gray_img[y + 2, x + 2] += quant_error * 1 / 42

    return np.clip(gray_img, 0, 255).astype(np.uint8)


@jit(nopython=True)
def dither_fs2_core(gray_img):
    height, width = gray_img.shape

    for y in range(height):
        for x in range(width):
            old_pixel = gray_img[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            gray_img[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            # Distribute the quantization error to the neighboring pixels, with rounding
            if x + 1 < width:
                gray_img[y, x + 1] = min(255, max(0, gray_img[y, x + 1] + int(quant_error * 7 / 16)))
            if x - 1 >= 0 and y + 1 < height:
                gray_img[y + 1, x - 1] = min(255, max(0, gray_img[y + 1, x - 1] + int(quant_error * 3 / 16)))
            if y + 1 < height:
                gray_img[y + 1, x] = min(255, max(0, gray_img[y + 1, x] + int(quant_error * 5 / 16)))
            if x + 1 < width and y + 1 < height:
                gray_img[y + 1, x + 1] = min(255, max(0, gray_img[y + 1, x + 1] + int(quant_error * 1 / 16)))

    return gray_img

def dither_bayer(gray_img):
    # Define the Bayer matrix
    bayer_matrix = np.array([[0, 2],
                             [3, 1]])

    input_img = gray_img
    height, width = input_img.shape

    bayer_img = np.tile(bayer_matrix, (height // 2 + 1, width // 2 + 1))
    bayer_img = bayer_img[:height, :width]

    processed_img = np.where(input_img > bayer_img * 255 / 4, 255, 0).astype(np.uint8)

    return processed_img, input_img


def dither_noise(gray_img):
    equalized_img = cv2.equalizeHist(gray_img)
    normalized_img = cv2.normalize(equalized_img, None, 0, 255, cv2.NORM_MINMAX)

    # Create the noise image once and reuse it
    is_static=False
    if is_static:
        if not hasattr(dither_noise, "noise_img"):
            dither_noise.noise_img = np.zeros_like(normalized_img)
            cv2.randn(dither_noise.noise_img, 150, 64)
    else:
        dither_noise.noise_img = np.zeros_like(normalized_img)
        cv2.randn(dither_noise.noise_img, 200, 20)

    third_img = np.where(normalized_img > dither_noise.noise_img, 255, 0).astype(np.uint8)
    return third_img, normalized_img


def rgbify(image):
    h, w, d = image.shape
    nh = int(h * 2)
    nw = int(w * 1.5)
    
    rgb = np.zeros((nh, nw, 3), dtype=np.uint8)

    for x in range(w):
        for y in range(h):
            # Read rgb values of two neighboring pixels from the input image
            b, g, r = image[y, x, :]

            if (x%2 == 0):
               ox = (int)(x*1.5)
               rgb[2*y + 0, ox,     :] = (0, g, 0)
               rgb[2*y + 0, ox + 1, :] = (b, 0, 0)
               rgb[2*y + 1, ox,     :] = (0, 0, r)
            else:
                ox = 1+(int)(x*1.5 - 0.5)
                rgb[2*y + 0, ox,     :] = (0,  0,  r)
                rgb[2*y + 1, ox,     :] = (b,  0,  0)
                rgb[2*y + 1, ox - 1, :] = (0,  g,  0)

    return rgb

            
def process_image_impl(img, colors):
    dither_func = dither_fs2
    if colors==Colors.Monochrome:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_img, grayscale_img = dither_func(gray_img)
        return processed_img
    elif colors==Colors.EightColors or colors==Colors.ThreeColors:
        if colors == Colors.ThreeColors:
            h, w, d = img.shape
            img = cv2.resize(img, (int(w/1.5), int(h/2)))

        ch_blue, ch_green, ch_red = cv2.split(img)
        ch_blue_dit, _ = dither_func(ch_blue)
        ch_green_dit, _ = dither_func(ch_green)
        ch_red_dit, _ = dither_func(ch_red)
        processed_img_col = cv2.merge((ch_blue_dit, ch_green_dit, ch_red_dit))

        if colors==Colors.EightColors:
            return processed_img_col
        elif colors==Colors.ThreeColors:
            rgb = rgbify(processed_img_col)
            return rgb


def process_image(input_path, colors):
    img = cv2.imread(input_path)

    out_img = process_image_impl(img, colors)

    output_path = os.path.splitext(input_path)[0] + "_" + str(colors) + ".png"
    cv2.imwrite(output_path, out_img)
    cv2.imshow("Image", out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def dither_noise_demo(gray_img):
    height, width = gray_img.shape

    for y in range(height):
        for x in range(width):
            old_pixel = gray_img[y, x]
#            new_pixel = 0 if old_pixel < random.randint(0, 255) else 255
            new_pixel = 0 if old_pixel < int(np.random.normal(127, 45)) else 255
            gray_img[y, x] = new_pixel

    return gray_img


def process_video(input_path, colors, video_scale = 1, save_gif=False):
    # Check if the input file is an mp4 file
    output_path = os.path.splitext(input_path)[0] + ("_out.gif" if save_gif else "_out.mp4")

    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    width = int(width*video_scale)
    height = int(height*video_scale)

    # Create a VideoWriter object to save the processed video
    if not save_gif:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    else:
        gif_writer = imageio.get_writer(output_path, mode='I', fps=fps)

    # Process each frame of the video
    fcount = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fcount += 1
        if fcount > 100:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (0, 0), fx=video_scale, fy=video_scale)
        processed_frame, grayscale_frame = dither_fs2(frame)
#        processed_frame, grayscale_frame = dither_noise(frame)

        if save_gif:
            gif_writer.append_data(processed_frame)
        else:
            out.write(processed_frame)

        # Display the processed frame
        cv2.imshow("Processed Frame (Downscaled)", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture and video writer objects
    if save_gif:
        gif_writer.close()
    else:
        cap.release()
        out.release()


# Process an mp4 file
process_video("sample3.mp4", Colors.Monochrome, 0.33, True)

#process_image("sample6.jpg", Colors.ThreeColors)