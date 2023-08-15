import os
import cv2
import PIL
import numpy as np
from numba import jit

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

    equalized_img = cv2.equalizeHist(gray_img)
    normalized_img = cv2.normalize(equalized_img, None, 0, 255, cv2.NORM_MINMAX)

    height, width = normalized_img.shape

    bayer_img = np.tile(bayer_matrix, (height // 2 + 1, width // 2 + 1))
    bayer_img = bayer_img[:height, :width]

    processed_img = np.where(normalized_img > bayer_img * 255 / 4, 255, 0).astype(np.uint8)

    return processed_img, normalized_img

def dither_noise(gray_img, is_static=False):
    equalized_img = cv2.equalizeHist(gray_img)
    normalized_img = cv2.normalize(equalized_img, None, 0, 255, cv2.NORM_MINMAX)

    # Create the noise image once and reuse it
    if is_static:
        if not hasattr(dither_noise, "noise_img"):
            dither_noise.noise_img = np.zeros_like(normalized_img)
            cv2.randn(dither_noise.noise_img, 128, 50)
    else:
        dither_noise.noise_img = np.zeros_like(normalized_img)
        cv2.randn(dither_noise.noise_img, 128, 50)

    third_img = np.where(normalized_img > dither_noise.noise_img, 255, 0).astype(np.uint8)
    return third_img, normalized_img

def process(input_path):

    # Check if the input file is an mp4 file
    if input_path.endswith(".mp4"):
        output_path = os.path.splitext(input_path)[0] + "_out.mp4"

        # Open the video file
        cap = cv2.VideoCapture(input_path)

        # Get the frames per second (fps) of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_scale = 1
        width = int(width*video_scale)
        height = int(height*video_scale)

        # Create a VideoWriter object to save the processed video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

        # Process each frame of the video
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (0, 0), fx=video_scale, fy=video_scale)
            processed_frame, grayscale_frame = dither_fs2(frame)
            out.write(processed_frame)

            # Display the processed frame
            cv2.imshow("Processed Frame (Downscaled)", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Release the video capture and video writer objects
        cap.release()
        out.release()

    # Check if the input file is a jpeg image
    elif input_path.endswith(".jpg") or input_path.endswith(".jpeg") or input_path.endswith(".png"):
        img = cv2.imread(input_path)

        ch_blue, ch_green, ch_red = cv2.split(img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dither_func = dither_fs2
        ch_blue_dit, _ = dither_func(ch_blue)
        ch_green_dit, _ = dither_func(ch_green)
        ch_red_dit, _ = dither_func(ch_red)

        output_path = os.path.splitext(input_path)[0] + "_out_col.png"
        processed_img_col = cv2.merge((ch_blue_dit, ch_green_dit, ch_red_dit))
        cv2.imwrite(output_path, processed_img_col)
        cv2.imshow("Processed Image", processed_img_col)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        output_path = os.path.splitext(input_path)[0] + "_out_bw.png"
        processed_img, grayscale_img = dither_func(gray_img)
        cv2.imwrite(output_path, processed_img.astype(int), [cv2.IMWRITE_PNG_BILEVEL, 1])
        
        cv2.imshow("Processed Image", processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # If the input file is not an mp4 file or a jpeg image, raise an error
    else:
        raise ValueError("Input file must be an mp4 file or a jpeg image")

# Process an mp4 file
process("sample3.mp4")

# Process a jpeg image
#process("sample1.jpg")