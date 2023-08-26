import numpy as np
import cv2
import math

from enum import Enum
from numba import jit

from ocvl.processor.processor_base import ProcessorBase
from ocvl.processor.input_output import *


class DitherMethod(Enum):
    BAYER2 = 1
    BAYER4 = 2
    BAYER8 = 3
    NOISE = 4
    FLOYD_STEINBERG = 5
    STUCKI = 6
    ATKINSON = 7
    BURKES = 8


class DitherProcessor(ProcessorBase):
    def __init__(self):
        super(DitherProcessor, self).__init__("DitherProcessor")    
        self.__method = DitherMethod.STUCKI  

        self._add_input(Input(self))
        self._add_output(Output())

    @property
    def method(self):
        return self.__method

    @method.setter
    def method(self, value):
        self.__method = value

    def process(self):
        image = self._inputs[0].data.image

        if (self.__method==DitherMethod.STUCKI):
            output = DitherProcessor.__dither_core_stucki(image)
        elif (self.__method==DitherMethod.NOISE):
            output = DitherProcessor.__dither_core_noise(image)
        elif (self.__method==DitherMethod.BAYER2 or 
              self.__method==DitherMethod.BAYER4 or
              self.__method==DitherMethod.BAYER8):
            output = DitherProcessor.__dither_core_bayer(image, self.__method)
        elif (self.__method==DitherMethod.FLOYD_STEINBERG):
            output = DitherProcessor.__dither_core_floyd_steinberg(image)
        elif (self.__method==DitherMethod.ATKINSON):
            output = DitherProcessor.__dither_core_atkinson(image)
        elif (self.__method==DitherMethod.BURKES):
            output = DitherProcessor.__dither_core_burkes(image)
        else:
            raise NotImplementedError(f"Dither method {self.__method} is not implemented!")
        
        self._outputs[0].set(IoData(output))

    @staticmethod
    def __dither_core_bayer(gray_img, method):
        
        # Define the Bayer matrix
        if method==DitherMethod.BAYER2:
            bayer_matrix = np.array([[0, 2],
                                     [3, 1]])
            div = 4
        elif method==DitherMethod.BAYER4:
            bayer_matrix = np.array([[0, 8, 2, 10],
                                     [12, 4, 14, 6],
                                     [3, 11, 1, 9],
                                     [15, 7, 13, 5]])
            div = 16
        elif method==DitherMethod.BAYER8:                
            bayer_matrix = np.array([[0, 32, 8, 40, 2, 34, 10, 42],
                                     [48, 16, 56, 24, 50, 18, 58, 26],
                                     [12, 44, 4, 36, 14, 46, 6, 38],
                                     [60, 28, 52, 20, 62, 30, 54, 22],
                                     [3, 35, 11, 43, 1, 33, 9, 41],
                                     [51, 19, 59, 27, 49, 17, 57, 25],
                                     [15, 47, 7, 39, 13, 45, 5, 37],
                                     [63, 31, 55, 23, 61, 29, 53, 21]])
            div = 64

        input_img = gray_img
        height, width = input_img.shape

        bayer_img = np.tile(bayer_matrix, (height // (int)(math.sqrt(div)) + 1, width // (int)(math.sqrt(div)) + 1))
        bayer_img = bayer_img[:height, :width]

        processed_img = np.where(input_img > bayer_img * 255 / div, 255, 0).astype(np.uint8)

        return processed_img

    @staticmethod
    @jit(nopython=True)
    def __dither_core_burkes(gray_img):
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

    @staticmethod
    def __dither_core_noise(gray_img):
        # Create the noise image once and reuse it
        noise_img = np.zeros_like(gray_img)
        cv2.randn(noise_img, 128, 40)

        output_img = np.where(gray_img > noise_img, 255, 0).astype(np.uint8)
        return output_img
    
    @staticmethod
    @jit(nopython=True)
    def __dither_core_floyd_steinberg(gray_img):
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

    @staticmethod
    @jit(nopython=True)
    def __dither_core_atkinson(gray_img):
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

    @staticmethod
    @jit(nopython=True)
    def __dither_core_stucki(gray_img):
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