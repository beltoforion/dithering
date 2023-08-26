from ocvl.processor.processor_base import ProcessorBase
from ocvl.processor.input_output import *

import numpy as np
import cv2


class ScaleProcessor(ProcessorBase):
    def __init__(self):
        super(ScaleProcessor, self).__init__("ScaleProcessor")      
        self._scale = 1
        self.__interpolation = cv2.INTER_LINEAR
        self._add_input(Input(self))
        self._add_output(Output())


    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value

    @property
    def interpolation(self):
        return self.__interpolation
    
    @interpolation.setter
    def interpolation(self, value):
        self.__interpolation = value

    def process(self):
        image = self._inputs[0].data.image
        h, w = image.shape[:2]
        image_scaled = cv2.resize(image, (int(w*self._scale), int(h*self._scale) ), interpolation = self.interpolation)
        
        self._outputs[0].set(IoData(image_scaled))
