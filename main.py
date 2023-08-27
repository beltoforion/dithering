import os

from ocvl.processor.dither_processor import *
from ocvl.processor.greyscale_processor import *
from ocvl.processor.scale_processor import *
from ocvl.processor.rgb_split_processor import *
from ocvl.processor.rgb_join_processor import *
from ocvl.helper.opencv_helper import *
from ocvl.source.file_source import *
from ocvl.source.video_sink import *
from ocvl.source.file_sink import *


def dither_single_image(source : Source, sink : Sink, dither_method):
    greyscale_processor = GreyscaleProcessor()
    greyscale_processor.connect_input(0, source.output[0])

    scale_processor = ScaleProcessor()
    scale_processor.scale = 0.5
    scale_processor.connect_input(0, greyscale_processor.output[0])

    dither_processor = DitherProcessor()
    dither_processor.method = dither_method
    dither_processor.connect_input(0, scale_processor.output[0])

    # connect the sink to the last processors output
    dither_processor.output[0].connect(sink.input[0])

    file = source.file_path
    file_name, file_ext = os.path.splitext(file)
    sink.output_path = f"{file_name}_{dither_method.name}_monochrome{file_ext}"

    return source


# Apply dithering to all color channels and combine the image back into an rgb image
def dither_rgb(source : Source, sink : Sink, dither_method, rgb_method):
    scale_processor = ScaleProcessor()
    scale_processor.scale = 1
    scale_processor.interpolation = cv2.INTER_NEAREST
    scale_processor.connect_input(0, source.output[0])

    rgb_split_processor = RgbSplitProcessor()
    rgb_split_processor.connect_input(0, scale_processor.output[0])

    dither_blue = DitherProcessor()
    dither_blue.method = dither_method
    dither_blue.connect_input(0, rgb_split_processor.output[0])

    dither_green = DitherProcessor()
    dither_green.method = dither_method
    dither_green.connect_input(0, rgb_split_processor.output[1])

    dither_red = DitherProcessor()
    dither_red.method = dither_method
    dither_red.connect_input(0, rgb_split_processor.output[2])

    rgb_join_processor = RbgJoinProcessor()
    rgb_join_processor.method = rgb_method
    rgb_join_processor.connect_input(0, dither_blue.output[0])
    rgb_join_processor.connect_input(1, dither_green.output[0])
    rgb_join_processor.connect_input(2, dither_red.output[0])

    # connect the sink to the last processors output
    rgb_join_processor.output[0].connect(sink.input[0])

    file = source.file_path
    file_name, file_ext = os.path.splitext(file)
    sink.output_path = f"{file_name}_{dither_method.name}_{rgb_method.name}{file_ext}"

    return source

dither_method = DitherMethod.DIFFUSION_XY
rgb_method = RgbJoinMethod.COLOR
output_format = OutputFormat.SAME_AS_INPUT
input_file = "sample6.jpg"
#input_file = "gradient.jpg"
#input_file = "sample1.mp4"
#input_file = "sample3.mp4"

#
# Source and Sink setup
#

source = FileSource(input_file)
source.max_num_frames = 200

sink = VideoSink() if os.path.splitext(input_file)[1].lower()=='.mp4' else FileSink()
sink.output_format = output_format

#
# Job execution
#

job = dither_rgb(source, sink, dither_method, rgb_method)
#job = dither_single_image(source, sink, dither_method)
job.start()
