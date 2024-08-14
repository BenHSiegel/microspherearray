"""Grab in high frame rate mode for 10 seconds"""

from egrabber import *
import time
import datetime
import ctypes as ct
import cv2
import numpy as np
import os
import sys




class Inputs:
    def __init__(self, args):
    
        #Base options
        self.input_path = args.input_path
        self.man_dim_flag = bool(args.man_dim_flag)
        self.width = float(args.width)
        self.height = float(args.height)
        self.run_time = float(args.run_time)

        #Camera settings
        self.framerate = float(args.framerate)
        self.exposure = int(args.exposure)

        #Saving Options
        if args.output_csv_path:
            self.output_csv_path = args.output_csv_path
        else:
            self.output_csv_path = os.getcwd() + '/_' + str(datetime.datetime.now())[:19]
        self.save_flag = bool(args.save_flag)
        self.no_runs = int(args.no_runs)
        self.out_video_flag = bool(args.out_video_flag)
        if args.out_video:
            self.out_video = args.out_video
        else:
            self.out_video = os.getcwd() + '/_' + str(datetime.datetime.now())[:19]
        


def parse_args():
    import argparse
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description='Ben attempt object tracking',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Base options
    base_options = parser.add_argument_group('Base options')
    base_options.add_argument('-i', '--input-raw-file', dest='input_path', default="",
                            help="Path to input RAW file. If not specified, the live stream of the first available camera is used. "
                            "If it's a camera serial number, it will try to open that camera instead.")
    base_options.add_argument('-mandim', '--manually-set-dimensions', dest='man_dim_flag', type=bool, default=False,
                              help='Set to true to use manually input values for the dimensions of the camera.')
    base_options.add_argument('-w', '--width', dest='width', type=int, default=1920,
                              help="Camera's width in pixels (for the EoSense2.0MCX12 w=1920)(default = 1920)")
    base_options.add_argument('-h', '--height', dest='height', type=int, default=1080,
                              help="Camera's height in pixels (for the EoSense2.0MCX12 h=1080)(default = 1080)")
    base_options.add_argument('-rt', '--run-time', dest='run_time', type=float, default=10,
                              help='Amount of time to stream from the camera in seconds (default = 10).')
    
    #Camera settings options
    camera_settings = parser.add_argument_group('Camera Settings')
    camera_settings.add_argument('-fps', '--framerate', dest='fps', type=float, default='1000'
                                 help='Float to set the framerate of the camera (max = 2247 when in CXP12_X4 mode)')
    camera_settings.add_argument('-exp', '--exposure-time', dest='exposure', type=int, default='500',
                                 help="Exposure time of the camera in (default = 500)")
    # Saving options
    saving_options = parser.add_argument_group('Saving options')
    saving_options.add_argument('-csv', '--save-csv-path', dest='output_csv_path', type=str, default='',
                                help='File path of output CSV files that contain the information of detected objects, excluding the file extension. Default: \'EVK_\{timestamp\}.csv\' at location of script.')
    saving_options.add_argument('-csvf', '--save-flag', dest='save_flag', type=bool, default=False,
                                help="Flag that determines if measurements are recorded. Default: False.")
    saving_options.add_argument('-csvn', '--csv-runs', dest='no_runs', type=int, default=5,
                                help="Determines the number of runs that are required for saving. Default: 5 runs.")
    saving_options.add_argument('-ovf', '--out-video-flag', dest='out_video_flag', type=bool, default=False,
                                 help="Boolean to choose whether to save an avi file. Default = False")
    saving_options.add_argument('-ov', '--out-video', dest='out_video', type=str, default='',
                                 help="Path to an output AVI file to save the resulting video.")

    args = parser.parse_args()
    return args


def processImage(ptr, w, h, size):
    # processing code
    data = ct.cast(rgb.get_address(), ct.POINTER(ct.c_ubyte * rgb.get_buffer_size())).contents
    c = 3
    npdataarray = np.frombuffer(data, count=rgb.get_buffer_size(), dtype=np.uint8).reshape((h,w,c))
    pass


def main():
    """ Main """
    args = parse_args()
    inputs = Inputs(args)
    gentl = EGenTL()
    grabber = EGrabber(gentl)

    grabber.stream.set('BufferPartCount', 1)
    if inputs.man_dim_flag == True:
        w = inputs.width
        h = inputs.height
    else:
        w = grabber.stream.get('Width')
        h = grabber.stream.get('Height')

    grabber.stream.set('BufferPartCount', 100)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    fps = inputs.framerate

    outdir = '{}/{}.output'.format(os.getenv('OUTPUT_DIR', os.path.dirname(__file__)), os.path.splitext(os.path.basename(__file__))[0])
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    out = cv2.VideoWriter(os.path.join(outdir, 'output1.avi'), fourcc, fps, (w,  h))

    grabber.realloc_buffers(20)
    grabber.start()

    t_start = time.time()
    t_stop = t_start + 10
    t_show_stats = t_start + 1
    t = t_start
    while t < t_stop:
        with Buffer(grabber) as buffer:
            bufferPtr = buffer.get_info(BUFFER_INFO_BASE, INFO_DATATYPE_PTR)
            imageSize = buffer.get_info(BUFFER_INFO_CUSTOM_PART_SIZE, INFO_DATATYPE_SIZET)
            delivered = buffer.get_info(BUFFER_INFO_CUSTOM_NUM_DELIVERED_PARTS, INFO_DATATYPE_SIZET)
            processed = 0
            while processed < delivered:
                imagePtr = bufferPtr + processed * imageSize
                processImage(imagePtr, w, h, imageSize)
                processed = processed + 1
        if t >= t_show_stats:
            dr = grabber.stream.get('StatisticsDataRate')
            fr = grabber.stream.get('StatisticsFrameRate')
            print('{}x{} : {:.0f} MB/s, {:.0f} fps'.format(w, h, dr, fr))
            t_show_stats = t_show_stats + 1
        t = time.time()


if __name__ == "__main__":
    main()