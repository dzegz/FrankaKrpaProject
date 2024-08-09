from enum import IntEnum
import pyk4a

# k4a_fps_t
class FPS(IntEnum):
    FPS_5 = 0
    FPS_15 = 1
    FPS_30 = 2


# k4a_image_format_t
class ImageFormat(IntEnum):
    COLOR_MJPG = 0
    COLOR_NV12 = 1
    COLOR_YUY2 = 2
    COLOR_BGRA32 = 3
    DEPTH16 = 4
    IR16 = 5
    CUSTOM8 = 6
    CUSTOM16 = 7
    CUSTOM = 8


# k4a_depth_mode_t
class DepthMode(IntEnum):
    OFF = 0
    NFOV_2X2BINNED = 1
    NFOV_UNBINNED = 2
    WFOV_2X2BINNED = 3
    WFOV_UNBINNED = 4
    PASSIVE_IR = 5


# k4a_color_resolution_t
class ColorResolution(IntEnum):
    OFF = 0
    RES_720P = 1
    RES_1080P = 2
    RES_1440P = 3
    RES_1536P = 4
    RES_2160P = 5
    RES_3072P = 6


# k4a_wired_sync_mode_t
class WiredSyncMode(IntEnum):
    STANDALONE = 0
    MASTER = 1
    SUBORDINATE = 2


config = pyk4a.Config(
    color_resolution = ColorResolution.RES_1080P,
    color_format = ImageFormat.COLOR_BGRA32,
    depth_mode = DepthMode.WFOV_UNBINNED,
    camera_fps = FPS.FPS_15,
    synchronized_images_only = True,
    depth_delay_off_color_usec = 0,
    wired_sync_mode = WiredSyncMode.STANDALONE,
    subordinate_delay_off_master_usec = 0,
    disable_streaming_indicator = False,
)