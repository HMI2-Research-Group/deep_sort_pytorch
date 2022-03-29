import os
import cv2
from cv_bridge import CvBridge
import time
import argparse
import torch
import warnings
import numpy as np
import sys
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
import ros_numpy
import image_geometry

sys.path.append(os.path.join(os.path.dirname(__file__), "thirdparty/fast-reid"))


from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results


def disp_image(image):
    cv2.imshow("debug", image)
    cv2.waitKey(0)


class ROS_VideoTracker(object):
    def __init__(self, cfg, args, rgb_stream, depth_stream, point_stream):
        use_cuda = args.use_cuda and torch.cuda.is_available()
        self.cfg = cfg
        self.rgb_stream = None
        self.depth_stream = None
        self.args = args
        self.logger = get_logger("root")
        self.bridge = CvBridge()
        self.logger.info("Initializing ROS Video Tracker")
        rospy.init_node("ros_video_tracker", anonymous=True, disable_rostime=True)
        rospy.Subscriber(rgb_stream, Image, self.rgb_callback)
        rospy.Subscriber(depth_stream, Image, self.depth_callback)
        rospy.Subscriber(point_stream, PointCloud2, self.poincloud_callback)
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names
        # rospy.spin()

    def rgb_callback(self, msg):
        self.rgb_stream = self.bridge.imgmsg_to_cv2(msg)

    def depth_callback(self, msg):
        self.depth_stream = self.bridge.imgmsg_to_cv2(msg)

    def poincloud_callback(self, msg):
        self.point_stream = msg

    def get_depth_from_pixels(self, ori_depth, bbox_outputs):
        def recursive_non_nan_search(depth, x, y):
            # TODO: Recursively search for non-nan values around the point (x, y), not just 2 layer for loop
            for i in range(x - 3, x + 3):
                for j in range(y - 3, y + 3):
                    try:
                        if not np.isnan(depth[j, i]):
                            return depth[j, i]
                    except:
                        pass
            return np.nan

        # TODO: Initilize this 100 as a parameter
        human_positions = np.zeros((100, 3))
        for i in range(len(bbox_outputs)):
            xmin = bbox_outputs[i][0]
            ymin = bbox_outputs[i][1]
            xmax = bbox_outputs[i][2]
            ymax = bbox_outputs[i][3]
            human_index = bbox_outputs[i][4]
            mid_x, mid_y = (xmin + xmax) / 2, (ymin + ymax) / 2
            depth_val = recursive_non_nan_search(ori_depth, int(mid_x), int(mid_y))
            if not np.isnan(depth_val):
                human_positions[int(human_index)] = (
                    int(mid_x),
                    int(mid_y),
                    depth_val,
                )
        return human_positions

    def run(self):
        while not rospy.is_shutdown():
            results = []
            idx_frame = 0
            if self.rgb_stream is not None and self.depth_stream is not None:
                idx_frame += 1
                if idx_frame % self.args.frame_interval:
                    continue

                start = time.time()
                ori_im = self.rgb_stream
                ori_depth = self.depth_stream
                point_stream = self.point_stream
                data = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(point_stream)
                im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

                # do detection
                bbox_xywh, cls_conf, cls_ids = self.detector(im)

                # select person class
                mask = cls_ids == 0

                bbox_xywh = bbox_xywh[mask]
                # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
                bbox_xywh[:, 3:] *= 1.2
                cls_conf = cls_conf[mask]

                # do tracking
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_tlwh = []
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                    for bb_xyxy in bbox_xyxy:
                        bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                    results.append((idx_frame - 1, bbox_tlwh, identities))
                    all_pedestrian_depth = self.get_depth_from_pixels(ori_depth, outputs)

                end = time.time()

                if self.args.display:
                    cv2.imshow("test", ori_im)
                    cv2.waitKey(1)

                if self.args.save_path:
                    self.writer.write(ori_im)

                # save results
                if self.args.save_path:
                    write_results(self.save_results_path, results, "mot")

                # logging
                self.logger.info(
                    "time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}".format(
                        end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)
                    )
                )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_mmdetection", type=str, default="./configs/mmdet.yaml")
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--config_fastreid", type=str, default="./configs/fastreid.yaml")
    parser.add_argument("--fastreid", action="store_true")
    parser.add_argument("--mmdet", action="store_true")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str)
    # parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    if args.mmdet:
        cfg.merge_from_file(args.config_mmdetection)
        cfg.USE_MMDET = True
    else:
        cfg.merge_from_file(args.config_detection)
        cfg.USE_MMDET = False
    cfg.merge_from_file(args.config_deepsort)
    if args.fastreid:
        cfg.merge_from_file(args.config_fastreid)
        cfg.USE_FASTREID = True
    else:
        cfg.USE_FASTREID = False

    # with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
    #     vdo_trk.run()
    ros_vid_tracker = ROS_VideoTracker(
        cfg, args, "/realsense/color/image_raw", "/realsense/depth/image_rect_raw", "/realsense/depth/color/points"
    )
    ros_vid_tracker.run()
