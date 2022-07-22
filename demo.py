from sort import *
from detect import darknet
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="../../0001.mp4",
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="./detect/yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./detect/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./detect/voc.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.6,
                        help="remove detections with confidence below this value")
    return parser.parse_args()

def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))

def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path

def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height

def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted

def det2trt(frame, detections):
    det_trt = []
    for label, score, bbox in detections:
        bbox_adjusted = convert2original(frame, bbox)
        left, top, right, bottom = bbox2points(bbox_adjusted)
        det = [left, top, right, bottom]
        det.append(float(score)/100)
        det = np.array(det)
        det_trt.append(det)
    return np.array(det_trt)

def draw_boxes(detections, image, colors):
    import cv2
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        cv2.rectangle(image, (left, top), (right, bottom), (0,0,0), 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,0,0), 2)
    return image


if __name__ =="__main__":
    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)

    #create instance of SORT
    mot_tracker = Sort(max_age=30, min_hits=5, iou_threshold=0.5)
    colours = np.random.rand(32, 3)*255

    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    frame_num = 0
    max_num = 0
    # get detections
    while cap.isOpened():
        ret, frame = cap.read()
        frame_num = frame_num+1
        if frame_num > 882:
            print(frame_num)
        if not ret:
            break
        frame_rgb = frame
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                    interpolation=cv2.INTER_LINEAR)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, img_for_detect, thresh=args.thresh)
        detections_adjusted = []
        for label, confidence, bbox in detections:
            bbox_adjusted = convert2original(frame, bbox)
            detections_adjusted.append((str(label), confidence, bbox_adjusted))
        image = draw_boxes(detections_adjusted, frame_rgb, class_colors)
        fps = int(1/(time.time() - prev_time))
        # print("FPS: {}".format(fps))
        # darknet.print_detections(detections, args.ext_output)
        darknet.free_image(img_for_detect)
        if len(detections) >= 3:
            print(len(detections))
        detections = det2trt(frame_rgb, detections)
        # update SORT
        track_bbs_ids = mot_tracker.update(detections)

        # track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
        for d in track_bbs_ids:
            # print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame_num,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]))
            d = d.astype(np.int32)
            cv2.rectangle(image, (d[0],d[1]),(d[2],d[3]),colours[d[4]%32,:])
            cv2.putText(image, str(d[4]), (d[0],d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colours[d[4]%32,:], 2)
            cv2.putText(image, "frame_num:"+str(frame_num), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
            cv2.putText(image, "FPS:"+str(fps), (10,43), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
            if d[4]>max_num:
                max_num = d[4]
        cv2.imshow("frame",image)
        if cv2.waitKey(10) & 0xFF == 27:
            break
    print("max_num=", max_num)
    cap.release()