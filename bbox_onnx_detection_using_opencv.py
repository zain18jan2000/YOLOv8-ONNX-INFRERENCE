# Following is the code to run YOLOv8 ONNX inference for object detection

'''
Things to note --> YOLOv8 on inference provides 8400 detections.
In case of 1 class there will be 5 parameters for each detection.
In case of 2 class there will be 6 parameters for each detection and so on.
The first 4 will be BBox parameters presented as cx,cy,w,h while remaining lasts will be class scores
such that 5th parameter will be confidence value of class 0, 6th will be confidence value of class 1
and so on.
we only consider the class score having max value for every detection.
'''


import cv2
import numpy as np

def filter_Detections(results, thresh = 0.5):
    # if models is trained on 1 class
    if len(results[0]) == 5:
        considerable_detections = [detection for detection in results if detection[4] > thresh]
        considerable_detections = np.array(considerable_detections)
        return considerable_detections

    else:
        A = []
        for detection in results:

            class_id = detection[4:].argmax()
            confidence_score = detection[4:].max()

            new_detection = np.append(detection[:4],[class_id,confidence_score])

            A.append(new_detection)

        A = np.array(A)

        considerable_detections = [detection for detection in A if detection[-1] > thresh]
        considerable_detections = np.array(considerable_detections)
        print("This code might need to be updated if you have multiple classes.")
        return considerable_detections


def NMS(boxes, conf_scores, iou_thresh = 0.55):

    #  boxes [[x1,y1, x2,y2], [x1,y1, x2,y2], ...]

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2-x1)*(y2-y1)

    order = conf_scores.argsort()

    keep = []
    keep_confidences = []

    while len(order) > 0:
        idx = order[-1]
        A = boxes[idx]
        conf = conf_scores[idx]

        order = order[:-1]

        xx1 = np.take(x1, indices= order)
        yy1 = np.take(y1, indices= order)
        xx2 = np.take(x2, indices= order)
        yy2 = np.take(y2, indices= order)

        keep.append(A)
        keep_confidences.append(conf)

        # iou = inter/union

        xx1 = np.maximum(x1[idx], xx1)
        yy1 = np.maximum(y1[idx], yy1)
        xx2 = np.minimum(x2[idx], xx2)
        yy2 = np.minimum(y2[idx], yy2)

        w = np.maximum(xx2-xx1, 0)
        h = np.maximum(yy2-yy1, 0)

        intersection = w*h

        # union = areaA + other_areas - intesection
        other_areas = np.take(areas, indices= order)
        union = areas[idx] + other_areas - intersection

        iou = intersection/union

        boleans = iou < iou_thresh

        order = order[boleans]

        # order = [2,0,1]  boleans = [True, False, True]
        # order = [2,1]

    return keep, keep_confidences



def rescale_back(results,img_w,img_h):
    cx, cy, w, h, class_id, confidence = results[:,0], results[:,1], results[:,2], results[:,3], results[:,4], results[:,-1]
    cx = cx/640.0 * img_w
    cy = cy/640.0 * img_h
    w = w/640.0 * img_w
    h = h/640.0 * img_h
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2

    boxes = np.column_stack((x1, y1, x2, y2, class_id))
    keep, keep_confidences = NMS(boxes,confidence)
    print(np.array(keep).shape)
    return keep, keep_confidences



classes = None
with open('coco-classes.txt') as file:
    content = file.read()
    classes = content.split('\n')

del classes[-1]

# read the image
image = cv2.imread('bicycle.jpg')

#w, h = int(image.shape[1]*0.5), int(image.shape[0]*0.5)
#image = cv2.resize(image, (w,h))

# YOLOv8 need RGB image
img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
img_h,img_w = img.shape[:2]

# resize image to get the desired size (640,640) for inference
img = cv2.resize(img,(640,640))

# change the order of image dimension from (640,640,3) to (3,640,640)
img = img.transpose(2,0,1)

# add an extra dimension at index 0
img = img.reshape(1,3,640,640)

# scale to 0-1
img = img/255.0

# read the trained onnx model
net = cv2.dnn.readNetFromONNX('yolov8n.onnx')  # readNet() also works

net.setInput(img)

# run the inference
out = net.forward()

results = out[0]

# transpose the result just for ease
results = results.transpose()


# filter the detections to remove low confidence detections
results = filter_Detections(results)

# rescale the detections' parameters back to original image dimensions
rescaled_results, confidences = rescale_back(results,img_w,img_h)


for res, conf in zip(rescaled_results, confidences):

    x1,y1,x2,y2, cls_id = res
    cls_id = int(cls_id)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    conf = "{:.2f}".format(conf)
    # draw the bounding boxes
    cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,255),1)
    cv2.putText(image,classes[cls_id]+' '+conf,(x1,y1-17),
                cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(255,0,255),1)


cv2.imshow('img',image)
cv2.waitKey(0)
