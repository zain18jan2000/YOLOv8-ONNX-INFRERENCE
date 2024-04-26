# Following code is for running YOLOv8 pose estimation ONNX model using opencv DNN module

'''
Things to note --> YOLOv8 on inference provides 8400 detections.
In my case, there's only 1 class with 4 key points, so each detection has 17 parameters.

For more info visit ---> https://github.com/ultralytics/ultralytics/issues/4731
'''

import cv2
import numpy as np

def filter_Detections(results, thresh = 0.5):
    considerable_detections = [detection for detection in results if detection[4] > thresh]
    considerable_detections = np.array(considerable_detections)
    return considerable_detections

def NMS(boxes, conf_scores, iou_thresh = 0.55):

    '''
    boxes [[x1,y1, x2,y2, kptX1, kptY1, kptX2, kptY2, kptX3, kptY3, kptX4, kptY4],
            [x1,y1, x2,y2, kptX1, kptY1, kptX2, kptY2, kptX3, kptY3, kptX4, kptY4], ...]
    '''

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


# rescale all the values
def rescale_back(results,img_w,img_h):
    cx, cy, w, h, confidence = results[:,0], results[:,1], results[:,2], results[:,3], results[:,4]
    cx = cx/640.0 * img_w
    cy = cy/640.0 * img_h
    w = w/640.0 * img_w
    h = h/640.0 * img_h

    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2

    kptX1, kptY1 = results[:,5], results[:,6]
    kptX2, kptY2 = results[:,8], results[:,9]
    kptX3, kptY3 = results[:,11], results[:,12]
    kptX4, kptY4 = results[:,14], results[:,15]

    kptX1, kptY1 = kptX1/640.0 * img_w, kptY1/640.0 * img_h
    kptX2, kptY2 = kptX2/640.0 * img_w, kptY2/640.0 * img_h
    kptX3, kptY3 = kptX3/640.0 * img_w, kptY3/640.0 * img_h
    kptX4, kptY4 = kptX4/640.0 * img_w, kptY4/640.0 * img_h

    boxes = np.column_stack((x1, y1, x2, y2, kptX1, kptY1, kptX2, kptY2, kptX3, kptY3, kptX4, kptY4))

    keep, keep_confidences = NMS(boxes,confidence)

    print(np.array(keep).shape)

    return keep



# read the image
image = cv2.imread('k1.jpeg')

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
net = cv2.dnn.readNetFromONNX('best.onnx')  # readNet() also works

net.setInput(img)

# run the inference
out = net.forward()

results = out[0]

# transpose the result just for ease
results = results.transpose()

# filter the detections to remove low confidence detections
results = filter_Detections(results)

# rescale the detections' parameters back to original image dimensions
rescaled_results = rescale_back(results,img_w,img_h)


for res in rescaled_results:
    x1,y1,x2,y2, kpX1, kpY1, kpX2, kpY2, kpX3, kpY3, kpX4, kpY4 = res

    # draw the bounding boxes
    cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,255),1)

    # draw all the key points
    cv2.circle(image,(int(kpX1),int(kpY1)),3,(255,0,0),-1)
    cv2.circle(image,(int(kpX2),int(kpY2)),3,(255,0,0),-1)
    cv2.circle(image, (int(kpX3), int(kpY3)), 3, (255, 0, 0), -1)
    cv2.circle(image, (int(kpX4), int(kpY4)), 3, (255, 0, 0), -1)


cv2.imshow('img',image)
cv2.waitKey(0)
#cv2.imshow("result.jpg",image)

