import cv2
import numpy as np
import onnxruntime as ort
import time

def random_colour_masks(image, cls_id):
  colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190],[70, 80, 180],
             [0, 64, 64],[4, 64, 0],[64, 0, 64],[0, 64, 64], [64, 64, 0],[0, 64, 64],[64, 0, 64],[0, 64, 64],[192, 192, 64],
             [64, 0, 64],[0, 64, 64], [64, 64, 0],[0, 64, 64],[64, 0, 64],[0, 64, 64],[192, 192, 64],
             [64, 0, 64],[0, 64, 64], [64, 64, 0],[0, 64, 64],[64, 0, 64],[0, 64, 64],[192, 192, 64],
             [0, 64, 64],[4, 64, 0],[64, 0, 64],[0, 64, 64], [64, 64, 0],[0, 64, 64],[64, 0, 64],[0, 64, 64],[192, 192, 64],
             [0, 64, 64],[4, 64, 0],[64, 0, 64],[0, 64, 64], [64, 64, 0],[0, 64, 64],[64, 0, 64],[0, 64, 64],[192, 192, 64],
             [0, 64, 64],[4, 64, 0],[64, 0, 64],[0, 64, 64], [64, 64, 0],[0, 64, 64],[64, 0, 64],[0, 64, 64],[192, 192, 64],
             [0, 64, 64],[4, 64, 0],[64, 0, 64],[0, 64, 64], [64, 64, 0],[0, 64, 64],[64, 0, 64],[0, 64, 64],[192, 192, 64],]

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  b[image == 255], g[image == 255], r[image == 255] = colours[cls_id]
  coloured_mask = np.stack([b, g, r], axis=2)
  return coloured_mask

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def filter_Detections(results, thresh = 0.5):
    # if models is trained on 1 class
    if len(results[0]) == 25605:
        #considerable_detections = [detection for detection in results if detection[4] > thresh]
        #considerable_detections = np.array(considerable_detections)
        mask = results[:,4] > thresh
        considerable_detections = results[mask]

        return considerable_detections

    else:

        class_id = results[:,4:84].argmax(axis=1)
        confidence_score = results[:,4:84].max(axis=1)
        new_detection = np.hstack((results[:,:4], results[:,84:]))
        new_detection = np.column_stack((new_detection, class_id, confidence_score))


        mask = new_detection[:, -1] > thresh  # faster
        considerable_detections = new_detection[mask]





        #considerable_detections = [detection for detection in A if detection[-1] > thresh]  # slower
        #considerable_detections = np.array(considerable_detections)



        print("filter detection funtion might need to be changed for custom model")
        return considerable_detections


# NMS For Segmentation
def NMS(boxes, conf_scores, iou_thresh = 0.7):

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
    cx, cy, w, h, class_id, confidence = results[:,0], results[:,1], results[:,2], results[:,3], results[:,-2], results[:,-1]

    segmentation = results[:,4:-2]


    cx = cx/640.0 * img_w
    cy = cy/640.0 * img_h
    w = w/640.0 * img_w
    h = h/640.0 * img_h
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2

    segmentation = sigmoid(segmentation)

    boxes = np.column_stack((x1, y1, x2, y2,class_id, segmentation))
    keep, keep_confidences = NMS(boxes,confidence)

    keep = np.array(keep)


    return keep, keep_confidences



classes = None
with open('coco-classes.txt') as file:
    content = file.read()
    classes = content.split('\n')

del classes[-1]

image = cv2.imread('bicycle.jpg')


img_w,img_h = image.shape[1],image.shape[0]

img = cv2.resize(image,(640,640))
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = img.transpose(2,0,1)
img = img.reshape(1,3,640,640)

img = img / 255.0

print(img.shape)



onnx_model = ort.InferenceSession("yolov8n-seg.onnx")


t3 = time.time()

outputs = onnx_model.run(None, {"images":img.astype("float32")})
t4 = time.time()



#print("DNN OUTPUT LENGTH: ",len(results))
#print("ONNX OUTPUT LENGTH: ",len(outputs))

#print("dnn type: ",type(results))
#print("onnx type: ",type(outputs))

#print("DNN OUTPUT SHAPE: ",results.shape)
#print("ONNX OUTPUTS SHAPE: ",outputs[0].shape,outputs[1].shape)


print("ONNX INFERENCE TIME: ",t4-t3,"sec")


boxes = outputs[0][0][:84].transpose()
mask1 = outputs[0][0][84:].transpose()

mask2 = outputs[1][0]
mask2 = mask2.reshape(32,160*160)

mask = mask1 @ mask2



detections = np.hstack((boxes,mask))
#print("Detections shape: ", detections.shape)

t1 = time.time()

detections = filter_Detections(detections)


detections, confidences = rescale_back(detections,img_w,img_h)


#print(detections.shape)

boxes, mask = detections[:,:5], detections[:,5:]

mask = mask.reshape(mask.shape[0],160,160)
mask = (mask > 0.5).astype('uint8') * 255

t2 = time.time()

print("YOUR FUNCTIONS TIME:", t2 - t1, "sec")



for i, conf in enumerate(confidences):

    x1,y1,x2,y2, cls_id = boxes[i,:]

    cls_id = int(cls_id)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    conf = "{:.2f}".format(conf)

    seg = cv2.resize(mask[i,:,:], (img_w,img_h))
    bgr_mask = random_colour_masks(seg,cls_id)
    image = cv2.addWeighted(image, 1, bgr_mask, 0.5, 0)
    # draw the bounding boxes
    cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,255),1)
    cv2.putText(image,classes[cls_id]+' '+conf,(x1,y1-17),
                cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(255,0,255),1)
    #cv2.imshow("mask",bgr_mask)
    #cv2.waitKey(0)


cv2.imshow('img',image)
cv2.waitKey(0)


