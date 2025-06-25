import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the pre-trained model
final_model = tf.keras.models.load_model('my_model_weights.h5')

# Path to the image to test
image_path = './data/sm_test/airplane_601.jpg'

# Non-maximum Suppression function
def non_max_suppression(boxes, overlapThresh):
    """Perform non-maximum suppression on bounding boxes."""
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

# Load and process the image
image = cv2.imread(image_path)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
ssresults = ss.process()    

# After detecting regions and classifying them, collect bounding boxes and scores
imOut = image.copy()
boxes = []

for e, result in enumerate(ssresults):
    if e < 50:  # Limit to the top 50 region proposals
        x, y, w, h = result
        timage = image[y:y + h, x:x + w]
        resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
        resized = np.expand_dims(resized, axis=0)
        out = final_model.predict(resized)
        
        # Collect bounding boxes and corresponding confidence scores
        score = out[0][1]  # Assuming the second value is the confidence for the object class 
        if score > 0.5:  # Threshold to filter low confidence detections
            boxes.append([x, y, x + w, y + h, score])
            
# Convert list of boxes to numpy array for NMS
boxes = np.array(boxes)

# Apply Non-maximum Suppression (NMS)
nms_boxes = non_max_suppression(boxes, overlapThresh=0.3)

# Draw bounding boxes on the image
for box in nms_boxes:
    x1, y1, x2, y2 = box[:4]
    cv2.rectangle(imOut, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
        
# Display the output image with bounding boxes
plt.imshow(cv2.cvtColor(imOut, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Turn off axis
plt.show()
