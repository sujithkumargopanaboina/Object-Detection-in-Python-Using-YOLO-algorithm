#pip install pillow
#pip install opencv-python
from PIL import Image
import numpy as np
import cv2
#give the image name in the below ''
#image resize automation
image = Image.open('1.jpg')
print(image.size)
resized_image = image.resize((600,600))
print(resized_image.size)
resized_image.save('na.png')
image = Image.open('na.png')
print(image.size)
#yolov3 algoritham
#importing weights
classes_names=['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
model=cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
layer_names = model.getLayerNames()
output_layers=[layer_names[i[0]-1]for i in model.getUnconnectedOutLayers()]
#importing image
image=cv2.imread("na.png")
height, width, channels = image.shape
#cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True),size can be 224,224 for low quality 416,416 for medium quality.
blob=cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False)
model.setInput(blob)
outputs=model.forward(output_layers)
class_ids = []
confidences = []
boxes = []

for output in outputs:
    for identi in output:
        scores = identi[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence >= 0.1:
            # Object detected
            centerx = int(identi[0] * width)
            centery = int(identi[1] * height)
            w = int(identi[2] * width)
            h = int(identi[3] * height)
            # Rectangle coordinates
            x = int(centerx - w / 2)
            y = int(centery - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_HERSHEY_COMPLEX
colors = np.random.uniform(0, 255, size=(50, 3))
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        confidence = str("{:.2f}".format(confidences[i]))
        label = str(classes_names[class_ids[i]]+confidence)
        color = colors[i]

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
        cv2.putText(image, label,(x, y + 10), font, 0.8, color, 1)

cv2.imshow("Image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()