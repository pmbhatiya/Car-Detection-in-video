import cv2
import os
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

vidcap=cv2.VideoCapture('test_video.mp4')
success,image=vidcap.read()
count_frame=0
my_car_x=set()
my_car_y=set()
my_car_h=set()
my_car_w=set()

while success:
    cv2.imwrite('frame/frame%d.jpg'%count_frame,image)
    

# Loading image
    img = cv2.imread('frame/frame%d.jpg'%count_frame)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape


# Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

# Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    my_car=0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label=='car':
                my_car_x.add(x)
                my_car_y.add(y)
                my_car_w.add(w)
                my_car_h.add(h)
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
    my_car=min(len(my_car_x),min(len(my_car_y),min(len(my_car_w),len(my_car_h))))
    print("Up To this frame No ",count_frame,"Number of unique Cars are :",my_car)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    os.remove('frame/frame%d.jpg'%count_frame)
    success,image=vidcap.read()
    count_frame+=1
   



