import cv2
import numpy as np

def reduceflow (image):
    newim=cv2.GaussianBlur(image,(45,45),0)
    sharpim=cv2.addWeighted(image,1.5,newim,-1, 0)
    return sharpim

def reduceflow2 (image):
    newim=cv2.medianBlur(image,3)
    return newim
    
def reduceflow3(image):
    newim=cv2.bilateralFilter(image,15,150,150)
    return newim

def reduceflow4(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Non-local Means Denoising
    denoised_image = cv2.fastNlMeansDenoising(img_gray, None, h=10, searchWindowSize=21)
    return denoised_image
    
# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Define output layer names
output_layer_names = net.getUnconnectedOutLayersNames()

# Load COCO dataset class names
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

image= cv2.imread("road.jpg")
#image=reduceflow(image)
#image=reduceflow2(image)
#image=reduceflow3(image) (not good result)
image2=reduceflow4(image)
cv2.imshow("image2",image2)

#get the dimensions of the image
height, width, channels= image.shape
#image=cv2.resize(image,(1000,1000))

# Preprocess image for YOLO input
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# Set input to the network
net.setInput(blob)
outs = net.forward(output_layer_names)


class_ids = []
confidences = []
boxes = []

# Process output

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0:
            #print(scores)
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

# Non-maximum suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw bounding boxes and labels
colors = np.random.uniform(0, 255, size=(len(classes), 3))
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = colors[class_id]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display result
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
