import cv2
from skimage import exposure
from vehicules_detection import reduceflow


#make objects in the image more clear
image=cv2.imread("roads2.jpeg",0)
imeq=cv2.equalizeHist(image)
cv2.imshow("image",image)
cv2.imshow("image2",imeq)

#another method
imeq2=exposure.equalize_hist(image)
cv2.imshow("image3",imeq2)


image=reduceflow(image)
# object edges detection
edges=cv2.Canny(image ,30 ,100)
ct , _= cv2.findContours(edges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image ,ct ,-1 ,(0,255,0) ,2)
cv2.imshow("image1",image)






