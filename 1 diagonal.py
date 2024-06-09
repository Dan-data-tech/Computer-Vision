import cv2


image=cv2.imread("cr7.jpg")
image=cv2.resize(image,(500,500))


height,width,channels=image.shape

for i in range(height):
    for j in range(width):
        if i<j and j<width-1-i:
            image[i][j][0]=0
            image[i][j][1]=0
        elif i<j and j>width-1-i:
            image[i][j][0]=0
            image[i][j][2]=0
        elif i>j and j>width-1-i:
            image[i][j][1]=0
            image[i][j][2]=0
        else:
            image[i][j][0]=0

cv2.imshow("cr7", image)
            
