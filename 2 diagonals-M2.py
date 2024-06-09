import cv2
import numpy as np


image = cv2.imread('cr7.jpg')
image=cv2.resize(image,(500,500))
height,width,channels=image.shape

#print(channels)

center_x = width // 2
center_y = height // 2


diagonal1_points = [(0, 0), (height, width)]  
diagonal2_points = [(0 , width), (height, 0)]  
center_point=[(center_x, center_y)]


triangle1_vertices = [diagonal1_points[0], center_point[0], diagonal2_points[0]]
triangle2_vertices = [diagonal1_points[0], diagonal2_points[1],  center_point[0]]
triangle3_vertices = [ center_point[0], diagonal1_points[1], diagonal2_points[1]]
triangle4_vertices = [diagonal1_points[1], diagonal2_points[0],  center_point[0]]


mask = np.zeros_like(image)
#print(mask)

cv2.fillConvexPoly(mask, np.array(triangle1_vertices), (0, 255, 255 ))
cv2.fillConvexPoly(mask, np.array(triangle2_vertices), (0, 0, 255))
cv2.fillConvexPoly(mask, np.array(triangle3_vertices), (0, 255, 0))
cv2.fillConvexPoly(mask, np.array(triangle4_vertices), (255, 0, 0))
#print(mask)

result = cv2.bitwise_and(image, mask)

cv2.imshow('cr7', result)
