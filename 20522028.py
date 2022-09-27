import numpy as np
from tqdm import tqdm
import cv2
import sys
import queue
import matplotlib.pyplot as plt

def get_connected_components(img, threshold): # <200 => hong cau
    flag = np.zeros_like(img)
    components = []
    h, w = img.shape
    for row in tqdm(range(h), desc='Finding connected components'):
        for col in range(w):
            if flag[row, col] == 1: 
                continue
            if img[row, col] >= threshold: 
                count = 0
                q = queue.Queue()
                q.put([row, col])
                flag[row, col] = 1
                while not q.empty():
                    p = q.get()
                    x = p[1]
                    y = p[0]
                    count += 1
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            cur_x = x + i
                            cur_y = y + j
                            if cur_x>=0 and cur_x<w and cur_y>=0 and cur_y<h and flag[cur_y, cur_x]==0 and img[cur_y, cur_x]>=threshold:
                                flag[cur_y, cur_x] = 1
                                q.put([cur_y, cur_x])
                components.append(count)
    return components


def otsu(img):
    hist = img.flatten() 
    thresholds = list(set(hist))
    intra_class_var = sys.maxsize
    threshold = 0
    for i in range(1, len(thresholds)): 
        class1 = hist[np.where(hist<thresholds[i])]
        class2 = hist[np.where(hist>=thresholds[i])]
        var1 = np.var(class1)
        var2 = np.var(class2)
        if intra_class_var > len(class1)*var1+len(class2)*var2:
            intra_class_var = len(class1)*var1+len(class2)*var2
            threshold = thresholds[i]
    binary_img = img.copy()
    binary_img[np.where(img>=threshold)] = 0 
    binary_img[np.where(img<threshold)] = 255
    return binary_img

def dilation(img, kernel, origin, iters=1):
    h, w = img.shape 
    result = img.copy()
    for iter in range(iters):
        flag_img = np.zeros_like(img)
        for i in tqdm(range(h), desc=f'Dilatting {iter}-th...'):
            for j in range(w):
                if (i-1<0 and j-1<0):
                    flag_img[i, j] = np.sum(result[i:i+2, j:j+2] & kernel[1:3, 1:3])

                else:
                    if (i-1<0 and i+1<h):
                        flag_img[i, j] = 0#np.sum(result[i:i+2, x:j+2] & kernel[1:3, 0:3])

                    else :
                        if (j-1<0 and j+1<w):
                            x = i-1
                            flag_img[i, j] = 0#np.sum(result[x:i+2, j:j+2] & kernel[0:3, 1:3])
 
                        else:
                            if (j-1>=0 and j+1>=w and i-1>=0 and i+1>=h):
                                flag_img[i, j] = np.sum(result[i-1:i+1, j-1:j+1] & kernel[0:2, 0:2])
                            else:
                                if (j-1>=0 and j+1>=w):
                                    flag_img[i, j] = np.sum(result[i-1:i+2, j-1:j+1] & kernel[0:3, 0:2])
                                else:
                                    if (i-1>=0 and i+1>=h):
                                        flag_img[i, j] = np.sum(result[i-1:i+1, j-1:j+2] & kernel[0:2, 0:3])
                                    else:
                                        flag_img[i, j] = np.sum(result[i-1:i+2, j-1:j+2] & kernel[0:3, 0:3])
        result[np.where(flag_img>0)] = 255
    return result

def erosion(img, kernel, origin, iters=1):
    h, w = img.shape
    result = img.copy() 
    checksum = np.sum(kernel)
    for iter in range(iters):
        flag_img = np.zeros_like(img)
        cur_result = np.zeros_like(img)
        new_kernel = np.zeros_like(kernel)
        for i in tqdm(range(h), desc=f'Erossing {iter}-th...'):
            for j in range(w):
                if (i-1<0 and j-1<0):
                    flag_img[i, j] = np.sum(result[i:i+2, j:j+2] & kernel[1:3, 1:3])

                else:
                    if (i-1<0 and i+1<h):
                        flag_img[i, j] = 0#np.sum(result[i:i+2, x:j+2] & kernel[1:3, 0:3])

                    else :
                        if (j-1<0 and j+1<w):
                            x = i-1
                            flag_img[i, j] = 0#np.sum(result[x:i+2, j:j+2] & kernel[0:3, 1:3])
 
                        else:
                            if (j-1>=0 and j+1>=w and i-1>=0 and i+1>=h):
                                flag_img[i, j] = np.sum(result[i-1:i+1, j-1:j+1] & kernel[0:2, 0:2])
                            else:
                                if (j-1>=0 and j+1>=w):
                                    flag_img[i, j] = np.sum(result[i-1:i+2, j-1:j+1] & kernel[0:3, 0:2])
                                else:
                                    if (i-1>=0 and i+1>=h):
                                        flag_img[i, j] = np.sum(result[i-1:i+1, j-1:j+2] & kernel[0:2, 0:3])
                                    else:
                                        flag_img[i, j] = np.sum(result[i-1:i+2, j-1:j+2] & kernel[0:3, 0:3])

                # x1 = i - origin[0]; y1 = j - origin[1]
                # x2 = i + kernel.shape[0] - origin[0]; y2 = j + kernel.shape[1] - origin[1]
                # k_x1 = 0; k_y1 = 0
                # k_x2 = kernel.shape[0]; k_y2 = kernel.shape[1] 
                
                # if x1 < 0: 
                #     k_x1 -= x1
                #     x1 = 0
                # if y1 < 0: 
                #     k_y1 -= y1 
                #     y1 = 0
                # if x2 > h: 
                #     k_x2 -= (x2 - h)
                #     x2 = h
                # if y2 > w: 
                #     k_y2 -= (y2 - w)
                #     y2 = w  
                # flag_img[i, j] = np.sum(result[x1:x2, y1:y2] & kernel[k_x1:k_x2, k_y1:k_y2])
        cur_result[np.where(flag_img==checksum)] = 255
        result = cur_result.copy()
    return result

def closing(img, kernel, origin, iters=1):
    result = img.copy()
    for iter in range(iters):
        print(f'Closing process {iter}-th...')
        result = dilation(result, kernel, origin)
        result = erosion(result, kernel, origin)
    return result

def opening(img, kernel, origin, iters=1):
    result = img.copy()
    for iter in range(iters):
        print(f'Opening process {iter}-th...')
        result = erosion(result, kernel, origin)
        result = dilation(result, kernel, origin)
    return result

if __name__ == "__main__":
    # Read source image
    img = cv2.imread('img.png', 0)

    # Apply Otsu's method
    after_otsu = otsu(img) 

    kernel = np.array(
        [[1,1,1],
        [1,1,1],
        [1,1,1]]
    ).astype(np.uint8)
    origin = (1, 1)

    # Image Processing
    result = opening(after_otsu, kernel, origin, 3)
    cv2.imwrite('kq3.png', result)
    bloods = get_connected_components(result, 255)
    blood_num = len(bloods)
    blood_area = round(np.mean(np.array(bloods)), 2)

    plt.subplot(5, 3, 2), plt.imshow(img, cmap='gray'), plt.title("Source image")
    plt.subplot(5, 3, 7), plt.imshow(after_otsu, cmap='gray'), plt.title("After Otsu's method")
    plt.subplot(5, 3, 8), plt.imshow(kernel*255, cmap='gray'), plt.title(f"Structure Element \n Origin={origin}")
    plt.subplot(5, 3, 9), plt.imshow(result, cmap='gray'), plt.title(f"After 5-th Opening")
    plt.subplot(5, 3, 14), plt.imshow(result, cmap='gray'), plt.title(f"Number of cell={blood_num} \n Average cell area={blood_area}")
    plt.savefig('result.png')