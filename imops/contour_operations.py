import cv2
import numpy as np
from scipy import ndimage
import sympy
from math import atan2, cos, sin, sqrt, pi
from skimage.morphology import disk, skeletonize    
from scipy.spatial import distance
import matplotlib.pyplot as plt

import mlutils.imops.image_operations as iops

def drawAxis(img, p_, q_, color, scale):
      p = list(p_)
      q = list(q_)
     
      ## [visualization1]
      angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
      hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
     
      # Here we lengthen the arrow by a factor of scale
      q[0] = p[0] - scale * hypotenuse * cos(angle)
      q[1] = p[1] - scale * hypotenuse * sin(angle)
      cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
     
      # create the arrow hooks
      p[0] = q[0] + 9 * cos(angle + pi / 4)
      p[1] = q[1] + 9 * sin(angle + pi / 4)
      cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
     
      p[0] = q[0] + 9 * cos(angle - pi / 4)
      p[1] = q[1] + 9 * sin(angle - pi / 4)
      cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
      ## [visualization1]
      
      
def rotate_image(image, angle):
      image_center = tuple(np.array(image.shape[1::-1]) / 2)
      rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
      result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
      return result
  
    
  
    
  
"""
three_channel_mean
- Gets 3-channel mean from an image

Input:
img = RGB or BGR image in (cv2 mat)
m = mask (np.uint8)

Output:
[A, B, C], [A_ave, B_ave, C_ave] 

Projects:
FSDT
"""
def three_channel_mean(img, m):
    A,B,C = cv2.split(img)
    # A_ave, B_ave, C_ave = cv2.mean(img, mask=m)[:3]
    # A_ave = round(A_ave, 2)
    # B_ave = round(B_ave, 2)
    # C_ave = round(C_ave, 2)
    A = m*A
    B = m*B
    C = m*C
    A_ave = np.average(A[np.where(A!=0)])
    B_ave = np.average(B[np.where(B!=0)])
    C_ave = np.average(C[np.where(C!=0)])

    return [A, B, C], [A_ave, B_ave, C_ave] 
  
    


  
"""
get_color_info
- Gets RGB and HSV color info from an image

Input:
img = RGB or BGR image in (cv2 mat)
m = mask (np.uint8)
input_format = "BGR" or "RGB"
show_output = 0 or 1

Output:
r_ave, g_ave, b_ave, h_ave, s_ave, v_ave

Projects:
FSDT
"""
def get_color_info(img, m, rgb_ref, input_format="BGR", show_output=0):
    img_cv2 = img.copy()
    h, w, _ = img_cv2.shape

    m = iops.polygon2mask(m, h, w)
    mx = cv2.bitwise_and(img_cv2, img_cv2, mask=m)
    
    if input_format == "BGR":
        rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    else:
        rgb = img_cv2
    hsv = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)

    
    [rr, gg, bb], [r_ave, g_ave, b_ave] = three_channel_mean(rgb, m)
    [hh, ss, vv], [h_ave, s_ave, v_ave] = three_channel_mean(hsv, m)



    # r_delta = []
    # for i in range(len(rgb_ref[0])):
    #     r_delta.append(rgb_ref[0][i] - [r_ave, g_ave, b_ave][i])
    # print(r_delta)

    if show_output == 1:
        plt.figure(dpi=300)
        plt.subplot(2,4,1)
        plt.title("Masked")
        plt.imshow(mx[...,::-1])
        plt.axis(False)
        plt.subplot(2,4,5)
        plt.title("Mask")
        plt.imshow(m, cmap="gray")
        plt.axis(False)
        
        plt.subplot(2,4,2)
        plt.title("R={}".format(r_ave))
        plt.imshow(rr, cmap="gray")
        plt.axis(False)
        plt.subplot(2,4,3)
        plt.title("G={}".format(g_ave))
        plt.imshow(gg, cmap="gray")
        plt.axis(False)
        plt.subplot(2,4,4)
        plt.title("B={}".format(b_ave))
        plt.imshow(bb, cmap="gray")
        plt.axis(False)
        
        plt.subplot(2,4,6)
        plt.title("H={}".format(h_ave))
        plt.imshow(hh, cmap="gray")
        plt.axis(False)
        plt.subplot(2,4,7)
        plt.title("S={}".format(s_ave))
        plt.imshow(ss, cmap="gray")
        plt.axis(False)
        plt.subplot(2,4,8)
        plt.title("V={}".format(v_ave))
        plt.imshow(vv, cmap="gray")
        plt.axis(False)
        
        plt.show()


    return r_ave, g_ave, b_ave, h_ave, s_ave, v_ave


def get_max_contour(m):
    cnts, hierarchy = cv2.findContours(np.asarray(m).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    if len(cnts) > 0:
        max_contour = sorted(cnts, key=cv2.contourArea, reverse= True)[0]
        return max_contour

    

def fit_n_gon(mask, n=4):
    contours, hierarchy = cv2.findContours(mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(contours[0])
    epsilon = 0.1*cv2.arcLength(contours[0],True)
    n_iter, max_iter = 0, 100
    lb, ub = 0., 1.

    while True:
        n_iter += 1
        if n_iter > max_iter:
            return hull, contours

        k = (lb + ub)/2.
        eps = k*cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, eps, True)

        if len(approx) > n:
            lb = (lb + ub)/2.
        elif len(approx) < n:
            ub = (lb + ub)/2.
        else:
            return approx, contours




def show_ngon_lengths(img, n_polygon, n_edges, edge_color=(125,0,0), text_color=(0,255,0)):
    img_output = img.copy()
    cv2.drawContours(img_output, [n_polygon], 0, edge_color, 1)
    for i, sides in enumerate(n_polygon):
        j = i+1
        if i == n_edges-1:
            j = 0
        try:
            c1 = n_polygon[i][0]
            c2 = n_polygon[j][0] 
            dx = int(distance.euclidean(c1, c2))
            
            m = [int((c1[0]+c2[0])/2), int((c1[1]+c2[1])/2)]
            m_text = [int((c1[0]+c2[0])/2)-20, int((c1[1]+c2[1])/2)]
            cv2.circle(img_output, m, radius=3, color=(255, 0, 0), thickness=-1)
            
            cv2.putText(img_output, str(dx), m_text,  cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        except:
            pass
    
    
    return img_output




def orientation_correction(contours, img, fill_contour=1, reshape=True, debug=0):
    img_output = img.copy()
    
    angle = 0
    for i, pts in enumerate(contours):
        area = cv2.contourArea(pts)
        if area < 3700 or 100000 < area:
          continue
        if fill_contour == 1:
            cv2.drawContours(img_output, contours, i, (255, 255, 255), -1)
        else:
            cv2.drawContours(img_output, contours, i, (255, 255, 255), 2)
        

        # Construct a buffer used by the pca analysis
        sz = len(pts)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
          data_pts[i,0] = pts[i,0,0]
          data_pts[i,1] = pts[i,0,1]
       
        # Perform PCA analysis
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
       
        # Store the center of the object
        cntr = (int(mean[0,0]), int(mean[0,1]))
    
        p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
        p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
        angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
        

    angle = int(np.rad2deg(angle))
    

    # Rotate
    img_rotate = ndimage.rotate(img.copy(), angle, reshape=reshape)
    img_mask_rotate = ndimage.rotate(img_output.copy(), angle, reshape=reshape)    


    

    # Unpad the image
    gray = cv2.cvtColor(img_mask_rotate, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 3700 or 100000 < area:
            continue
        (x,y,w,h) = cv2.boundingRect(contour)
    img_rotate_unpadded = img_rotate[y:y+h, x:x+w]
    img_mask_rotate_unpadded = img_mask_rotate[y:y+h, x:x+w]
    
    # Convert to binary
    img_mask_rotate_unpadded = cv2.cvtColor(img_mask_rotate_unpadded, cv2.COLOR_BGR2GRAY)
    _, img_mask_rotate_unpadded = cv2.threshold(img_mask_rotate_unpadded, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    
    # Get centers; needed for coordinate rotation
    org_center = (np.array(img.shape[:2][::-1])-1)/2.
    rot_center = (np.array(img_rotate_unpadded.shape[:2][::-1])-1)/2.


    if debug == 1:
        plt.figure(dpi=150)
        plt.subplot(1,3,1)
        plt.imshow(img)
        plt.subplot(1,3,2)
        plt.imshow(img_rotate_unpadded)
        plt.subplot(1,3,3)
        plt.imshow(img_mask_rotate_unpadded)
        plt.show()
    
    return img_rotate_unpadded, img_mask_rotate_unpadded, angle, org_center, rot_center
 
    
 
"""
get_contour_features(img)
- Gets contour features from the image

Output:
features
"""

def get_contour_features(img):
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
    ratio = round(w/h, 2)
    n_pixels = int(cv2.countNonZero(img))
    
    return ratio, n_pixels


    
def get_contours_and_blob(img):
    if len(img.shape)<3:
        img = cv2.merge([img,img,img])*255
        
    img_output = img.copy()
    gray = cv2.cvtColor(img_output, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for i, pts in enumerate(contours):
        area = cv2.contourArea(pts)
        if area < 3700 or 100000 < area:
            continue
        cv2.drawContours(img_output, contours, i, (255, 255, 255), -1)
    return contours, img_output




'''



'''

def distance_function_skeleton(img):

    skel = skeletonize(img, method='lee')
    skel_cnt, _ = cv2.findContours(skel, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    return skel_cnt
  



"""
half_cut(img, cut=<string>, debug=<boolean>)
- Cuts image into half

Output:
img_cut = original image with cutting lines
half1 = half left/top
half2 = half right/bottom
half1_mask = half left/top mask
half2_mask = half right/bottom mask 
offset = pixel offset after cutting
"""

def half_cut(img, cut="vertical", apply_median=1, debug=0):
    img_cut = img.copy()
    half_x = int(img_cut.shape[1]/2)
    half_y = int(img_cut.shape[0]/2)  
    y = img_cut.shape[0]
    x = img_cut.shape[1]
    
    if cut == "horizontal":
        if debug == 1:
            cv2.line(img_cut, (int(0), int(half_y)), 
                     (img_cut.shape[1], int(half_y)), 
                     (255, 0, 0), 3, cv2.LINE_AA)
    
            
        half1 = img_cut[0:half_y, 0:x]
        tmp = cv2.cvtColor(half1, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(tmp, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if apply_median == 1:
            half1_mask = ndimage.median_filter(mask, size=5)
        else:
            half1_mask = mask
        
        half2 = img_cut[half_y:y, 0:x]
        tmp = cv2.cvtColor(half2, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(tmp, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if apply_median == 1:
            half2_mask = ndimage.median_filter(mask, size=5)
        else:
            half2_mask = mask
        
        
        offset = half_y
        
        
    
    if cut == "vertical":
        if debug == 1:
            cv2.line(img_cut, (half_x, 0), 
                 (half_x, img_cut.shape[1]), 
                 (255, 0, 0), 3, cv2.LINE_AA)
            
        half1 = img_cut[0:y, 0:half_x]
        tmp = cv2.cvtColor(half1, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(tmp, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if apply_median == 1:
            half1_mask = ndimage.median_filter(mask, size=5)
        else:
            half1_mask = mask
        
        
        half2 = img_cut[0:y, half_x:x]
        tmp = cv2.cvtColor(half2, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(tmp, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        if apply_median == 1:
            half2_mask = ndimage.median_filter(mask, size=5)
        else:
            half2_mask = mask
            
        offset = half_x
    
    return img_cut, half1, half2, half1_mask, half2_mask, offset



def segment_by_quantization(img, n, threshold=1):    
    indices = np.arange(0,256)   # List of all colors 
    divider = np.linspace(0,255,n+1)[1] # we get a divider (255/n)
    quantiz = np.int0(np.linspace(0,255,n)) # we get quantization colors
    color_levels = np.clip(np.int0(indices/divider),0,n-1) # color levels 0,1,2..
    palette = quantiz[color_levels] # Creating the palette
    quantized = palette[img]  # Applying palette on image
    quantized = cv2.convertScaleAbs(quantized) # Converting image back to uint8
    
    if threshold == 1:
        if n == 2:
            th, quantized = cv2.threshold(quantized, 254, 255, cv2.THRESH_BINARY)
        if n == 3:
            th, quantized = cv2.threshold(quantized, 126, 255, cv2.THRESH_BINARY)
        if n == 4:
            th, quantized = cv2.threshold(quantized, int(255/3)-1, 255, cv2.THRESH_BINARY)
    return quantized


def get_child_contours(contours, hierarchy):
    
    child_contours = []
    child_hierarchy = []
    for cnt, h in zip(contours, hierarchy[0]):
        # print(h)
        if h[3] != - 1:
            child_contours.append(cnt)
            child_hierarchy.append(h)
    
    return child_contours, child_hierarchy