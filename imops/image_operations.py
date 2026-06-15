import numpy as np
from itertools import groupby
import pycocotools.mask
import pycocotools.mask as mask_util
import cv2
import os
import matplotlib.pyplot as plt
from PIL import ImageColor
import skimage
from scipy.interpolate import splprep, splev
import mlutils.imops.box_converter as bops
import mlutils.shortcuts.shortcuts as shortcuts


 
def colour_science_img_to_cv2(image):
    # Only for color corrections
    import colour
    from colour.hints import cast, NDArray
    image = np.asarray(image)[..., :3]

    if image.dtype == np.uint8:
        return image    
    img = (colour.cctf_encoding(image)*255)[:,:,::-1]

    # Dumb approach but it works...
    cv2.imwrite("tmp.png", img)
    img = cv2.imread("tmp.png")
    return img

def cv2_to_colour_science_img(img):
    # Only for color corrections
    import colour
    return colour.cctf_decoding((img/255)[:,:,::-1].astype(np.float32))



def apply_colour_correction(img, colour_checker_data, debug=0):
    import colour
    """
    apply_colour_correction
    - Applies color correction using a detected colour checker from colour science. 

    Requirements:
    colour_science 0.4.4
    colour_checker_detection latest
    
    Input:
    img = image in cv2 format
    colour_checker_data = obtainable by detect_colour_checkers_segmentation(img_cc, additional_data=True) of colour_checker_detection
    
    Output:
    cv2_img (corrected image)
    """
    if len(colour_checker_data) > 0:
        cc = colour_checker_data[0]
        swatch_colours = cc.swatch_colours
        swatch_masks = cc.swatch_masks
        cc_image = cc.colour_checker
    
        D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
        REFERENCE_COLOUR_CHECKER = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
        REFERENCE_SWATCHES = colour.XYZ_to_RGB(colour.xyY_to_XYZ(list(REFERENCE_COLOUR_CHECKER.data.values())),
                                                'sRGB', 
                                                REFERENCE_COLOUR_CHECKER.illuminant)
        
        if debug:
            # Using the additional data to plot the colour checker and masks.
            masks_i = np.zeros(cc_image.shape)
            for i, mask in enumerate(swatch_masks):
                masks_i[mask[0]:mask[1], mask[2]:mask[3], ...] = 1
            
            colour.plotting.plot_image(colour.cctf_encoding(np.clip(cc_image + masks_i * 0.25, 0, 1)))
            
            colour_checker_rows = REFERENCE_COLOUR_CHECKER.rows
            colour_checker_columns = REFERENCE_COLOUR_CHECKER.columns
            swatches_xyY = colour.XYZ_to_xyY(colour.RGB_to_XYZ(swatch_colours, 'sRGB', D65))
            colour_checker = colour.characterisation.ColourChecker(
                "ok",
                dict(zip(REFERENCE_COLOUR_CHECKER.data.keys(), swatches_xyY)),
                D65, colour_checker_rows, colour_checker_columns)
            colour.plotting.plot_multi_colour_checkers(
                [REFERENCE_COLOUR_CHECKER, colour_checker])
            
            swatches_f = colour.colour_correction(swatch_colours, swatch_colours, REFERENCE_SWATCHES)
            swatches_f_xyY = colour.XYZ_to_xyY(colour.RGB_to_XYZ(swatches_f, 'sRGB', D65))
            colour_checker = colour.characterisation.ColourChecker(
                '{0} - CC'.format("ok"),
                dict(zip(REFERENCE_COLOUR_CHECKER.data.keys(), swatches_f_xyY)),
                D65, colour_checker_rows, colour_checker_columns)
            colour.plotting.plot_multi_colour_checkers([REFERENCE_COLOUR_CHECKER, colour_checker])


        corrected_img = colour.colour_correction(cv2_to_colour_science_img(img), swatch_colours, REFERENCE_SWATCHES)
        # colour.plotting.plot_image(colour.cctf_encoding(corrected_img))
        cv2_img = colour_science_img_to_cv2(corrected_img)
        return cv2_img
    else:
        print("ERROR: Colour checker data error! Now returning the same uncorrected image...")
        return img


    
def rescale_image(img, f=50):
    """
    rescale_image
    - Rescales an image based on a defined rescaling factor
    
    Input:
    img = image in cv2 format
    f = 0-100 (0-100%)
    
    Output:
    rescaled_img
    """
    rescaled_img = img.copy()
    width = int(img.shape[1] * f / 100)
    height = int(img.shape[0] * f / 100)
    dimension = (width, height)
    rescaled_img = cv2.resize(rescaled_img, dimension, interpolation = cv2.INTER_AREA)
    return rescaled_img


def resize_with_padding(img, expected_size):
    """
    resize_with_padding
    - Resizes an image with black padding
    
    Input:
    img = image in cv2 format
    expected_size = (h, w, c)
    
    Output:
    padded_img
    
    Projects:
    FSDT
    """
    h, w, _ = img.shape
    pad_bottom = abs(expected_size[0] - h)
    pad_right = abs(expected_size[1] - w)
    padded_img = cv2.copyMakeBorder(img,0,pad_bottom,0,pad_right,borderType=cv2.BORDER_CONSTANT,value=[0,0,0])
    return padded_img

def rotate_coordinates(kxy, angle, org_center, rot_center):
    """
    rotate_coordinates
    - Rotates coordinates; most useful for keypoints
    
    Input:
    kxy = (x,y) coordinates (int[])
    angle = angle in degrees (float)
    org_center = (x,y) centroid of original image (int[]) 
    rot_center = (x,y) centroid of rotated image (int[])
    
    Output:
    new_kxy = new (x,y) coordinates
    
    
    Projects:
    Masenro3
    """
    kx, ky = kxy
    org = (kx,ky)-org_center
    rad_angle = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(rad_angle) + org[1]*np.sin(rad_angle),
            -org[0]*np.sin(rad_angle) + org[1]*np.cos(rad_angle) ])
    new_kxy = new + rot_center
    return new_kxy

def find_roi(img_cv2, w_out, h_out, x_offset=0, y_offset=0, show_output=0):
    """
    find_roi
    - Crops a certain part of an image using simple binary thresholding for finding the roi
    
    Input:
    i = image filename
    
    
    Output:
    new_kxy = new (x,y) coordinates
    
    
    Projects:
    FSDT
    """
    
    try:
        img = img_cv2.copy()
        img = img[:, x_offset:img_cv2.shape[1]]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to hsv and binarize
        hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        thresh = cv2.inRange(v, 120, 255)
        
        # Detect edges
        edges = cv2.dilate(cv2.Canny(thresh,0,255),None)
        edges = cv2.dilate(edges, (15,15), iterations=75)
        
        # Get contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Get biggest contour
        c = sorted_contours[0]
        mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        masked = cv2.drawContours(mask, [c], -1, 255, -1)
        
        # Get centroid of contour
        M = cv2.moments(c)
        mx = M['m10'] / M['m00']
        my = M['m01'] / M['m00']
        cv2.circle(masked, (round(mx), round(my)), 100, (0, 255, 0), -1)

        # Get displacements
        cx1 = int(mx-(w_out/2))+x_offset
        cx2 = int(mx+(w_out/2))+x_offset
        cy1 = int(my-(h_out/2))+y_offset
        cy2 = int(my+(h_out/2))+y_offset
        
        cx1 = max(0, cx1)
        cy1 = max(0, cy1)
        
        
        roi = img_cv2[cy1:cy2, cx1:cx2]
        roi = resize_with_padding(roi, (h_out, w_out))
        
        
        # result = np.full((target_size_h, target_size_w, 3), (255,0,0), dtype=np.uint8)
        
        # # compute center offset
        # x_center = (target_size_w - img.shape[1]) // 2
        # y_center = (target_size_h - img.shape[0]) // 2
        
        # # copy img image into center of result image
        # result[y_center:y_center+img.shape[0], 
        #        x_center:x_center+img.shape[1]] = roi

        
    
        
        if show_output == 1:
            plt.figure(dpi=300, figsize=(5,5))
            plt.subplot(2, 2, 1)
            plt.imshow(thresh, cmap='gray', vmin=0, vmax=255)
            plt.axis("off")
            plt.subplot(2, 2, 2)
            plt.imshow(edges)
            plt.axis("off")
            plt.subplot(2, 2, 3)
            plt.imshow(masked)
            plt.axis("off")
            plt.subplot(2, 2, 4)
            plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.subplots_adjust(wspace=0.01)
            plt.show()
        
        new_coords = [cx1, cy1, cx2, cy2]
        valid = 1
    except:
        roi = img_cv2
        new_coords = []
        valid = 0
    return roi, new_coords, valid
    



def find_color_reference(img_cv2, color="Red", color_dict=[], show_output=0):
    """
    find_color_reference

    Projects:
    FSDT
    """

    if len(color_dict) == 0:
        color_dict = {"Red": [[0, 50, 50], [10, 255, 255]],
                    "Green": [[50, 190, 140], [60, 210, 150]],
                    "Blue": [[110, 205, 255], [120, 220, 255]]}


    roi_x1 = 500
    roi_x2 = 1400
    roi_y1 = 900
    roi_y2 = 2000
    img = img_cv2[roi_y1:roi_y2, roi_x1:roi_x2]
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)



    # lower mask 
    lower = np.array(color_dict[color][0])
    upper = np.array(color_dict[color][1])
    mask = cv2.inRange(img_hsv, lower, upper)


    # set my output img to zero 
    output_img = img.copy()
    output_img[np.where(mask==0)] = 0

    # for debugging
    # plt.imshow(img[:,:,::-1])
    # plt.imshow(output_img[:,:,::-1])
    # plt.imshow(mask)


    kernel = np.ones((5, 5), np.uint8) 
    mask = cv2.dilate(mask, kernel, iterations=1) 
    contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)                                  

    # for debugging
    # cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    black_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        
        # for debugging
        # cv2.putText(img, "{}x{}".format(w,h), (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 1)    
        # print(x,y,w,h)
        if w < 120 and w > 80 and h < 120 and h > 80:
            reference_contour = c
            cv2.drawContours(image=black_img, contours=[c], contourIdx=-1, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    color_roi = cv2.bitwise_and(black_img, img)
    b = cv2.split(color_roi)[0].max()
    g = cv2.split(color_roi)[1].max()
    r = cv2.split(color_roi)[2].max()

    # plt.figure(dpi=300)
    # plt.imshow(color_roi[:,:,::-1])


    return (r,g,b), color_roi

















def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def box_dims(b):
    w = b[2] - b[0]
    h = b[3] - b[1]
    return w, h

def box_area(b):
    x1, y1, x2, y2 = b
    w = x2-x1
    l = y2-y1
    a = l * w
    return a

def mask2polygon(m):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = np.ascontiguousarray(m)  # some versions of cv2 does not support incontiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    h = res[1]
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
    # We add 0.5 to turn them into real-value coordinate space. A better solution
    # would be to first +0.5 and then dilate the returned polygon by 0.5.
    polygons = [x + 0.5 for x in res if len(x) >= 6]
    return polygons, h

def mask2polygon2(m):
    contours_list = skimage.measure.find_contours(m, level=0.1)
    polygons = []
    for i, contour in enumerate(contours_list):
        polygon_points = skimage.measure.approximate_polygon(contour, tolerance=0.5)
        polygons.append(polygon_points)
    return polygons

def polygon2mask(p, h, w):
    rle = mask_util.frPyObjects(p, int(h), int(w))
    rle = mask_util.merge(rle)
    mask = mask_util.decode(rle)[:, :]
    return mask

def multiclass_nms(outputs, IOU_THRESHOLD):   
    import torch
    instances = outputs["instances"].to("cpu")
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    boxes = instances.pred_boxes.tensor.numpy()
    masks = instances.pred_masks.numpy()

    nms_boxes = boxes.copy()
    final_boxes = []
    final_scores = []
    final_classes = []
    final_masks = []
    for i, b in enumerate(boxes):
        for j, n in enumerate(nms_boxes):
            if n.sum() > 0 and b.sum() > 0:
                iou_val = float(iou(b, n))
                if iou_val > IOU_THRESHOLD:
                    s1 = round(float(scores[i]), 2)
                    s2 = round(float(scores[j]), 2)
                    
                    if s1 >= s2:    
                        my_box = boxes[i]
                        my_score = s1
                        my_mask = masks[i]
                        my_class = classes[i]
                        nms_boxes[j] = torch.tensor([0,0,0,0])
                    if s2 > s1:  
                        my_box = nms_boxes[j]
                        my_score = s2
                        my_class = classes[j]
                        my_mask = masks[j]
                        boxes[i] = torch.tensor([0,0,0,0])
                        
        if any(my_box is d_ for d_ in final_boxes):
            pass
        else:
            final_boxes.append(my_box)
            final_scores.append(my_score)
            final_classes.append(my_class)
            final_masks.append(my_mask)
    return final_boxes, final_scores, final_classes, final_masks


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1]) # 2 to 1
    xB = min(boxA[2], boxB[2]) # 1 to 2
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def iou2(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou

def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = iou2(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def sigmoid(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 1 / (1 + np.exp(-x))










def smooth_contours(contours):
    """_summary_

    Args:
        contours (_type_): _description_

    Returns:
        _type_: _description_
    """
    smoothened = []
    for contour in contours:
        x,y = contour.T
        x = x.tolist()[0]
        y = y.tolist()[0]
        if len(x) > 5 and len(y) > 5:
            tck, u = splprep([x,y], u=None, s=1.0, per=1)
            u_new = np.linspace(u.min(), u.max(), 1000) ## maximum number of contour points: 50
            x_new, y_new = splev(u_new, tck, der=0)
            res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
            smoothened.append(np.asarray(res_array, dtype=np.int32))
    return smoothened


def instance_to_semantic(class_names, img_height, img_width, outputs):
    """_summary_

    Args:
        class_names (_type_): _description_
        img_height (_type_): _description_
        img_width (_type_): _description_
        outputs (_type_): _description_

    Returns:
        _type_: _description_
    """
    class_masks = [0] * len(class_names)
    semantic_mask = np.zeros((img_height, img_width))
    for c, _ in enumerate(class_names):
        mask = np.zeros((img_height, img_width))
        class_masks[c] = mask
        for cl, _, _, m in zip(outputs[0], outputs[1], outputs[2], outputs[3]):          
            if cl == c:
                mask = mask + m/255
                class_masks[cl] = mask * (cl+1)
    semantic_mask = semantic_mask + class_masks[c]
    return semantic_mask, class_masks

def show_img_using_plt(img, pos, name, fontsize=6):
    """_summary_

    Args:
        img (_type_): _description_
        pos (_type_): _description_
        name (_type_): _description_
        fontsize (int, optional): _description_. Defaults to 6.
    """
    # if len(img.shape)<3:
    #     img=cv2.merge([img,img,img])*255
    plt.subplot(3,3,pos)
    plt.title(name, fontsize=fontsize)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def embed_output(img_output, a, cfg, custom_label=""):

    # Required input:
    # s = float
    # b = x1, y1, x2, y2
    # m = 
    # p = 

    if cfg.operation != "semantic_segmentation":
        cl = a.cl - 1
        b = a.box
        s = a.score
        pp = a.polygons
        m = a.mask
        x1, y1, x2, y2 = np.int0(b)
        bw = int(abs(x2-x1))
        bh = int(abs(y2-y1))

        color = ImageColor.getcolor(cfg.display_cfg["colors"][cl], "RGB")
        color = color[::-1] 


        if cfg.display_cfg["show_boxes"]:
            cv2.rectangle(img_output, (x1, y1), (x2, y2), color=color, thickness=cfg.display_cfg["thickness"])

        # Show labels
        if cfg.display_cfg["show_labels"]:

            if cfg.display_cfg["label_level"] >= 1:
                cv2.putText(img_output, "{} ({})".format(str(cfg.class_names[cl]), str(round(s,2))), (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, cfg.display_cfg["label_scale"], color, cfg.display_cfg["thickness"])
                
            if cfg.display_cfg["label_level"] >= 2:
                cv2.putText(img_output, "{} x {}".format(int(bh), int(bw)), (int(x1), int(y1)-35), cv2.FONT_HERSHEY_SIMPLEX, cfg.display_cfg["label_scale"], color, cfg.display_cfg["thickness"])
                cv2.putText(img_output, "x:{} y:{}".format(int(x1), int(y1)), (int(x1), int(y1)-65), cv2.FONT_HERSHEY_SIMPLEX, cfg.display_cfg["label_scale"], color, cfg.display_cfg["thickness"])

            if cfg.display_cfg["label_level"] == 1:
                if custom_label != "":
                    cv2.putText(img_output, str(custom_label), (int(x1), int(y1)-105), cv2.FONT_HERSHEY_SIMPLEX, cfg.display_cfg["label_scale"], color, cfg.display_cfg["thickness"])



        if cfg.display_cfg["show_contour"] and cfg.operation != "object_detection":


            polygon_lengths = [len(px) for px in pp]

            if cfg.display_cfg["contour_style"] == "simple":     
                try:   
                    max_polygon = pp[polygon_lengths.index(max(polygon_lengths))]
                    px = [max_polygon.astype(np.int64).reshape(-1, 2)]
                    cv2.drawContours(img_output, px, 0, color, cfg.display_cfg["thickness"]*2)
                except:
                    pass

            if cfg.display_cfg["contour_style"] == "complex":
                for polygon in pp:
                    px = np.asarray(polygon).astype(np.int64).reshape(-1, 2)
                    cv2.drawContours(img_output, [px], -1, color, cfg.display_cfg["thickness"])



            if cfg.display_cfg["show_contour_alpha"]:
                colored_mask = np.expand_dims(m, 0).repeat(3, axis=0)
                colored_mask = np.moveaxis(colored_mask, 0, -1)
                masked = np.ma.MaskedArray(img_output, mask=colored_mask, fill_value=color)
                image_overlay = masked.filled()
                img_output = cv2.addWeighted(img_output, 1 - cfg.display_cfg["output_alpha"], image_overlay, cfg.display_cfg["output_alpha"], 0)


    else:
        cl = a.cl - 1
        color = ImageColor.getcolor(cfg.display_cfg["colors"][cl], "RGB")
        color = color[::-1] 
        
        m = a.mask
        m = cv2.resize(m.astype(np.uint8), (img_output.shape[1], img_output.shape[0]))*255
        polygons, _ = mask2polygon(m)

        if len(polygons) > 0:
            polygon_lengths = [len(p) for p in polygons]
            max_polygon = polygons[polygon_lengths.index(max(polygon_lengths))]
            px = [max_polygon.astype(np.int64).reshape(-1, 2)]
            for polygon in polygons:
                px = polygon.astype(np.int64).reshape(-1, 2)
                if cfg.display_cfg["show_contour"]:
                    cv2.drawContours(img_output, [px], 0, color, cfg.display_cfg["thickness"])
                    
            if cfg.display_cfg["show_contour_alpha"]:
                colored_mask = np.expand_dims(m, 0).repeat(3, axis=0)
                colored_mask = np.moveaxis(colored_mask, 0, -1)
                masked = np.ma.MaskedArray(img_output, mask=colored_mask, fill_value=color)
                image_overlay = masked.filled()
                img_output = cv2.addWeighted(img_output, 1 - cfg.display_cfg["output_alpha"], image_overlay, cfg.display_cfg["output_alpha"], 0)



    return img_output

def display_outputs(file_name, img_output, img_output_post=None, show_post_process=0):
    # Display output
    if show_post_process == 0:
        img_copy = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB).copy()
        plt.imshow(img_copy)
        plt.axis("off")
        plt.title(os.path.basename(file_name), fontsize=8)
        wm = plt.get_current_fig_manager()
        wm.window.showMaximized()
        plt.show(block=True)

    if show_post_process == 1:
        img_copy = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB).copy()
        img_copy2 = cv2.cvtColor(img_output_post, cv2.COLOR_BGR2RGB).copy()

        plt.subplot(1, 2, 1)
        plt.imshow(img_copy)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(img_copy2)
        plt.axis("off")
        
        wm = plt.get_current_fig_manager()
        wm.window.showMaximized()
        plt.show(block=True)


def grab_objects(img, preds, cfg):
    # if cfg.operation == "object_detection":
    #     for p in preds:
    #         cl = p.cl - 1
    #         b = p.box
    #         # b = bops.xywh2xyxy(b)


    #         class_name = cfg.class_names[cl]
    #         b = list(map(int, b))
    #         x1, x2, y1, y2 = b


    #         if cfg.crop_style == "fixed":
    #             cs = int(cfg.crop_size/2)
    #             mx = int((x2+x1)/2)
    #             my = int((y2+y1)/2)

    #             cx1 = max(0, int(mx - cs) - cfg.crop_pad) 
    #             cy1 = max(0, int(my - cs) - cfg.crop_pad)
    #             cx2 = min(img.shape[1], int(mx + cs) + cfg.crop_pad)
    #             cy2 = min(img.shape[0], int(my + cs) + cfg.crop_pad)
                

    #             cropped_img = img[cy1:cy2, cx1:cx2]

    #             if cropped_img.shape[0] == cs*2 and cropped_img.shape[1] == cs*2:
    #                 cropped_filename = os.path.basename(file_name)
    #                 cropped_filename = os.path.splitext(cropped_filename)[0]
    #                 cropped_filename = "{}_{}_{}.png".format(cropped_filename, cx1, cy1)    
    #                 cv2.imwrite("{}/{}/{}".format(cfg.experiment_folder_models_target_model_ver_weights_cropped, class_name, cropped_filename), cropped_img)


    #         elif cfg.crop_style == "predicted":
    #             cx1 = x1
    #             cy1 = y1
    #             cx2 = x2 
    #             cy2 = y2

    #             if cfg.crop_square_crop:
    #                 m1 = int((x2+x1)/2)
    #                 m2 = int((y2+y1)/2)
    #                 lx = int((x2-x1)/2)
    #                 wy = int((y2-y1)/2)
    #                 lorw = max(lx, wy)
    #                 cx1 = int(m1 - lorw)
    #                 cx2 = int(m1 + lorw)
    #                 cy1 = int(m2 - lorw)
    #                 cy2 = int(m2 + lorw)

    #             cropped_img = img[cy1:cy2, cx1:cx2]

                
    #             cropped_filename = os.path.basename(file_name)
    #             cropped_filename = os.path.splitext(cropped_filename)[0]
    #             cropped_filename = "{}_{}_{}.png".format(cropped_filename, cx1, cy1)    
    #             cv2.imwrite("{}/{}/{}".format(cfg.experiment_folder_models_target_model_ver_weights_cropped, class_name, cropped_filename), cropped_img)

    cropped_objects = []
    if cfg.operation == "instance_segmentation":
        black = np.zeros((img.shape[0] , img.shape[1], 3), dtype = "uint8")

        for p in preds:
            cl = p.cl - 1
            b = p.box
            m = p.mask
            b = list(map(int, b))
            x1, y1, x2, y2 = b

            mask_array = (np.asarray(m*1)).astype(np.uint8)
            mask_shaped = mask_array.reshape(img.shape[0], img.shape[1], 1)
            mask_out = cv2.bitwise_and(img, img, mask=mask_shaped)
            output_black = cv2.bitwise_or(black, mask_out)


            if cfg.crop_style == "fixed":
                cs = int(cfg.crop_size/2)
                mx = int((x2+x1)/2)
                my = int((y2+y1)/2)

                cx1 = max(0, int(mx - cs))
                cx2 = min(img.shape[1], int(mx + cs))
                cy1 = max(0, int(my - cs))
                cy2 = min(img.shape[0], int(my + cs))
                    
                if cfg.crop_transparent == 0:
                    cropped_img = img[cy1:cy2, cx1:cx2]
                if cfg.crop_transparent == 1:
                    cropped_img = output_black[cy1:cy2, cx1:cx2]
                if cfg.crop_transparent == 2:
                    cropped_img = output_black[cy1:cy2, cx1:cx2]
                    tmp = cropped_img.copy()
                    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
                    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
                    b, g, r = cv2.split(cropped_img)
                    rgba = [b,g,r, alpha]
                    cropped_img = cv2.merge(rgba,4)


            elif cfg.crop_style == "predicted":
                if cfg.crop_square_crop == True:
                    m1 = int((x2+x1)/2)
                    m2 = int((y2+y1)/2)
                    lx = int((x2-x1)/2)
                    wy = int((y2-y1)/2)
                    lorw = max(lx, wy)
                    cx1 = int(m1 - lorw)
                    cx2 = int(m1 + lorw)
                    cy1 = int(m2 - lorw)
                    cy2 = int(m2 + lorw)

                cx1 = max(0, x1)
                cx2 = min(img.shape[1], x2)
                cy1 = max(0, y1)
                cy2 = min(img.shape[0], y2)

                if cfg.crop_transparent == 0:
                    cropped_img = img[cy1:cy2, cx1:cx2]
                if cfg.crop_transparent == 1:
                    cropped_img = output_black[cy1:cy2, cx1:cx2]
                if cfg.crop_transparent == 2:
                    try:
                        cropped_img = output_black[cy1:cy2, cx1:cx2]
                        tmp = cropped_img.copy()
                        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
                        _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
                        b, g, r = cv2.split(cropped_img)
                        rgba = [b,g,r, alpha]
                        cropped_img = cv2.merge(rgba,4)
                    except:
                        pass
                
            obj_dict = {"img": cropped_img, "offsets": [cx1, cx2, cy1, cy2]}
            cropped_objects.append(obj_dict)
    return cropped_objects

def crop_images(img, preds, file_name, cfg):

    for cl in cfg.class_names:
        try:
            os.mkdir("{}/{}".format(cfg.experiment_folder_models_target_model_ver_weights_cropped, cl))
        except:
            pass


    if cfg.operation == "object_detection":
        for p in preds:
            cl = p.cl - 1
            b = p.box
            # b = bops.xywh2xyxy(b)


            class_name = cfg.class_names[cl]
            b = list(map(int, b))
            x1, x2, y1, y2 = b


            if cfg.crop_style == "fixed":
                cs = int(cfg.crop_size/2)
                mx = int((x2+x1)/2)
                my = int((y2+y1)/2)

                cx1 = max(0, int(mx - cs) - cfg.crop_pad) 
                cy1 = max(0, int(my - cs) - cfg.crop_pad)
                cx2 = min(img.shape[1], int(mx + cs) + cfg.crop_pad)
                cy2 = min(img.shape[0], int(my + cs) + cfg.crop_pad)
                

                cropped_img = img[cy1:cy2, cx1:cx2]

                if cropped_img.shape[0] == cs*2 and cropped_img.shape[1] == cs*2:
                    cropped_filename = os.path.basename(file_name)
                    cropped_filename = os.path.splitext(cropped_filename)[0]
                    cropped_filename = "{}_{}_{}.png".format(cropped_filename, cx1, cy1)    
                    cv2.imwrite("{}/{}/{}".format(cfg.experiment_folder_models_target_model_ver_weights_cropped, class_name, cropped_filename), cropped_img)


            elif cfg.crop_style == "predicted":
                cx1 = x1
                cy1 = y1
                cx2 = x2 
                cy2 = y2

                if cfg.crop_square_crop:
                    m1 = int((x2+x1)/2)
                    m2 = int((y2+y1)/2)
                    lx = int((x2-x1)/2)
                    wy = int((y2-y1)/2)
                    lorw = max(lx, wy)
                    cx1 = int(m1 - lorw)
                    cx2 = int(m1 + lorw)
                    cy1 = int(m2 - lorw)
                    cy2 = int(m2 + lorw)

                cropped_img = img[cy1:cy2, cx1:cx2]

                
                cropped_filename = os.path.basename(file_name)
                cropped_filename = os.path.splitext(cropped_filename)[0]
                cropped_filename = "{}_{}_{}.png".format(cropped_filename, cx1, cy1)    
                cv2.imwrite("{}/{}/{}".format(cfg.experiment_folder_models_target_model_ver_weights_cropped, class_name, cropped_filename), cropped_img)


    if cfg.operation == "instance_segmentation":
        black = np.zeros((img.shape[0] , img.shape[1], 3), dtype = "uint8")

        for p in preds:
            cl = p.cl - 1
            b = p.box
            # b = bops.xywh2xyxy(b)
            m = p.mask


            class_name = cfg.class_names[cl]
            b = list(map(int, b))
            x1, y1, x2, y2 = b

            mask_array = (np.asarray(m*1)).astype(np.uint8)
            mask_shaped = mask_array.reshape(img.shape[0], img.shape[1], 1)
            mask_out = cv2.bitwise_and(img, img, mask=mask_shaped)
            output_black = cv2.bitwise_or(black, mask_out)


            if cfg.crop_style == "fixed":
                cs = int(cfg.crop_size/2)
                mx = int((x2+x1)/2)
                my = int((y2+y1)/2)

                cx1 = max(0, int(mx - cs))
                cx2 = min(img.shape[1], int(mx + cs))
                cy1 = max(0, int(my - cs))
                cy2 = min(img.shape[0], int(my + cs))
                    
                if cfg.crop_transparent == 0:
                    cropped_img = img[cy1:cy2, cx1:cx2]
                if cfg.crop_transparent == 1:
                    cropped_img = output_black[cy1:cy2, cx1:cx2]
                if cfg.crop_transparent == 2:
                    cropped_img = output_black[cy1:cy2, cx1:cx2]
                    tmp = cropped_img.copy()
                    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
                    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
                    b, g, r = cv2.split(cropped_img)
                    rgba = [b,g,r, alpha]
                    cropped_img = cv2.merge(rgba,4)

                if cropped_img.shape[0] == cs*2 and cropped_img.shape[1] == cs*2:
                    cropped_filename = os.path.basename(file_name)
                    cropped_filename = os.path.splitext(cropped_filename)[0]
                    cropped_filename = "{}_{}_{}.png".format(cropped_filename, cx1, cy1)    
                    cv2.imwrite("{}/{}/{}".format(cfg.experiment_folder_models_target_model_ver_weights_cropped, class_name, cropped_filename), cropped_img)


            elif cfg.crop_style == "predicted":


                if cfg.crop_square_crop == True:
                    m1 = int((x2+x1)/2)
                    m2 = int((y2+y1)/2)
                    lx = int((x2-x1)/2)
                    wy = int((y2-y1)/2)
                    lorw = max(lx, wy)
                    cx1 = int(m1 - lorw)
                    cx2 = int(m1 + lorw)
                    cy1 = int(m2 - lorw)
                    cy2 = int(m2 + lorw)

                cx1 = max(0, x1)
                cx2 = min(img.shape[1], x2)
                cy1 = max(0, y1)
                cy2 = min(img.shape[0], y2)

                if cfg.crop_transparent == 0:
                    cropped_img = img[cy1:cy2, cx1:cx2]
                if cfg.crop_transparent == 1:
                    cropped_img = output_black[cy1:cy2, cx1:cx2]
                if cfg.crop_transparent == 2:
                    try:
                        cropped_img = output_black[cy1:cy2, cx1:cx2]
                        tmp = cropped_img.copy()
                        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
                        _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
                        b, g, r = cv2.split(cropped_img)
                        rgba = [b,g,r, alpha]
                        cropped_img = cv2.merge(rgba,4)
                    except:
                        pass
                
                if cropped_img.shape[0] != 0 and cropped_img.shape[1] != 0:
                    cropped_filename = os.path.basename(file_name)
                    cropped_filename = os.path.splitext(cropped_filename)[0]
                    cropped_filename = "{}_{}_{}.png".format(cropped_filename, cx1, cy1)    
                    cv2.imwrite("{}/{}/{}".format(cfg.experiment_folder_models_target_model_ver_weights_cropped, class_name, cropped_filename), cropped_img)