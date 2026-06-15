import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import ImageColor

class ann_data():
    def __init__(self):
        
        self.cl = 0

        # x1, y1, x2, y2
        self.box = []
        self.centroid = []
        self.polygons = []

        self.mask = []

        self.area = 0
        
        self.keypoint = []
        self.nkeypoint = 0
        self.crowd = 0
        self.score = 0
        
        self.tag = []
        

        self.custom = {}

def get_child_contours(contours, hierarchy):
    
    child_contours = []
    child_hierarchy = []
    for cnt, h in zip(contours, hierarchy[0]):
        # print(h)
        if h[3] != - 1:
            child_contours.append(cnt)
            child_hierarchy.append(h)
    
    return child_contours, child_hierarchy

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

def box_dims(b):
    w = b[2] - b[0]
    h = b[3] - b[1]
    return w, h

def mask2polygon(m):
    mask = np.ascontiguousarray(m)  
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    h = res[1]
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    polygons = [x + 0.5 for x in res if len(x) >= 6]
    return polygons, h

def embed_output(img_output, a, cfg, custom_label=""):
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

def visualize_annotations(img, anns, cfg, custom_label="", ignore_classes=[], show_output=True):
    if show_output == True:
        show_output = cfg.display_cfg["show_output_window"]

    anns_clone = anns.copy()

    img_output = img.copy()
    h, w, _ = img_output.shape
    anns_clone.reverse()

    for a in anns_clone:
        if a.cl not in ignore_classes:
            img_output = embed_output(img_output, 
                                        a, 
                                        cfg, 
                                        custom_label)
        
    if show_output == 1:
        plt.figure(dpi=150)
        plt.imshow(img_output[...,::-1])
        wm = plt.get_current_fig_manager()
        wm.window.showMaximized()
        plt.show(block=True)
    return img_output

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
