import matplotlib.pyplot as plt
import cv2
import torch
import time
import numpy as np
import os 
import pandas as pd

from imutils import perspective
from scipy.spatial import distance as dist
from tqdm import tqdm

import stomatalia_tools.utils.detectron2 as d2
import stomatalia_tools.config as config
import stomatalia_tools.utils.shortcuts as s
import stomatalia_tools.utils.tools as tools

import requests
import math

class data_saver():
    def __init__(self):
        
        self.columns = ["Basename",                     "Filename",                     "Elapsed_time",                 "Known_distance_um",           "Known_distance_px",           "Pixel_ratio",
                        "Image_resolution_l_px",        "Image_resolution_w_px",
                        "Observation_size_l_um",        "Observation_size_w_um",        "Observation_area_um2",
                        "N_stomata",                    "N_pavement_cells", 
                        "Min_stomata_area_px",          "Ave_stomata_area_px",          "Max_stomata_area_px",          "Std_stomata_area_px",
                        "Min_stomata_area_um2",         "Ave_stomata_area_um2",         "Max_stomata_area_um2",         "Std_stomata_area_um2",
                        "Min_stomata_ratio",            "Ave_stomata_ratio",            "Max_stomata_ratio",            "Std_stomata_ratio",

                        "Min_stomata_aperture_w_px",    "Ave_stomata_aperture_w_px",    "Max_stomata_aperture_w_px",    "Std_stomata_aperture_w_px",
                        "Min_stomata_aperture_l_px",    "Ave_stomata_aperture_l_px",    "Max_stomata_aperture_l_px",    "Std_stomata_aperture_l_px",
                        "Min_stomata_aperture_w_um",    "Ave_stomata_aperture_w_um",    "Max_stomata_aperture_w_um",    "Std_stomata_aperture_w_um",
                        "Min_stomata_aperture_l_um",    "Ave_stomata_aperture_l_um",    "Max_stomata_aperture_l_um",    "Std_stomata_aperture_l_um",

                        "Min_pavement_cell_area_px",    "Ave_pavement_cell_area_px",    "Max_pavement_cell_area_px",    "Std_pavement_cell_area_px", 
                        "Min_pavement_cell_area_um2",   "Ave_pavement_cell_area_um2",   "Max_pavement_cell_area_um2",   "Std_pavement_cell_area_um2", 

                        "Stomatal_density_s_um2",       "Stomatal_index"]
        
        metadata = {"Columns" : ["Known_distance_um",
                        "Known_distance_px",
                        "Pixel_ratio",
                        "Image_resolution_l px",
                        "Image_resolution_w_px",
                        "Observation_area_l_um", 
                        "Observation_area_w_px",
                        "N_stomata",
                        "N_pavement_cells",
                        "Min/ave/max/std_stomata_area_px",
                        "Min/ave/max/std_stomata_area_um2",
                        "Min/ave/max/std_stomata_ratio",
                        "Min/ave/max/std_stomata_aperture_w_px", 
                        "Min/ave/max/std_stomata_aperture_l_px", 
                        "Min/ave/max/std_stomata_aperture_w_um", 
                        "Min/ave/max/std_stomata_aperture_l_um", 
                        "Min/ave/max/std_pavement_cell_area_px", 
                        "Min/ave/max/std_pavement_cell_area_um",
                        "Stomatal density",
                        "Stomatal index"],

                         "Descriptions": ["reference distance in pixels for pixel to um conversion",
                                "reference distance in um for pixel to um conversion",
                                "multiplier for converting px to um",
                                "image length in pixels",
                                "image width in pixels",
                                "actual image length in um",
                                "actual image width in um",
                                "number of stomata detected",
                                "number of pavement cells detected",
                                "stomata area in pixels^2",
                                "stomata area in um^2",
                                "stomata ratio (L/W)",
                                "aperture width in pixels",
                                "aperture length in pixels",
                                "aperture width in um",
                                "aperture length in um",
                                "pavement cell area in pixels",
                                "pavement cell area in um",
                                "N_stomata/um^2",
                                "N_stomata/N_stomata+N_pavement_cells) * 100"]}
                        


        self.df = pd.DataFrame(columns=self.columns)
        self.definitions_df = pd.DataFrame(metadata)
       
    def append_data(self, data):
        series_data = pd.Series(data=data, index=self.columns)
        self.df = self.df.append(series_data, ignore_index=True)
        # self.df = pd.concat([self.df, pd.DataFrame(series_data)], ignore_index=True)

    
    def save_data(self, filename):
        writer = pd.ExcelWriter(filename, engine = 'xlsxwriter')
        self.df.to_excel(writer, sheet_name="Data")
        self.definitions_df.to_excel(writer, sheet_name="Metadata")
        writer.close()

class algorithm:

    def __init__(self, cfg_filename):
        self.cfg = config.config(cfg_filename=cfg_filename, target_model="d2_insseg")
        url =  self.cfg.cfg["models"]["d2_insseg"]["inference"]["model_weights"]
        output_path = "model.pth"

        if not os.path.exists(output_path):
            for i in range(10):
                r = requests.get(url, stream=True)

                print("Attempt", i+1, "Status:", r.status_code)
                r.raise_for_status()
                total_size = int(r.headers.get("Content-Length", 0))

                if r.status_code == 200:
                    with open(output_path, "wb") as f, tqdm(
                                total=total_size,
                                unit="B",
                                unit_scale=True,    
                                desc="Downloading"
                            ) as bar:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                bar.update(len(chunk))
                    break

        self.cfg.inference_model_weights = output_path
        self.detectron2_model = d2.detectron2_model(self.cfg, "d2_insseg")

    def preprocess_image(self, img):
        
        image_raw = cv2.resize(img, (256, 256))
        img_yuv = cv2.cvtColor(image_raw, cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        image_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab) 
        # image_rgb = cv2.GaussianBlur(image_rgb, (25, 25), 0)
        image_rgb = cv2.medianBlur(image_rgb, 5)
        image_gray = image_rgb[:,:,0]
        
        return image_raw, image_rgb, image_gray

    def measure_object_area(self, p, max_polygon, pixel_ratio):
        box = cv2.minAreaRect(max_polygon.astype(np.int64).reshape(-1, 2))
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = (tl+tr)/2
        (blbrX, blbrY) = (bl+br)/2
        (tlblX, tlblY) = (tl+bl)/2
        (trbrX, trbrY) = (tr+br)/2
            
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        length_px = int(dA)
        width_px = int(dB)
        length_um = int(length_px*pixel_ratio)
        width_um = int(width_px*pixel_ratio)
        area_px = math.pi * (length_px/2) * (width_px/2)
        area_um = math.pi * (length_um/2) * (width_um/2)

        p.area_um = area_um
        p.area_px = area_px
        p.ratio = length_px / width_px

        return p

    def measure_aperture_size(self, raw_img, img, p, pixel_ratio):
        size = 5
        iterations = 2

        raw_img = cv2.resize(raw_img, (256, 256))
        

        raw, rgb, gray = self.preprocess_image(img)
        output = tools.segment_by_quantization(rgb, n=4, threshold=1)
        output = cv2.morphologyEx(output.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((size, size), np.uint8), iterations=iterations)[:,:,0]
        rgb_clone = rgb.copy()
        contours, hierarchy = cv2.findContours(output.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        bh, bw = tools.box_dims(p.box)
        cx = int(bw/2)
        cy = int(bh/2)
        
        rw = 0
        rh = 0
        if len(contours) > 0:
            contours, hierarchy = tools.get_child_contours(contours, hierarchy)
            
            if len(contours) > 0:    
                rw_vals = []
                rh_vals = [] 
                cnt_vals = []
                r_vals = []
                dist_vals = []


                for cnt, h in zip(contours, hierarchy[0]):
                    r = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(r)
                    box = np.uint8(box)


                    x, y = r[0]
                    rw = int(min(r[1][0], r[1][1]))
                    rh = int(max(r[1][0], r[1][1]))

                    dist_val = int(dist.euclidean((cx, cy), (x, y)))

                    


                    rw_vals.append(rw)
                    rh_vals.append(rh)
                    cnt_vals.append(cnt)
                    r_vals.append(r)
                    dist_vals.append(dist_val)
                    


                rw = max(rw_vals)
                rh = max(rh_vals)
                index = rw_vals.index(rw)
                cnt = cnt_vals[index]
                r = r_vals[index]
                dist_val = dist_vals[index]

                if rw >= 10 and rh >= 10 and dist_val <= 70:           
                    cv2.drawContours(raw_img, [cnt], -1, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.drawContours(raw_img, [cnt], -1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    rw = 0
                    rh = 0
                    
        
        p.aperture_w_px = rw
        p.aperture_l_px = rh
        p.aperture_w_um = int(rw * pixel_ratio)
        p.aperture_l_um = int(rh * pixel_ratio)
        return p, raw_img

    def process_image(self, file_name=None, img=None, scale=20):
        bname = os.path.basename(file_name)


        # Perform inference on the image
        start = time.perf_counter() 

        img = cv2.imread(file_name)
        
        if img.shape[1] != 2560 and img.shape[0] != 1920:
            img = cv2.resize(img, (2560, 1920))

        results = self.detectron2_model.process_image(file_name=file_name)
        print(len(results["output"]["preds"]))
        self.img_output_final = img.copy()
        elapsed_time = (time.perf_counter() - start)
        img_l, img_w, _ = img.shape


        


        img_output_post = img.copy()
        stomata_sizes_px = []
        stomata_sizes_um = []
        stomata_ratios = []
        stomata_aperture_sizes_w_px = []
        stomata_aperture_sizes_l_px = []
        stomata_aperture_sizes_w_um = []
        stomata_aperture_sizes_l_um = []
        pavement_cell_sizes_px = []
        pavement_cell_sizes_um = []
        
        

        # 0.483 = 100um/(672-465)
        # 0.1092 = 20um/(2483-2300)
        # 0.259 = 50um/(193)
        if scale == 100:
            pixels = 207
        elif scale == 20:
            pixels = 183
        elif scale == 50:
            pixels = 193
        else:
            pixels = 183

        known_distance = scale
        pixel_ratio = round(known_distance/pixels, 4)


        start = time.perf_counter()
        for i, p in enumerate(results["output"]["preds"]):   
            cl = p.cl-1
            print(cl, p.box)
        
            polygons = p.polygons
            polygon_lengths = [len(p) for p in polygons]
            max_polygon = polygons[polygon_lengths.index(max(polygon_lengths))]
            if cl == 0:
                

                # Get object area
                p = self.measure_object_area(p, max_polygon, pixel_ratio)


                b = list(map(int, p.box))
                x1, y1, x2, y2 = b
                m = p.mask
                black = np.zeros((img.shape[0] , img.shape[1], 3), dtype = "uint8")
                mask_array = (np.asarray(m*1)).astype(np.uint8)
                mask_shaped = mask_array.reshape(img.shape[0], img.shape[1], 1)
                mask_out = cv2.bitwise_and(img, img, mask=mask_shaped)
                output_black = cv2.bitwise_or(black, mask_out)
                cropped_img_raw = img[y1:y2, x1:x2]
                cropped_img_blk = output_black[y1:y2, x1:x2]
                p, img_aperture = self.measure_aperture_size(cropped_img_raw, cropped_img_blk, p, pixel_ratio)

                # Display aperture
                self.img_output_final[y1:y2, x1:x2] = cv2.resize(img_aperture, (x2-x1, y2-y1))
                cv2.putText(self.img_output_final, "{} ({})".format(self.cfg.class_names[cl], round(p.score, 2)), (int(x1), int(y1)-60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), self.cfg.display_cfg["thickness"])
                cv2.putText(self.img_output_final, "AL:{} AW:{}".format(p.aperture_l_px, p.aperture_w_px), (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), self.cfg.display_cfg["thickness"])
                

                if p.area_um != 0:
                    stomata_sizes_px.append(p.area_px)
                    stomata_sizes_um.append(p.area_um)
                    stomata_ratios.append(p.ratio)
                    stomata_aperture_sizes_l_px.append(p.aperture_w_px)
                    stomata_aperture_sizes_w_px.append(p.aperture_l_px)
                    stomata_aperture_sizes_l_um.append(p.aperture_w_um)
                    stomata_aperture_sizes_w_um.append(p.aperture_l_um)

            if cl == 1:            
                area_px = cv2.contourArea(max_polygon.astype(np.int64).reshape(-1, 2))
                area_um = int(area_px*pixel_ratio)

                p.area_um = area_um
                p.area_px = area_px

                if p.area_um != 0:
                    pavement_cell_sizes_px.append(p.area_px)
                    pavement_cell_sizes_um.append(p.area_um)


        
        


        self.img_output_final = tools.visualize_annotations(img=self.img_output_final, 
                                                                 anns=results["output"]["preds"],
                                                                 cfg=self.cfg)   







        postprocess_elapsed_time = (time.perf_counter() - start)
        total_elapsed_time = elapsed_time + postprocess_elapsed_time
        total_elapsed_time = round(total_elapsed_time, 2)
        file_name = os.path.splitext(os.path.basename(file_name))[0]


        obs_size_l_um = round(img_l * pixel_ratio, 2)
        obs_size_w_um = round(img_w * pixel_ratio, 2)
        obs_size_area_um2 = obs_size_l_um * obs_size_w_um
        n_stomata = results["output"]["class_total"][0]
        n_pavement_cells = results["output"]["class_total"][1]


        stomata_size_px_stats = s.compute_statistics(stomata_sizes_px)
        stomata_size_um_stats = s.compute_statistics(stomata_sizes_um)
        stomata_ratio_stats = s.compute_statistics(stomata_ratios)
        stomata_aperture_size_w_px_stats = s.compute_statistics(stomata_aperture_sizes_w_px)
        stomata_aperture_size_l_px_stats = s.compute_statistics(stomata_aperture_sizes_l_px)
        stomata_aperture_size_w_um_stats = s.compute_statistics(stomata_aperture_sizes_w_um)
        stomata_aperture_size_l_um_stats = s.compute_statistics(stomata_aperture_sizes_l_um)
        pavement_cell_size_px_stats = s.compute_statistics(pavement_cell_sizes_px)
        pavement_cell_size_um_stats = s.compute_statistics(pavement_cell_sizes_um)


        try:
            stomatal_density = n_stomata / obs_size_area_um2
            stomatal_index = n_stomata / (n_stomata + n_pavement_cells) * 100
        except Exception as e:
            stomatal_density = 0
            stomatal_index = 0
            print(e)


        data_output = [file_name, bname, total_elapsed_time, known_distance, pixels, pixel_ratio,
                        img_l, img_w,
                        obs_size_l_um, obs_size_w_um, obs_size_area_um2,
                        n_stomata, n_pavement_cells, 
                        *stomata_size_px_stats,
                        *stomata_size_um_stats,
                        *stomata_ratio_stats,
                        *stomata_aperture_size_w_px_stats,
                        *stomata_aperture_size_l_px_stats,
                        *stomata_aperture_size_w_um_stats,
                        *stomata_aperture_size_l_um_stats,
                        *pavement_cell_size_px_stats,
                        *pavement_cell_size_um_stats,
                        stomatal_density, stomatal_index]
        

        final_results = {"img": img, "output": {
                                    "img_output": self.img_output_final, 
                                    "preds": results["output"]["preds"], 
                                    "class_total": results["output"]["class_total"],
                                    "data_output":data_output, 
                                    "elapsed_time": elapsed_time}}



        return final_results



