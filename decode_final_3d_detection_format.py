# conda activate centernew
import sys
import os 
import numpy as np
from tqdm import tqdm
import cv2

# *************************************************
# Info:  Decoder for ddd_lumpi_wo_8bins_w_offsets #
# *************************************************
PROJECT_PATH = '/home/qingwu/CodeSecondProjects/KITTI/KITTI_centernet2d'
CENTERNET_PATH = os.path.join(PROJECT_PATH, 'src/lib')
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts import opts


MODEL_PATH = os.path.join(PROJECT_PATH, 'models/kitti_centernet3d.pth')
TASK = 'ddd'
DATASET = 'kitti'
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
print(f'opt: {opt}')
detector = detector_factory[opt.task](opt)
save_path = os.path.join(PROJECT_PATH, 'results')
task = 'kitti_ctdet_3d'
box_colors = {1 : (0, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255)}
cls_objects = [1, 2, 3]
cls_objects_num2str = {'1': 'Pedestrian', '2': 'Car', '3': 'Cyclist'}

image_num = [154, 447, 220, 140, 314, 297, 270, 800, 390, 803, 294, 373, 78, 340, 106, 376, 209, 145, 339, 1059, 837]


save_image = False
image_path = '/media/disk3/Data-qingwu/dataset/kitti_tracking/training/image_02'
# sequence_index = 0 
for sequence_index in range(21):

    vis_save_path_3d = os.path.join(save_path, task, f'{"%04d" % sequence_index}', 'output_images')
    txt_save_path_3d = os.path.join(save_path, task, f'{"%04d" % sequence_index}', f'results_{"%04d" % sequence_index}')

    txt_tracking_path_2d= os.path.join(save_path, task, f'{DATASET}_{task}_result_final.txt')

    check_path = [save_path, vis_save_path_3d, txt_save_path_3d]

    for tmp_path in check_path:
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path, exist_ok=True)

                
    for image_index in tqdm(range(image_num[sequence_index])):
        txt_tmp_path = os.path.join(txt_save_path_3d, "%06d.txt" %image_index)
        img =  f'{image_path}/{"%04d" % sequence_index}/{"%06d" %image_index}.png'
        image = cv2.imread(img)
        ret = detector.run(img)['results']
        print(f'==> Keys of ret: {ret.keys()}')
        # Deal with predictions
        with open(txt_tmp_path, 'w') as file_detection:
            for cls_object in cls_objects:
                box_color = box_colors[cls_object]
                text_color = (0, 255, 255) 
                results = ret[cls_object]
                # print(f'Number of {cls_objects_num2str[str(cls_object)]}: {len(results)}')
                for index in range(len(results)):
                    result_index = results[index]
                    # print(f'==> len of result_index: {len(result_index)}')
                    # print(f'==> len of result_index: {result_index}')
                    cls_str = [cls_objects_num2str[str(cls_object)]] # cls
                    alpha = result_index[0]
                    x1, y1, x2, y2 = result_index[1], result_index[2], result_index[3], result_index[4]
                    w, h, l = result_index[5], result_index[6], result_index[7]
                    x, y, z, roty = result_index[8], result_index[9], result_index[10], result_index[11]
                    score = result_index[12]
                    # print(f'==> cls: {cls_str} | bbox: {[x1, y1, x2, y2]} | dimension: {[w, h, l]} | location: {[x, y, z]} angle: {roty} |score: {score}')

                    cls_str = [cls_objects_num2str[str(cls_object)]]
                    bbox_2d = [x1, y1, x2, y2]
                    bbox_3d = [w, h, l, x, y, z, roty] # l, w, h, x, y, z, rot_y, depth
                    # Cyclist 0 0 0 1198 677 1264 740 0.81 1.7 1.87 2.67 16.74 -1.34 -0.79 43.86
                    score = np.float64("{:.2f}".format(score))
                    if x1 > 0 and y1 > 0 and x1 < x2 and y1 < y2 and x2 < 1640 and y2 < 1232 and score > 0.2:
                        # #****************************************************#
                        # #  🚀 Compare 3d results generation    #
                        # #****************************************************#
                        anno_list_temp = {}
                        anno_list_temp['l_3d'] = l
                        anno_list_temp['w_3d'] = w
                        anno_list_temp['h_3d'] = h
                        anno_list_temp['x_3d'] = x
                        anno_list_temp['y_3d'] = y
                        anno_list_temp['z_3d'] = z
                        anno_list_temp['heading_3d'] = roty
                        # kitti_format: frame, track_id, type, truncated, occluded, alpha, bbox, dimensions, location, rot_y, score
                        x1 = int(np.float64(x1))
                        y1 = int(np.float64(y1))
                        x2 = int(np.float64(x2))
                        y2 = int(np.float64(y2))
                        l = round(np.float64(l), 2)
                        w = round(np.float64(w), 2)
                        h = round(np.float64(h), 2)
                        x = round(np.float64(x), 2)
                        y = round(np.float64(y), 2)
                        z = round(np.float64(z), 2)
                        roty = round(np.float64(roty), 2)

                        label_tmp = f'{image_index} {cls_str[0]} {-1} {-1} {-1} {-1} {x1} {y1} {x2} {y2} {l} {w} {h} {x} {y} {z} {roty} {score}\n'
                        label_detection_tmp = f'{cls_object - 1} {x1} {y1} {x2} {y2} {score} {x} {y} {z} {w} {h} {l} {roty}\n'
                        # print(f'==> label_detection_tmp: {label_detection_tmp}')
                        file_detection.write(label_detection_tmp)
                        # print(f'{label_tmp}')

                        # # Draw center point on pixel image
                        # lidar_points_center = np.array([x, y, z])
                        # draw_lidar_center_points(image=image, lidar_points=lidar_points_center, camera_info=opt.camera_info)

                        # # Draw 3d bounding boxes
                        # draw_3d_points(image=image, anno_list=anno_list_temp, camera_info=opt.camera_info, color_box=box_color)
            if save_image:
                cv2.imwrite(os.path.join(vis_save_path_3d, f'{"%06d" % image_index}.jpg'), image)
    if save_image:
        print(f'Please find images in {vis_save_path_3d}')
    print(f'Please find the detection txts in: {txt_save_path_3d}')

