import os

folder_path = '/media/disk3/Data-qingwu/dataset/kitti_tracking/training/image_02'

for folder_name in os.listdir(folder_path):
    folder_full_path = os.path.join(folder_path, folder_name)
    if os.path.isdir(folder_full_path):
        num_files = len(os.listdir(folder_full_path))
        print(f"文件夹 {folder_name} 中的文件数量为: {num_files}")
