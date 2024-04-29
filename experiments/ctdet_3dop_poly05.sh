conda activate centernew
cd /home/qingwu/CodeSecondProjects/KITTI/KITTI_centernet2d
cd src
# train
python main.py ctdet --exp_id check_error_80_3 --dataset kitti --kitti_split 3dop --batch_size 32 --master_batch 7 --num_epochs 30 --lr_step 4 --gpus 0
python main.py ctdet --exp_id check_error_80 --dataset kitti --kitti_split 3dop --batch_size 32 --master_batch 7 --num_epochs 10 --lr_step 6 --gpus 0
# test
python test.py ddd --exp_id 3dop --dataset kitti --kitti _split 3dop --resume
# If use the pretrained model 
python test.py ddd --exp_id 3dop --dataset kitti --kitti_split 3dop --load_model /home/qingwu/Desktop/kitti_centernet/CenterNet/models/ddd_3dop.pth
cd ..
