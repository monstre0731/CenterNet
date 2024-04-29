conda activate centernew
cd src
# train
python main.py ddd --exp_id back_to_mtl --dataset kitti --kitti_split 3dop --batch_size 32 --master_batch 7 --num_epochs 70 --lr_step 45,60 --gpus 0
# test
python test.py ddd --exp_id 3dop --dataset kitti --kitti_split 3dop --resume
# If use the pretrained model 
python test.py ddd --exp_id 3dop --dataset kitti --kitti_split 3dop --load_model /home/qingwu/Desktop/kitti_centernet/CenterNet/models/ddd_3dop.pth
cd ..
