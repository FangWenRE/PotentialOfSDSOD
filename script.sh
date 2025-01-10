d1="simple1"
d2="simple2"

#============= 1 ==============
# SPLG on simple image 1
simple1_save_path="/opt/PolienofSOD/dataset/Simples/$d1"
nohup python -u process/generate_simple_mask.py --gpu="cuda:0"\
                                                --input_path="/opt/PolienofSOD/dataset/SimpleImages/$d1"\
                                                --output_path=$simple1_save_path\
                                                > "./logs/labels/$d1.log" 2>&1 &

# SPLG on simple image 2
simple2_save_path="/opt/PolienofSOD/dataset/Simples/$d2"
nohup python -u process/generate_simple_mask.py --gpu="cuda:1"\
                                                --input_path="/opt/PolienofSOD/dataset/SimpleImages/$d2"\
                                                --output_path=$simple2_save_path\
                                                > "./logs/labels/$d2.log" 2>&1 &
wait

#============= 2 ==============
# train the initial model d1
CUDA_VISIBLE_DEVICES=0 nohup python3 -u train.py maxsum --data_path=$simple1_save_path\
                                                        --weight="" --gpus=0 --save_tar=$d1\
                                                         > "./logs/training/$d1.log" 2>&1 &
# train the initial model d2
CUDA_VISIBLE_DEVICES=1 nohup python3 -u train.py maxsum --data_path=$simple2_save_path\
                                                        --weight="" --gpus=1 --save_tar=$d2\
                                                        > "./logs/training/$d2.log" 2>&1 &
wait

#============= 3 ==============
# Use the trained models d1 & d2 to inference on complex images with infer_map.py
python infer_map.py --image_path="/opt/PolienofSOD/dataset/ComplexImages/image"\
                    --save_dir="/opt/PolienofSOD/dataset/ComplexImages/v1infer"\
                    --load_path="/opt/PolienofSOD/weight/$d1/best.pth"

python infer_map.py --image_path="/opt/PolienofSOD/dataset/ComplexImages/image"\
                    --save_dir="/opt/PolienofSOD/dataset/ComplexImages/v2infer"\
                    --load_path="/opt/PolienofSOD/weight/$d2/best.pth"
wait


# Take the union of the inference maps of models d1 and d2 on complex images.
python process/uion_d1d2.py --path="/opt/PolienofSOD/dataset/ComplexImages"
wait

#============= 4 ==============
# CPLG 
r1=refinedr1
refinedr1_save_path="/opt/PolienofSOD/dataset/Complexities/$r1"
nohup python -u process/generate_complex_mask.py --gpu="cuda:0"\
                                                 --input_path="/opt/PolienofSOD/dataset/ComplexImages/image"\
                                                 --output_path=$refinedr1_save_path\
                                                 --cinfer_target="v12infer"\
                                                  > "./logs/labels/$r1.log" 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 nohup python3 -u train.py maxsum --data_path=$refinedr1_save_path\
                                                        --weight="" --gpus=0 --save_tar=$r1\
                                                         > "./logs/training/$r1.log" 2>&1 &
wait

# Use the trained model r1 to perform inference on complex images, 
# and save the results in the `r1infer` directory at the same level as the complex image.
python infer_map.py --image_path="/opt/PolienofSOD/dataset/ComplexImages/image"\
                    --save_dir="/opt/PolienofSOD/dataset/ComplexImages/r1infer"\
                    --load_path="/opt/PolienofSOD/weight/$r1/best.pth"
wait

#============= 5 ==============
r2=refinedr2
refinedr2_save_path="/opt/PolienofSOD/dataset/Complexities/$r2"
nohup python -u process/generate_complex_mask.py --gpu="cuda:0"\
                                                 --input_path="/opt/PolienofSOD/dataset/ComplexImages/image"\
                                                 --output_path=$refinedr2_save_path\
                                                 --cinfer_target="r1infer"\
                                                  > "./logs/labels/$r2.log" 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 nohup python3 -u train.py maxsum --data_path=$refinedr2_save_path\
                                                        --weight="" --gpus=0 --save_tar=$r2\
                                                         > "./logs/training/$r2.log" 2>&1 &
wait

#============= 6 ==============
# Performance evaluation & save Saliency Maps.
CUDA_VISIBLE_DEVICES=0 python3 -u test.py maxsum --gpus=0 --weight="/opt/PolienofSOD/weight/$r2/best.pth"  --save_tar=$r2 --save 
