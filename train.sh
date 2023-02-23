# python3 train.py config/default.yaml > ./logs/test.log
# python3 train.py config/default.yaml > ./logs/vqvae10.log

rm -rf /home/u21s052015/code/tra-vq/ckpt/or_vae10-A/*
python train.py config/default.yaml > ./logs/or_vae/or_vae10-A.log

python3 generate.py config/default.yaml ./ckpt/or_vae10/step_950-RC_2.820-KL_0.010-model.pt \
generations/or_vae10/ 10 5 > ./logs/or_gen10.log