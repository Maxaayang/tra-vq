# source ~/.zshrc
# conda activate col

# python3 train.py config/default.yaml > ./logs/test.log
rm -rf ./ckpt/vqvae400/*
python3 train.py config/default.yaml > ./logs/vqvae/vqvae400.log

# python3 generate.py config/default.yaml ./ckpt/juke50/params/step_9850-RC_1.760-KL_0.000-model.pt \
# generations/juke50/ 10 5 > ./logs/gen50.log
# python3 train.py config/default.yaml > ./logs/vqvae10.log

# python train.py config/default.yaml > ./logs/vqvae/new_vqvae50.log


# rm -rf ./generations/juke10/*
# python3 generate.py config/default.yaml ./ckpt/juke10/step_950-RC_2.696-KL_0.000-model.pt \
# generations/juke10/ 10 5 > ./logs/gen/gen_vq10.log