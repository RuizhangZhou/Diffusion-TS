# nohup python main.py --name rounD_single_09-23_seq250 \
# --config_file /home/rzhou/Projects/Diffusion-TS/Config/ika_multi.yaml --gpu 2 --train \
# >> /home/rzhou/Projects/Diffusion-TS/OUTPUT/rounD_single_09-23_seq250.log 2>&1 &



nohup python main.py --name rounD_single_09-23_seq250 \
--config_file Config/ika_multi.yaml --gpu 2 --sample 0 --milestone 10 \
>> OUTPUT/rounD_single_09-23_seq250/logs/rounD_single_09-23_seq250.log 2>&1 &