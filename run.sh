# nohup python main.py --name inD_map07-17_interval10_seq500_nfea10_pad-300 \
# --config_file /home/rzhou/Projects/Diffusion-TS/Config/ika_multi.yaml --gpu 3 --train \
# >> /home/rzhou/Projects/Diffusion-TS/OUTPUT/inD_map07-17_interval10_seq500_nfea10_pad-300.log 2>&1 &



nohup python main.py --name rounD_map01_interval1_seq500_nfea10_pad-300 \
--config_file Config/ika_multi.yaml --gpu 2 --sample 0 --milestone 10 \
>> OUTPUT/rounD_map01_interval1_seq500_nfea10_pad-300/logs/rounD_map01_interval1_seq500_nfea10_pad-300.log 2>&1 &