# nohup python main.py --name rounD_map02-08_interval100_seq500_reduced_nfea30 \
# --config_file /home/rzhou/Projects/Diffusion-TS/Config/ika_multi.yaml --gpu 3 --train \
# >> /home/rzhou/Projects/Diffusion-TS/OUTPUT/rounD_map02-08_interval100_seq500_reduced_nfea30.log 2>&1



nohup python main.py --name rounD_map02-08_interval100_seq500_reduced_nfea30 \
--config_file Config/ika_multi.yaml --gpu 3 --sample 0 --milestone 10 \
>> OUTPUT/rounD_map02-08_interval100_seq500_reduced_nfea30/logs/rounD_map02-08_interval100_seq500_reduced_nfea30.out 2>&1