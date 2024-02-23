python /home/shpark/hw0_starter_code/main.py --factorization_weight 0.99 --regression_weight 0.01 --logdir run/shared=True_LF=0.99_LR=0.01
# python /home/shpark/hw0_starter_code/main.py --factorization_weight 0.5 --regression_weight 0.5 --logdir run/shared=True_LF=0.5_LR=0.5
# python /home/shpark/hw0_starter_code/main.py --factorization_weight 0.5 --regression_weight 0.5 --logdir run/shared=False_LF=0.5_LR=0.5 --no_shared_embeddings
python /home/shpark/hw0_starter_code/main.py --factorization_weight 0.99 --regression_weight 0.01 --logdir run/shared=False_LF=0.99_LR=0.01 --no_shared_embeddings