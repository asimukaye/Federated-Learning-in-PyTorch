# clients (K) = 10
# rounds (R) = 100
# epochs (E) = 10
# batches (B) = 10
# beta (momentum) = 0

python3 main.py \
            --exp_name "test_cgsv_MNIST_CNN_IID_C1.0_B10" --seed 42 --device cuda \
            --dataset MNIST \
            --split_type iid --test_fraction 0 \
            --model_name TwoCNN --resize 28 --hidden_size 200 \
            --algorithm cgsv --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 acc5 \
            --K 3 --R 100 --E 2 --C 1.0 --B 10 --beta 0 \
            --optimizer SGD --lr 0.215 --lr_decay 0.95 --lr_decay_step 25 --criterion CrossEntropyLoss