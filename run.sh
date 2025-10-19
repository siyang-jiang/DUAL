### Model list : "ProtoNet","MatchingNet"
### Dataset list :"mini_imagenet", "cifar100", "femnist", "tiered_imagenet"

ulimit -n 1000000
dataset_no=0
gpu=0
batch="MAIN"

# ERM pretraining 
python3 scripts/run_exp.py --gpu $gpu --shot 0 --target 0 --dataset $dataset_no --batch $batch --model 0 --encoding_r 0

# Few-shot testing
for shot in 1 5
do
    for target in 8 16
    do
        for model in 0 1
            do
            python3 scripts/run_exp.py --gpu $gpu --model $model --shot $shot --target $target --dataset $dataset_no --batch $batch --testing_lrs
            done
        model=0
    done
done
