cd src
mkdir results
mkdir results/statistics
mkdir results/visualize
mkdir results/predictions
mkdir results/relation-iou
mkdir results/relation

rm results/predictions/*
rm results/relation/*
rm results/relation-iou/*

python3 test_boundary.py --config config/abc.yaml --pretrain ../checkpoints/abc-pretrained.pth
python3 eval_iou.py results/predictions
python3 eval_ap.py results/predictions 1
python3 eval_ap.py results/predictions results/statistics/AP.txt 0
