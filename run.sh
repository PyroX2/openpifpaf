python3 -m openpifpaf.train --dataset custom \
--apollo-square-edge=769 \
--basenet=shufflenetv2k16 --lr=0.00002 --momentum=0.95  --b-scale=5.0 \
--epochs=300 --lr-decay 160 260 --lr-decay-epochs=10  --weight-decay=1e-5 \
--weight-decay=1e-5  --val-interval 10 --loader-workers 16 --custom-upsample 2 \
--custom-bmin 2 --batch-size 8
