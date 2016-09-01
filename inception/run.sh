NOWDIR=`dirname "$0"`
NOWDIR=`cd $NOWDIR; pwd`
export PYTHONPATH=/home/zhangxuesen/workshops/FinalProject2/inception/

python $NOWDIR/inception/flowers_train.py \
    --train_dir=$NOWDIR/log \
    --data_dir=/home/zhangxuesen/workshops/data/tfrecord \
    --initial_learning_rate=0.05 \
    --max_steps=110000 \
    --num_epochs_per_decay=30 \
    --batch_size=100 \
    --image_size=32 \
