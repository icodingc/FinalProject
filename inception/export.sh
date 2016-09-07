NOWDIR=`dirname "$0"`
NOWDIR=`cd $NOWDIR; pwd`

python $NOWDIR/inception/inception_export.py \
	--checkpoint_dir=$NOWDIR/log_0905 \
	--export_dir=$NOWDIR/trained_model \
	--image_size=32 \
