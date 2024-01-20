ROOT_DIR=data/vqa_v2_data
mdkir -p $RROOT_DIR

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip -P $ROOT_DIR
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip -P $ROOT_DIR
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -P $ROOT_DIR
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P $ROOT_DIR
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P $ROOT_DIR

unzip $ROOT_DIR/annotations_trainval2014.zip -d $ROOT_DIR
unzip $ROOT_DIR/v2_Questions_Train_mscoco.zip -d $ROOT_DIR
unzip $ROOT_DIR/v2_Questions_Val_mscoco.zip -d $ROOT_DIR
unzip $ROOT_DIR/v2_Annotations_Train_mscoco.zip -d $ROOT_DIR
unzip $ROOT_DIR/v2_Annotations_Val_mscoco.zip -d $ROOT_DIR

python src/contvqa/get_image_based_settings.py --coco_path $ROOT_DIR/annotations --vqa_path $ROOT_DIR