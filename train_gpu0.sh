# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/pvtc.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PVTC_2/" \
#       SOLVER.TOTAL_EPOCH '1' \

# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/pvtc.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PVTC_2/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/PVTC_2/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_pvtc_model.pth" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/pvtc.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PVTC_3/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/pvtc.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PVTC_3/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/PVTC_3/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_pvtc_model.pth" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/pvtc.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PVTC_4/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/pvtc.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PVTC_4/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/PVTC_4/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_pvtc_model.pth" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/pvtc.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PVTC_5/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/pvtc.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PVTC_5/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/PVTC_5/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_pvtc_model.pth" \
#
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/pvtc.yaml \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PVTC_6/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/pvtc.yaml \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PVTC_6/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/PVTC_6/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_pvtc_model.pth" \
#
#
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/pvtc.yaml \
#       DATA.PERCENTAGE '0.7' \
#       DATA.NUMBER_CLASSES "7" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PVTC_7/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/pvtc.yaml \
#       DATA.PERCENTAGE '0.7' \
#       DATA.NUMBER_CLASSES "7" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PVTC_7/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/PVTC_7/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_pvtc_model.pth" \


python tune_fgvc.py \
      --train-type "prompt" \
      --config-file configs/prompt/pvtc2.yaml \
      DATA.PERCENTAGE '0.2' \
      DATA.NUMBER_CLASSES "2" \
      MODEL.TYPE "vit" \
      DATA.BATCH_SIZE "20" \
      MODEL.PROMPT.DEEP "True" \
      MODEL.PROMPT.DROPOUT "0.1" \
      MODEL.PROMPT.NUM_TOKENS "10" \
      DATA.FEATURE "sup_vitb16_imagenet21k" \
      MODEL.MODEL_ROOT "models/" \
      OUTPUT_DIR "./Prompt_OOD/PVTC_S2_2/" \
      SOLVER.TOTAL_EPOCH '1' \

python tune_fgvc.py \
      --train-type "prompt" \
      --config-file configs/prompt/pvtc2.yaml \
      DATA.PERCENTAGE '0.2' \
      DATA.NUMBER_CLASSES "2" \
      MODEL.TYPE "vit" \
      DATA.BATCH_SIZE "20" \
      MODEL.PROMPT.DEEP "True" \
      MODEL.PROMPT.DROPOUT "0.1" \
      MODEL.PROMPT.NUM_TOKENS "10" \
      DATA.FEATURE "sup_vitb16_imagenet21k" \
      MODEL.MODEL_ROOT "models/" \
      OUTPUT_DIR "./Prompt_OOD/PVTC_S2_2/" \
      SOLVER.TOTAL_EPOCH '0' \
      MODEL.WEIGHT_PATH "./Prompt_OOD/PVTC_S2_2/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_pvtc_model.pth" \


# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/pvtc2.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PVTC_S2_3/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/pvtc2.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PVTC_S2_3/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/PVTC_S2_3/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_pvtc_model.pth" \
#
#
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/pvtc2.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PVTC_S2_4/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/pvtc2.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PVTC_S2_4/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/PVTC_S2_4/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_pvtc_model.pth" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/pvtc2.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PVTC_S2_5/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/pvtc2.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PVTC_S2_5/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/PVTC_S2_5/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_pvtc_model.pth" \
#
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/pvtc.yaml \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PVTC_S2_6/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/pvtc.yaml \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PVTC_S2_6/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/PVTC_S2_6/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_pvtc_model.pth" \



# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/cotton.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/COTTON_2/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/cotton.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/COTTON_2/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/COTTON_2/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_cotton_model.pth" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/cotton.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/COTTON_3/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/cotton.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/COTTON_3/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/COTTON_3/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_cotton_model.pth" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/cotton.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/COTTON_4/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/cotton.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/COTTON_4/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/COTTON_4/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_cotton_model.pth" \
#
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/cotton2.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/COTTON_S2_2/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/cotton2.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/COTTON_S2_2/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/COTTON_S2_2/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_cotton_model.pth" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/cotton2.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/COTTON_S2_3/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/cotton2.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/COTTON_S2_3/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/COTTON_S2_3/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_cotton_model.pth" \
#

# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/strawberry.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/STRAWBERRY_2/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/strawberry.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/STRAWBERRY_2/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/STRAWBERRY_2/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_strawberry_model.pth" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/strawberry.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/STRAWBERRY_3/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/strawberry.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/STRAWBERRY_3/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/STRAWBERRY_3/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_strawberry_model.pth" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/strawberry.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/STRAWBERRY_4/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/strawberry.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/STRAWBERRY_4/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/STRAWBERRY_4/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_strawberry_model.pth" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/strawberry.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/STRAWBERRY_5/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/strawberry.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/STRAWBERRY_5/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/STRAWBERRY_5/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_strawberry_model.pth" \
#
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/strawberry.yaml \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/STRAWBERRY_6/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/strawberry.yaml \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/STRAWBERRY_6/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/STRAWBERRY_6/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_strawberry_model.pth" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/strawberry2.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/STRAWBERRY_S2_2/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/strawberry2.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/STRAWBERRY_S2_2/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/STRAWBERRY_S2_2/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_strawberry_model.pth" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/strawberry2.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/STRAWBERRY_S2_3/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/strawberry2.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/STRAWBERRY_S2_3/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/STRAWBERRY_S2_3/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_strawberry_model.pth" \
#
#
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/strawberry2.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/STRAWBERRY_S2_4/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/strawberry2.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/STRAWBERRY_S2_4/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/STRAWBERRY_S2_4/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_strawberry_model.pth" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/strawberry2.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/STRAWBERRY_S2_5/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/strawberry2.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/STRAWBERRY_S2_5/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/STRAWBERRY_S2_5/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_strawberry_model.pth" \
#
#


# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/mango.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/MANGO_2/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/mango.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/MANGO_2/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/MANGO_2/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_mango_model.pth" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/mango.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/MANGO_3/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/mango.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/MANGO_3/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/MANGO_3/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_mango_model.pth" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/mango.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/MANGO_4/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/mango.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/MANGO_4/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/MANGO_4/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_mango_model.pth" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/mango.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/MANGO_5/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/mango.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/MANGO_5/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/MANGO_5/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_mango_model.pth" \
#
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/mango.yaml \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/MANGO_6/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/mango.yaml \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/MANGO_6/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/MANGO_6/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_mango_model.pth" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/mango2.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/MANGO_S2_2/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/mango2.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/MANGO_S2_2/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/MANGO_S2_2/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_mango_model.pth" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/mango2.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/MANGO_S2_3/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/mango2.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/MANGO_S2_3/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/MANGO_S2_3/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_mango_model.pth" \
#
#
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/mango2.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/MANGO_S2_4/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/mango2.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/MANGO_S2_4/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/MANGO_S2_4/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_mango_model.pth" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/mango2.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/MANGO_S2_5/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/mango2.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/MANGO_S2_5/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/MANGO_S2_5/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_mango_model.pth" \
#
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/mango2.yaml \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       MODEL.TYPE "vit" \
#       DATA.BATCH_SIZE "20" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/MANGO_S2_6/" \
#       SOLVER.TOTAL_EPOCH '1' \
#



# ##### PLANTVILLAGE DATA_SCALE#####
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/plant_village.yaml \
#       DATA.PERCENTAGE '0.1' \
#       MODEL.TYPE "vit" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       DATA.BATCH_SIZE "24" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PLANTVILLAGE_2/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# ##### PLANTVILLAGE DATA_SCALE#####
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/plant_village.yaml \
#       DATA.PERCENTAGE '0.1' \
#       MODEL.TYPE "vit" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       DATA.BATCH_SIZE "24" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PLANTVILLAGE_2/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/PLANTVILLAGE_2/sup_vitb16_imagenet21k/lr3.90625_wd0.001/run1/val_plant_village_model.pth" \


# ##### PLANTVILLAGE DATA_SCALE#####
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/plant_village.yaml \
#       DATA.PERCENTAGE '0.1' \
#       MODEL.TYPE "vit" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       DATA.BATCH_SIZE "24" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PLANTVILLAGE_3/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# ##### PLANTVILLAGE DATA_SCALE#####
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/plant_village.yaml \
#       DATA.PERCENTAGE '0.1' \
#       MODEL.TYPE "vit" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       DATA.BATCH_SIZE "24" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PLANTVILLAGE_3/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/PLANTVILLAGE_3/sup_vitb16_imagenet21k/lr1.953125_wd0.001/run1/val_plant_village_model.pth" \
#
#
# ##### PLANTVILLAGE DATA_SCALE#####
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/plant_village.yaml \
#       DATA.PERCENTAGE '0.1' \
#       MODEL.TYPE "vit" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       DATA.BATCH_SIZE "24" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PLANTVILLAGE_4/" \
#       SOLVER.TOTAL_EPOCH '1' \
#
# ##### PLANTVILLAGE DATA_SCALE#####
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/plant_village.yaml \
#       DATA.PERCENTAGE '0.1' \
#       MODEL.TYPE "vit" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       DATA.BATCH_SIZE "24" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PLANTVILLAGE_4/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Prompt_OOD/PLANTVILLAGE_4/sup_vitb16_imagenet21k/lr1.5625_wd0.001/run1/val_plant_village_model.pth" \


# sleep 4h
# python train.py \
#       --config-file configs/finetune/plant_village.yaml \
#       DATA.PERCENTAGE '0.1' \
#       DATA.BATCH_SIZE "24" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PLANTVILLAGE_1/" \
#
# python train.py \
#       --config-file configs/finetune/plant_village.yaml \
#       DATA.BATCH_SIZE "24" \
#       DATA.PERCENTAGE '0.1' \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       MODEL.TYPE "ssl-vit" \
#       DATA.FEATURE "mae_vitb16" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PLANTVILLAGE_1/" \
#
# python train.py \
#       --config-file configs/finetune/plant_village.yaml \
#       DATA.BATCH_SIZE "24" \
#       DATA.PERCENTAGE '0.1' \
#       SOLVER.BASE_LR "0.001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       MODEL.TYPE "ssl-vit" \
#       DATA.FEATURE "mae_vitb16" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PLANTVILLAGE_1/" \
#
#
# python train.py \
#       --config-file configs/finetune/plant_village.yaml \
#       DATA.BATCH_SIZE "24" \
#       DATA.PERCENTAGE '0.2' \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PLANTVILLAGE_2/" \
#
# python train.py \
#       --config-file configs/finetune/plant_village.yaml \
#       DATA.BATCH_SIZE "24" \
#       DATA.PERCENTAGE '0.2' \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       MODEL.TYPE "ssl-vit" \
#       DATA.FEATURE "mae_vitb16" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PLANTVILLAGE_2/" \
#
# python train.py \
#       --config-file configs/finetune/plant_village.yaml \
#       DATA.BATCH_SIZE "24" \
#       DATA.PERCENTAGE '0.2' \
#       SOLVER.BASE_LR "0.001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       MODEL.TYPE "ssl-vit" \
#       DATA.FEATURE "mae_vitb16" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PLANTVILLAGE_2/" \
#
#
# python train.py \
#       --config-file configs/finetune/plant_village.yaml \
#       DATA.BATCH_SIZE "24" \
#       DATA.PERCENTAGE '0.3' \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PLANTVILLAGE_3/" \
#
# python train.py \
#       --config-file configs/finetune/plant_village.yaml \
#       DATA.BATCH_SIZE "24" \
#       DATA.PERCENTAGE '0.3' \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       MODEL.TYPE "ssl-vit" \
#       DATA.FEATURE "mae_vitb16" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PLANTVILLAGE_3/" \
#
# python train.py \
#       --config-file configs/finetune/plant_village.yaml \
#       DATA.BATCH_SIZE "24" \
#       DATA.PERCENTAGE '0.3' \
#       SOLVER.BASE_LR "0.001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       MODEL.TYPE "ssl-vit" \
#       DATA.FEATURE "mae_vitb16" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PLANTVILLAGE_3/" \
#
#
# python train.py \
#       --config-file configs/finetune/plant_village.yaml \
#       DATA.BATCH_SIZE "24" \
#       DATA.PERCENTAGE '0.4' \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PLANTVILLAGE_4/" \
#
# python train.py \
#       --config-file configs/finetune/plant_village.yaml \
#       DATA.BATCH_SIZE "24" \
#       DATA.PERCENTAGE '0.4' \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       MODEL.TYPE "ssl-vit" \
#       DATA.FEATURE "mae_vitb16" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PLANTVILLAGE_4/" \
#
# python train.py \
#       --config-file configs/finetune/plant_village.yaml \
#       DATA.BATCH_SIZE "24" \
#       DATA.PERCENTAGE '0.3' \
#       SOLVER.BASE_LR "0.001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       MODEL.TYPE "ssl-vit" \
#       DATA.FEATURE "mae_vitb16" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PLANTVILLAGE_4/" \


# python train.py \
#       --config-file configs/finetune/plant_village.yaml \
#       DATA.BATCH_SIZE "64" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PLANTVILLAGE/" \
#
# python train.py \
#       --config-file configs/finetune/plant_village.yaml \
#       DATA.BATCH_SIZE "64" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       MODEL.TYPE "ssl-vit" \
#       DATA.FEATURE "mae_vitb16" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PLANTVILLAGE/" \
#
# python train.py \
#       --config-file configs/finetune/plant_village.yaml \
#       DATA.BATCH_SIZE "64" \
#       SOLVER.BASE_LR "0.001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       MODEL.TYPE "ssl-vit" \
#       DATA.FEATURE "mae_vitb16" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PLANTVILLAGE/" \




# python train.py \
#       --config-file configs/linear/pvts.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvts_2/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvts.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvts_3/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvts.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvts_4/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvts.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvts_5/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvts.yaml \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvts_6/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
#
# python train.py \
#       --config-file configs/linear/pvts.yaml \
#       DATA.PERCENTAGE '0.7' \
#       DATA.NUMBER_CLASSES "7" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvts_7/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
#
# python train.py \
#       --config-file configs/linear/pvts2.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvts_S2_2/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvts2.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvts_S2_3/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvts2.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvts_S2_4/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvts2.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvts_S2_5/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvts2.yaml \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvts_S2_6/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
#
#
#
#
#
#
#
#
# python train.py \
#       --config-file configs/linear/pvtg.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtg_2/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvtg.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtg_3/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvtg.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtg_4/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvtg.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtg_5/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvtg.yaml \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtg_6/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
#
# python train.py \
#       --config-file configs/linear/pvtg.yaml \
#       DATA.PERCENTAGE '0.7' \
#       DATA.NUMBER_CLASSES "7" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtg_7/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
#
# python train.py \
#       --config-file configs/linear/pvtg2.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtg_S2_2/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvtg2.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtg_S2_3/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvtg2.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtg_S2_4/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvtg2.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtg_S2_5/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvtg2.yaml \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtg_S2_6/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
#
#
#
#
#
#
#
#
#
#
# python train.py \
#       --config-file configs/linear/pvtc.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtc_2/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvtc.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtc_3/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvtc.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtc_4/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvtc.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtc_5/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvtc.yaml \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtc_6/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
#
# python train.py \
#       --config-file configs/linear/pvtc.yaml \
#       DATA.PERCENTAGE '0.7' \
#       DATA.NUMBER_CLASSES "7" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtc_7/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
#
# python train.py \
#       --config-file configs/linear/pvtc2.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtc_S2_2/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvtc2.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtc_S2_3/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvtc2.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtc_S2_4/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvtc2.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtc_S2_5/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/pvtc2.yaml \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtc_S2_6/" \
#       SOLVER.TOTAL_EPOCH 1 \


# python train.py \
#       --config-file configs/linear/strawberry.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/strawberry_2/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/strawberry.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/strawberry_3/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/strawberry.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/strawberry_4/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/strawberry.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/strawberry_5/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/strawberry.yaml \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/strawberry_6/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/strawberry2.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/strawberry_S2_2/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/strawberry2.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/strawberry_S2_3/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/strawberry2.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/strawberry_S2_4/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/strawberry2.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/strawberry_S2_5/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
#
#
#
# python train.py \
#       --config-file configs/linear/mango.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/mango_2/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/mango.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/mango_3/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/mango.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/mango_4/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/mango.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/mango_5/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/mango.yaml \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/mango_6/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/mango2.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/mango_S2_2/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/mango2.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/mango_S2_3/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/mango2.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/mango_S2_4/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/mango2.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/mango_S2_5/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/cotton.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/COTTON_2/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/cotton.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/COTTON_3/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/cotton.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/COTTON_4/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/cotton2.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/COTTON_S2_2/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
# python train.py \
#       --config-file configs/linear/cotton2.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "5.0" \
#       SOLVER.WEIGHT_DECAY "0.0001" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/COTTON_S2_3/" \
#       SOLVER.TOTAL_EPOCH 1 \
#
#
#
# ##### PLANTVILLAGE DATA_SCALE#####
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/plant_village.yaml \
#       DATA.PERCENTAGE '0.1' \
#       MODEL.TYPE "vit" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       DATA.BATCH_SIZE "24" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "20" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PLANTVILLAGE_1/" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/plant_village.yaml \
#       DATA.PERCENTAGE '0.2' \
#       MODEL.TYPE "vit" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       DATA.BATCH_SIZE "24" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PLANTVILLAGE_2/" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/plant_village.yaml \
#       DATA.PERCENTAGE '0.3' \
#       MODEL.TYPE "vit" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       DATA.BATCH_SIZE "24" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PLANTVILLAGE_3/" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/plant_village.yaml \
#       DATA.PERCENTAGE '0.4' \
#       MODEL.TYPE "vit" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       DATA.BATCH_SIZE "24" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PLANTVILLAGE_4/" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/plant_village.yaml \
#       MODEL.TYPE "vit" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       DATA.BATCH_SIZE "24" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PLANTVILLAGE/" \
#
#
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/plant_village.yaml \
#       DATA.PERCENTAGE '0.1' \
#       MODEL.TYPE "ssl-vit" \
#       DATA.FEATURE "mae_vitb16" \
#       DATA.BATCH_SIZE "24" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "20" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PLANTVILLAGE_1/" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/plant_village.yaml \
#       DATA.PERCENTAGE '0.2' \
#       MODEL.TYPE "ssl-vit" \
#       DATA.FEATURE "mae_vitb16" \
#       DATA.BATCH_SIZE "24" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PLANTVILLAGE_2/" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/plant_village.yaml \
#       DATA.PERCENTAGE '0.3' \
#       MODEL.TYPE "ssl-vit" \
#       DATA.FEATURE "mae_vitb16" \
#       DATA.BATCH_SIZE "24" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PLANTVILLAGE_3/" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/plant_village.yaml \
#       DATA.PERCENTAGE '0.4' \
#       MODEL.TYPE "ssl-vit" \
#       DATA.FEATURE "mae_vitb16" \
#       DATA.BATCH_SIZE "24" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PLANTVILLAGE_4/" \
#
#
# python tune_fgvc.py \
#       --train-type "prompt" \
#       --config-file configs/prompt/plant_village.yaml \
#       MODEL.TYPE "ssl-vit" \
#       DATA.FEATURE "mae_vitb16" \
#       DATA.BATCH_SIZE "24" \
#       MODEL.PROMPT.DEEP "True" \
#       MODEL.PROMPT.DROPOUT "0.1" \
#       MODEL.PROMPT.NUM_TOKENS "10" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Prompt_OOD/PLANTVILLAGE/" \

# python train.py \
#       --config-file configs/finetune/cotton.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/COTTON_2/" \
# #
# #
# python train.py \
#       --config-file configs/finetune/cotton.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/COTTON_3/" \
# #
# #
# #
# python train.py \
#       --config-file configs/finetune/cotton.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/COTTON_4/" \
# #
# #
# #
# python train.py \
#       --config-file configs/finetune/cotton2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/COTTON_S2_2/" \
# #
# #
# python train.py \
#       --config-file configs/finetune/cotton2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/COTTON_S2_3/" \
# #
# #
# #
# #
# ###########MANGO ########
# python train.py \
#       --config-file configs/finetune/mango.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/MANGO_2/" \
# #
# #
# python train.py \
#       --config-file configs/finetune/mango.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/MANGO_3/" \
# #
# #
#
# python train.py \
#       --config-file configs/finetune/mango.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/MANGO_4/" \
# #
# #
# python train.py \
#       --config-file configs/finetune/mango.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/MANGO_5/" \
# #
# python train.py \
#       --config-file configs/finetune/mango.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/MANGO_6/" \
# #
# #
# #
# python train.py \
#       --config-file configs/finetune/mango2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/MANGO_S2_2/" \
# #
# #
# python train.py \
#       --config-file configs/finetune/mango2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/MANGO_S2_3/" \
#
#
#
# python train.py \
#       --config-file configs/finetune/mango2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/MANGO_S2_4/" \
# #
# #
# python train.py \
#       --config-file configs/finetune/mango2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/MANGO_S2_5/" \
# #
# ###STRAWBERRY ####
# #
# python train.py \
#       --config-file configs/finetune/strawberry.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/STRAWBERRY_2/" \
# #
# #
# python train.py \
#       --config-file configs/finetune/strawberry.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/STRAWBERRY_3/" \
# #
# #
# #
# python train.py \
#       --config-file configs/finetune/strawberry.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/STRAWBERRY_4/" \
# #
# #
# python train.py \
#       --config-file configs/finetune/strawberry.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/STRAWBERRY_5/" \
# #
# python train.py \
#       --config-file configs/finetune/strawberry.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/STRAWBERRY_6/" \
# #
# #
# #
# python train.py \
#       --config-file configs/finetune/strawberry2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/STRAWBERRY_S2_2/" \
# #
# #
# python train.py \
#       --config-file configs/finetune/strawberry2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/STRAWBERRY_S2_3/" \
# #
# #
# #
# python train.py \
#       --config-file configs/finetune/strawberry2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/STRAWBERRY_S2_4/" \
# #
# #
# python train.py \
#       --config-file configs/finetune/strawberry2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/STRAWBERRY_S2_5/" \
# #
# ###PVTC ####
# python train.py \
#       --config-file configs/finetune/pvtc.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTC_2/" \
#
#
# python train.py \
#       --config-file configs/finetune/pvtc.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTC_3/" \
#
#
# python train.py \
#       --config-file configs/finetune/pvtc.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTC_4/" \
#
#
# python train.py \
#       --config-file configs/finetune/pvtc.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTC_5/" \
#
# python train.py \
#       --config-file configs/finetune/pvtc.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTC_6/" \
#
#
# python train.py \
#       --config-file configs/finetune/pvtc.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.7' \
#       DATA.NUMBER_CLASSES "7" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTC_7/" \
#
#
# python train.py \
#       --config-file configs/finetune/pvtc2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTC_S2_2/" \
#
#
# python train.py \
#       --config-file configs/finetune/pvtc2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTC_S2_3/" \
#
#
#
# python train.py \
#       --config-file configs/finetune/pvtc2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTC_S2_4/" \
#
#
# python train.py \
#       --config-file configs/finetune/pvtc2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTC_S2_5/" \
#
# python train.py \
#       --config-file configs/finetune/pvtc2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTC_S2_6/" \
#
# #
# ###PVTG ####
# #
# python train.py \
#       --config-file configs/finetune/pvtg.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTG_2/" \
# #
# #
# python train.py \
#       --config-file configs/finetune/pvtg.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTG_3/" \
# #
# #
# python train.py \
#       --config-file configs/finetune/pvtg.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTG_4/" \
# #
# #
# python train.py \
#       --config-file configs/finetune/pvtg.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTG_5/" \
# #
# python train.py \
#       --config-file configs/finetune/pvtg.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTG_6/" \
# #
# #
# python train.py \
#       --config-file configs/finetune/pvtg.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.7' \
#       DATA.NUMBER_CLASSES "7" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTG_7/" \
# #
# #
# python train.py \
#       --config-file configs/finetune/pvtg2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTG_S2_2/" \
# #
# #
# python train.py \
#       --config-file configs/finetune/pvtg2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTG_S2_3/" \
# #
# #
# #
# python train.py \
#       --config-file configs/finetune/pvtg2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTG_S2_4/" \
# #
# #
# python train.py \
#       --config-file configs/finetune/pvtg2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTG_S2_5/" \
# #
# python train.py \
#       --config-file configs/finetune/pvtg2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTG_S2_6/" \
# #
# #
# ###PVTS ####
# python train.py \
#       --config-file configs/finetune/pvts.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTS_2/" \
# #
# python train.py \
#       --config-file configs/finetune/pvts.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTS_3/" \
# #
# python train.py \
#       --config-file configs/finetune/pvts.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTS_4/" \
# #
# python train.py \
#       --config-file configs/finetune/pvts.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTS_5/" \
# #
# python train.py \
#       --config-file configs/finetune/pvts.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTS_6/" \
# #
# python train.py \
#       --config-file configs/finetune/pvts.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.7' \
#       DATA.NUMBER_CLASSES "7" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTS_7/" \
# #
# python train.py \
#       --config-file configs/finetune/pvts2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTS_S2_2/" \
# #
# python train.py \
#       --config-file configs/finetune/pvts2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTS_S2_3/" \
# #
# python train.py \
#       --config-file configs/finetune/pvts2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTS_S2_4/" \
# #
# python train.py \
#       --config-file configs/finetune/pvts2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTS_S2_5/" \
# #
# python train.py \
#       --config-file configs/finetune/pvts2.yaml \
#       DATA.BATCH_SIZE "64" \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PVTS_S2_6/" \
#
#
# # ######## PLANTViLLAGE#############
# python train.py \
#       --config-file configs/finetune/plant_village.yaml \
#       DATA.BATCH_SIZE "64" \
#       SOLVER.BASE_LR "0.0001" \
#       SOLVER.WEIGHT_DECAY "0.01" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./finetune_OOD/PLANTViLLAGE/" \
#


