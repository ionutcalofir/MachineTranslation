# python main.py \
# 	--phase=train \
# 	--model_path=models/model_w2v_freeze_enro.pt \
# 	--text_preprocessor=stemming \
# 	--use_w2v=True

python main.py \
	--phase=train \
	--model_path=models/model_w2v_freeze_wo_stemming_enro.pt \
	--use_w2v=True
