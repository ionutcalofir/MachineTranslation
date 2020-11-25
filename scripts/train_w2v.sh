# python main.py \
# 	--phase=train \
# 	--model_path=models/model_w2v_enro.pt \
# 	--text_preprocessor=stemming \
# 	--use_w2v=True \
# 	--freeze_w2v=False

python main.py \
	--phase=train \
	--model_path=models/model_w2v_wo_stemming_enro.pt \
	--use_w2v=True \
	--freeze_w2v=False
