# python main.py \
# 	--phase=train \
# 	--model_path=models/model_baseline_smaller_vocab_enro.pt

# python main.py \
# 	--phase=train \
# 	--model_path=models/model_w2v_smaller_vocab_enro.pt \
# 	--text_preprocessor=stemming \
# 	--use_w2v=True \
# 	--freeze_w2v=False

# python main.py \
# 	--phase=train \
# 	--model_path=models/model_w2v_freeze_smaller_vocab_enro.pt \
# 	--text_preprocessor=stemming \
# 	--use_w2v=True



python main.py \
	--phase=train \
	--model_path=models/model_w2v_smaller_vocab_wo_stemming_enro.pt \
	--use_w2v=True \
	--freeze_w2v=False
