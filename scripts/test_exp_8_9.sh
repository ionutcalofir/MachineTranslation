python main.py \
	--phase=test \
	--model_path=models/model_w2v_smaller_vocab_enro.pt \
	--text_preprocessor=stemming \
	--use_w2v=True \
	--freeze_w2v=False

python main.py \
	--phase=test \
	--model_path=models/model_w2v_smaller_vocab_wo_stemming_enro.pt \
	--use_w2v=True \
	--freeze_w2v=False

python main.py \
	--phase=val \
	--model_path=models/model_w2v_smaller_vocab_enro.pt \
	--text_preprocessor=stemming \
	--use_w2v=True \
	--freeze_w2v=False

python main.py \
	--phase=val \
	--model_path=models/model_w2v_smaller_vocab_wo_stemming_enro.pt \
	--use_w2v=True \
	--freeze_w2v=False
