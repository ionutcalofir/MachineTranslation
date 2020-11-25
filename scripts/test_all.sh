# python main.py \
# 	--phase=test \
# 	--model_path=models/model_baseline_enro.pt

# python main.py \
# 	--phase=inference \
# 	--sentence="Let's go to the park" \
# 	--model_path=models/model_baseline_enro.pt


# python main.py \
# 	--phase=test \
# 	--model_path=models/model_bert_tokenizer_enro.pt \
# 	--text_preprocessor=bert_tokenizer

# python main.py \
# 	--phase=inference \
# 	--sentence="Let's go to the park" \
# 	--model_path=models/model_bert_tokenizer_enro.pt \
# 	--text_preprocessor=bert_tokenizer


# python main.py \
# 	--phase=test \
# 	--model_path=models/model_w2v_enro.pt \
# 	--text_preprocessor=stemming \
# 	--use_w2v=True \
# 	--freeze_w2v=False

# python main.py \
# 	--phase=inference \
# 	--sentence="Let's go to the park" \
# 	--model_path=models/model_w2v_enro.pt \
# 	--text_preprocessor=stemming \
# 	--use_w2v=True \
# 	--freeze_w2v=False


# python main.py \
# 	--phase=test \
# 	--model_path=models/model_w2v_freeze_enro.pt \
# 	--text_preprocessor=stemming \
# 	--use_w2v=True

# python main.py \
# 	--phase=inference \
# 	--sentence="Let's go to the park" \
# 	--model_path=models/model_w2v_freeze_enro.pt \
# 	--text_preprocessor=stemming \
# 	--use_w2v=True


python main.py \
	--phase=test \
	--model_path=models/model_stemming_enro.pt \
	--text_preprocessor=stemming

# python main.py \
# 	--phase=inference \
# 	--sentence="Let's go to the park" \
# 	--model_path=models/model_stemming_enro.pt \
# 	--text_preprocessor=stemming
