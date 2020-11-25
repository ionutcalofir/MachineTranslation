python main.py \
	--phase=val \
	--model_path=models/model_baseline_enro.pt

python main.py \
	--phase=test \
	--model_path=models/model_baseline_enro.pt

python main.py \
	--phase=inference \
	--sentence="Let's go to the park" \
	--model_path=models/model_baseline_enro.pt
