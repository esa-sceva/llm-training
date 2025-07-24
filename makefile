include .env

# Allow overrides from CLI (e.g. make train DEVICES=1)
export

VENV_DIR=venv
PYTHON=$(VENV_DIR)/bin/python
PIP=$(VENV_DIR)/bin/pip

install:
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install litgpt wandb

dataset:
	curl -L $(DATASET_URL) -o $(JSON_PATH)

model:
	litgpt download $(MODEL_NAME) --access_token="$(HF_TOKEN)"

config:
	curl -L $(CONFIG_URL) -o $(CONFIG_PATH)

wandb:
	wandb login "$(WANDB_TOKEN)"

train:
	litgpt finetune lora \
		--data JSON \
		--data.json_path $(JSON_PATH) \
		--data.val_split_fraction $(VAL_SPLIT) \
		--out_dir $(OUT_DIR) \
		--config $(CONFIG_PATH) \
	
evaluate:
	litgpt evaluate microsoft/phi-2/ \
		--batch_size 4 \
		--tasks "hellaswag,truthfulqa_mc2,mmlu" \
		--out_dir evaluate_model/

all: install config dataset model wandb train