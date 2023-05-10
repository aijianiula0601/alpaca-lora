import os
import sys
import json

pdj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"pdj:{pdj}")
sys.path.append(pdj)

alpaca_data_cleaned_archive_f = f"{pdj}/alpaca_data_cleaned_archive.json"
alpaca_data_gpt4_f = f"{pdj}/alpaca_data_cleaned_archive.json"

cleaned_data = json.load(open(alpaca_data_cleaned_archive_f))
gpt4_data = json.load(open(alpaca_data_gpt4_f))

print(f"cleaned:{len(cleaned_data)},gpt4:{len(gpt4_data)}")
all_data = cleaned_data + gpt4_data

save_f = "/mnt/cephfs/hjh/train_record/nlp/lora_stanford_alpaca/alpaca_cleand_and_gpt4.json"
save_debug_f = "/mnt/cephfs/hjh/train_record/nlp/lora_stanford_alpaca/debug.json"

json.dump(all_data, fp=open(save_f, 'w'))
json.dump(all_data[:50], fp=open(save_debug_f, 'w'))

print(f"save to:{save_f}")

print(json.load(open(save_f))[:10])
