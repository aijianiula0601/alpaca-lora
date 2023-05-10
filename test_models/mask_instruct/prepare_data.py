import os
import sys
import json

pdj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(pdj)

from test_models.mask_instruct.data_utils import get_dataset_prompt

DEFAULT_SEGMENT_TOKEN = "### "


def _preprocess_example(conversation_dic):
    """
    :param conversation_dic: example:{
        "background":"---",
        "human_name":"a",
        "bot_name":"b",
        "qas":{
            "turn_0":{"question":"---","answer":"---"},
            ...
            "turn_n":{"question":"---","answer":"---"}
        }
    }
    :return: dic
    """
    dataset_name = conversation_dic['dataset_name']

    turn_n = len(conversation_dic["qas"])
    human_name = conversation_dic['human_name']
    bot_name = conversation_dic['bot_name']
    background_str = get_dataset_prompt(dataset_name, human_name, bot_name,
                                        background=conversation_dic['background']).strip("\n")

    qas_string = ""

    for i in range(turn_n):
        cur_turn_qa = conversation_dic['qas'][f'turn_{i}']
        cur_question_string = DEFAULT_SEGMENT_TOKEN + human_name + ": " + cur_turn_qa["question"].strip("\n")
        cur_answer_string = DEFAULT_SEGMENT_TOKEN + bot_name + ": " + cur_turn_qa["answer"].strip("\n")

        qas_string += cur_question_string + cur_answer_string

    return {"background": background_str, "qas": qas_string}


if __name__ == '__main__':
    org_f = "/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/multitrun_conversation/multi_dataset_qas.json"
    org_data = json.load(open(org_f))

    preprocessed_data = list(map(_preprocess_example, org_data))
    save_f = "/mnt/cephfs/hjh/train_record/nlp/lora_stanford_alpaca/conversation_mask_instruct/train.json"
    save_debug_f = "/mnt/cephfs/hjh/train_record/nlp/lora_stanford_alpaca/conversation_mask_instruct/debug.json"
    json.dump(preprocessed_data, open(save_f, 'w'))
    json.dump(preprocessed_data[:50], open(save_debug_f, 'w'))
    print(f"save to:{save_f}")
