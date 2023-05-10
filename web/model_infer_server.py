import logging
import orjson
import requests
from flask import Flask, request, Response

app = Flask(__name__)

app.logger.disabled = True
logging.getLogger("werkzeug").disabled = True
logging.getLogger().setLevel(logging.WARNING)

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

pdj = "/mnt/cephfs/zhuchengqi/git/LLM/bigo_stanford_alpaca/eval/transformers_jh/src"
sys.path.append(pdj)
import transformers

model = None
tokenizer = None
generator = None

logger = logging.getLogger()


def load_model(model_name, eight_bit=0, device_map="auto"):
    global model, tokenizer, generator

    print("Loading " + model_name + "...")
    # config
    gpu_count = torch.cuda.device_count()
    print('gpu_count', gpu_count)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        # device_map=device_map,
        # device_map="auto",
        torch_dtype=torch.float16,
        # max_memory = {0: "14GB", 1: "14GB", 2: "14GB", 3: "14GB",4: "14GB",5: "14GB",6: "14GB",7: "14GB"},
        # load_in_8bit=eight_bit,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        cache_dir="cache"
    ).cuda()


@torch.inference_mode()
def generate_stream(model, tokenizer, params, context_len=2048, stream_interval=2, device="cuda"):
    prompt = params["prompt"]
    prompt_len = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_words_list = params.get("stop_words_list", [])
    stop_words_list.append("\n")
    stop_words_list.append("</s>")

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            out = model(
                torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device=device)
            out = model(input_ids=torch.as_tensor([[token]], device=device),
                        use_cache=True,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            for stop_str in stop_words_list:
                pos = output.lower().rfind(stop_str.lower(), prompt_len)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
                    break
            yield output

        if stopped:
            break

    del past_key_values
    return output


print("loading model ... ")
# model_dir = '/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/multitrun_conversation/ft_outs/checkpoint-1000'
# model_dir = '/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/multi_turns_conversation_nomask/ft_out_nomask/checkpoint-1600'
# model_dir='/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/multitrun_conversation/ft_outs_mask_instruct/checkpoint-1400'
# model_dir = "/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/tmp/finetune_out_gpt4_share_sex_soda_cot_multi/checkpoint-4000/"
# model_dir = "/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/multitrun_conversation/ft_outs_mask_instruct/checkpoint-1400"
# model_dir="/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/multitrun_conversation/ft_outs_mask_instruct/checkpoint-1500"
model_dir = "/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/multitrun_conversation/ft_outs_mask_instruct/checkpoint-1200"
load_model(model_dir)
print('load model done!!!')
print('-' * 100)


def bot(prompt_input, temperature=0.7, max_gen_len=256, stop_words_list=None):
    assert stop_words_list is not None, "stop text is None!!!"
    params = {
        "prompt": prompt_input,
        "temperature": temperature,
        "max_new_tokens": max_gen_len,
        "stop_words_list": stop_words_list,
    }
    print(prompt_input)
    print("-" * 200)

    skip_echo_len = len(prompt_input.replace("</s>", " ")) + 1
    stream = generate_stream(model, tokenizer, params)
    generated_text = None
    for outputs in stream:
        generated_text = outputs[skip_echo_len:].strip()
        generated_text = generated_text.split(" ")
    generated_text = " ".join(generated_text)
    print("-" * 50 + "model generate text" + '-' * 50)
    print(generated_text)
    print("-" * 200)

    return generated_text.strip()


@app.route("/api", methods=["POST"])
def receive():
    try:
        params = orjson.loads(request.data)
    except orjson.JSONDecodeError:
        logger.error("Invalid json format in request data: [{}].".format(request.data))
        res = {"status": 400, "error_msg": "Invalid json format in request data.", "server_info": "", }
        return Response(orjson.dumps(res), mimetype="application/json;charset=UTF-8", status=200)

    if 'prompt_input' not in params or not isinstance(params['prompt_input'], str) \
            or 'temperature' not in params or not isinstance(params['temperature'], float) \
            or 'max_gen_len' not in params or not isinstance(params['max_gen_len'], int) \
            or 'stop_words_list' not in params or not isinstance(params['stop_words_list'], list):
        logger.error("Invalid json request.")
        res = {"status": 400, "error_msg": "Invalid json request.", "server_info": "", }
        return Response(orjson.dumps(res), mimetype="application/json;charset=UTF-8", status=200)

    # print('[http/receive]requests:', params)
    prompt_input = params.get('prompt_input', "")
    temperature = params.get('temperature', 0)
    max_gen_len = params.get('max_gen_len', "")
    stop_words_list = params.get('stop_words_list', [])

    result = bot(prompt_input, temperature, max_gen_len, stop_words_list)

    res = {"status": 200, "result": result, "error_msg": "", "server_info": ""}
    return Response(orjson.dumps(res), mimetype="application/json;charset=UTF-8", status=200)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
