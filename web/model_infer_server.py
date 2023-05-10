import logging
import orjson
import requests
from flask import Flask, request, Response
import os
import sys

pdj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"pdj:{pdj}")
sys.path.append(pdj)

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream

app = Flask(__name__)

app.logger.disabled = True
logging.getLogger("werkzeug").disabled = True
logging.getLogger().setLevel(logging.WARNING)

model = None
tokenizer = None
generator = None

logger = logging.getLogger()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def load_model(load_8bit: bool = False, base_model: str = "", lora_weights: str = "tloen/alpaca-lora-7b"):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert base_model, "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return model, tokenizer


def generate_stream(
        input_prompt,
        model,
        tokenizer,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stop_words_list=["###"],
        **kwargs,
):
    prompt_len = len(input_prompt)
    inputs = tokenizer(input_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }

    # Stream the reply 1 token at a time.
    # This is based on the trick of using 'stopping_criteria' to create an iterator,
    # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

    def generate_with_callback(callback=None, **kwargs):
        kwargs.setdefault(
            "stopping_criteria", transformers.StoppingCriteriaList()
        )
        kwargs["stopping_criteria"].append(
            Stream(callback_func=callback)
        )
        with torch.no_grad():
            model.generate(**kwargs)

    def generate_with_streaming(**kwargs):
        return Iteratorize(
            generate_with_callback, kwargs, callback=None
        )

    decoded_output = None
    with generate_with_streaming(**generate_params) as generator:
        for output in generator:
            # new_tokens = len(output) - len(input_ids[0])
            decoded_output = tokenizer.decode(output)

            if output[-1] in [tokenizer.eos_token_id]:
                break

            stopped = False
            for stop_str in stop_words_list:
                pos = decoded_output.lower().rfind(stop_str.lower(), prompt_len)
                if pos != -1:
                    decoded_output = decoded_output[:pos]
                    stopped = True
                    break
            if stopped:
                break

    assert decoded_output is not None, 'Error: decoded_output is None!'
    return decoded_output[prompt_len + 1:].strip()


# ----------------------------------------------------------------------------------------
# 加载模型
# ----------------------------------------------------------------------------------------
print("loading model ... ")
base_model = 'decapoda-research/llama-7b-hf'
# 加载网上模型
# lora_model = 'tloen/alpaca-lora-7b'
# 加载自己的模型
lora_model = "/mnt/cephfs/hjh/train_record/nlp/lora_stanford_alpaca/llama-7b-hf_alpaca_cleand_and_gpt4_fine_out"
model, tokenizer = load_model(load_8bit=True, base_model=base_model, lora_weights=lora_model)
print('load model done!!!')
print('-' * 100)


def bot(prompt_input, temperature=0.7, max_gen_len=256, stop_words_list=None):
    assert stop_words_list is not None, "stop text is None!!!"
    print(prompt_input)
    print("-" * 200)
    generated_text = generate_stream(prompt_input, model, tokenizer, temperature=temperature,
                                     max_new_tokens=max_gen_len, stop_words_list=stop_words_list)
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

    prompt_input = params.get('prompt_input', "")
    temperature = params.get('temperature', 0.7)
    max_gen_len = params.get('max_gen_len', "")
    stop_words_list = params.get('stop_words_list', [])

    result = bot(prompt_input, temperature, max_gen_len, stop_words_list)

    res = {"status": 200, "result": result, "error_msg": "", "server_info": ""}
    return Response(orjson.dumps(res), mimetype="application/json;charset=UTF-8", status=200)


if __name__ == '__main__':
    app.run(debug=False, port=5000)
