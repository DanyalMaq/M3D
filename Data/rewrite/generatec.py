from huggingface_hub import InferenceClient
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def make_model(path: str = "Qwen/Qwen2.5-3B-Instruct"):

    model_name = "Qwen/Qwen2.5-3B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def generate(model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None, text: str = None):

    if any([model is None, tokenizer is None, text is None]):
        print("Something was none in generate")

    # text = "Head/Neck: There is asymmetric fullness within the left tonsillar pillar, with asymmetric increased FDG uptake, with SUV max 10.9 (axial slice 60)."
    prompt = f"""
    You are an expert in radiology report parsing.

    Given the following sentence:
    \"\"\"{text}\"\"\"

    Extract and format the information in this exact format:
    Region: <region>
    Anatomy: <anatomy>
    Finding: You should just copy the sentence back here but remove the numerical values from it. 
    SUV Max: <value>
    Axial Slice: <value>

    If any field is missing, write "N/A".

    Return only the formatted fields.

    Begin:
    """
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    # generated_ids = [
    #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    # ]

    # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    response = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    # print(response)
    return response

def batch_generate(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, texts: list):
    responses = []
    
    messages_batch = [
        [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant helping me parse radiology reports."},
            {"role": "user", "content": f"""
                You are an expert in radiology report parsing.

                Given the following sentence:
                \"\"\"{text}\"\"\"

                Extract and format the information in this exact format:
                Region: <region>
                Anatomy: <anatomy>
                Finding: You should just copy the sentence back here but remove the numerical values from it. 
                SUV Max: <value>
                Axial Slice: <value>

                If any field is missing, write "N/A".

                Return only the formatted fields.

                Begin:
            """}
        ]
        for text in texts
    ]

    input_texts = tokenizer.apply_chat_template(
        messages_batch,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024
    )

    generated_ids = [
        output_ids[input_len:]
        for input_len, output_ids in zip(model_inputs["input_ids"].shape[1] * [model_inputs["input_ids"].size(1)], generated_ids)
    ]

    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(len(responses))

    # for response in responses:
        # print(response)

    return responses


def main():
    parser = argparse.ArgumentParser(description="Generate formatted radiology information.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-8B", help="Path or name of the model")
    # parser.add_argument("--text", type=str, default="Head/Neck: There is asymmetric fullness within the left tonsillar pillar, with asymmetric increased FDG uptake, with SUV max 10.9 (axial slice 60).", help="Input radiology text")

    args = parser.parse_args()

    # text = "Head/Neck: There is asymmetric fullness within the left tonsillar pillar, with asymmetric increased FDG uptake, with SUV max 10.9 (axial slice 60)."
    # complete = [text, text]

    model, tokenizer = make_model(args.model_path)
    # r = generate(model, tokenizer, args.text)
    # r = batch_generate(model, tokenizer, complete)
    print("printing all responses")
    # print(r)

    import os

    # Your generate function (assumed already defined elsewhere)
    # from your_module import generate

    BASE_DIR = '/mym3d/Data/data/training_npy_ct'
    # print(os.listdir(BASE_DIR))

    for root, dirs, files in os.walk(BASE_DIR):
        if 'text.txt' in files:
            text_path = os.path.join(root, 'text.txt')
            struct_path = os.path.join(root, 'struct.txt')
            
            # if os.path.exists(struct_path):
            #     print(f"Skipping {root}: struct.txt already exists.")
            #     continue
            
            try:
                text = None
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()

                # Assuming generate() returns a string
                result = generate(model, tokenizer, text)
                print(result)

                with open(struct_path, 'w', encoding='utf-8') as f:
                    f.write(result)

                print(f"Created struct.txt for {root}")
            except Exception as e:
                print(f"Error processing {root}: {e}")



if __name__ == "__main__":
    main()