import sys
sys.path.append("/mym3d/")

import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from LaMed.src.dataset.multi_dataset import MyCapDataset
# from Bench.dataset.multi_dataset import CapDataset, MyCapDataset
# If the model is not from huggingface but local, please uncomment and import the model architecture.
from LaMed.src.model.language_model import *
import evaluate

accuracy = evaluate.load("accuracy")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")
target_dir = "/LaMed-Phi3-4B-patch-finetune-0000"


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default=f"models/{target_dir}")
    parser.add_argument('--max_length', type=int, default=768)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])

    # data
    parser.add_argument('--data_root', type=str, default="./Data/data")
    parser.add_argument('--cap_data_path', type=str, default="/mym3d/Data/output.json")
    parser.add_argument('--output_dir', type=str, default=f"./LaMed/output/{target_dir}/eval_caption/")

    parser.add_argument('--proj_out_num', type=int, default=256)
    parser.add_argument('--seg_enable', type=bool, default=True)

    parser.add_argument(
        '--modality_keys',
        nargs='+',
        default=["pet", "ct", "mask"],
        help="Determines the types of files needed"
    )

    return parser.parse_args(args)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels
        

def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    model = LamedPhi3ForCausalLM.from_pretrained(
        args.model_name_or_path,
        # device_map='auto',
        trust_remote_code=True
    )
    model = model.to(device=device)

    test_dataset = MyCapDataset(args, tokenizer=tokenizer, mode='val') # test1k
    # from torch.utils.data import random_split
    # total_len = len(full_dataset)
    # train_len = int(0.8 * total_len)
    # eval_len = total_len - train_len
    # train_dataset, test_dataset = random_split(
    #     full_dataset,
    #     [train_len, eval_len],
    #     generator=torch.Generator().manual_seed(42)
    # )

    test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=32,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
    )  

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = os.path.join(args.output_dir, "eval_caption.csv")

    with open(output_path, mode='w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Question", "Ground Truth", "pred", "bleu", "rouge1", "meteor"]) #, "bert_f1"
        for sample in tqdm(test_dataloader):
            question = sample["question"]
            answer = sample['answer']

            input_id = tokenizer(question, return_tensors="pt")['input_ids'].to(device=device)
            pet = sample["pet"].to(device=device)
            mask = sample["mask"].to(device=device)
            ct = sample["ct"].to(device=device)


            generation = model.generate(pet, mask, ct, input_id, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, top_p=args.top_p, temperature=args.temperature)
            generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
            # print(generated_texts[0])

            result = dict()
            decoded_preds, decoded_labels = postprocess_text(generated_texts, answer)
            # accuracy_score = accuracy.compute(predictions=decoded_preds, references=decoded_labels)
            # result["accuracy"] = accuracy_score['accuracy']

            try:
                bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
                result["bleu"] = bleu_score['bleu']

                rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=['rouge1'])
                result["rouge1"] = rouge_score['rouge1']

                meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
                result["meteor"] = meteor_score['meteor']

                # bert_score = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
                # result["bert_f1"] = sum(bert_score['f1']) / len(bert_score['f1'])

                writer.writerow([question[0], answer[0], generated_texts[0], result["bleu"], result["rouge1"], result["meteor"]]) # , result["bert_f1"]
            except:
                print("Had an issue", answer[0])


if __name__ == "__main__":
    main()
    # python3 Bench/eval/eval_caption.py 