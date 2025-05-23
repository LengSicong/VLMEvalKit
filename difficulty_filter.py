import argparse
import json
from vllm import LLM, SamplingParams
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoProcessor, AutoTokenizer
import gc
import re

## CUDA_VISIBLE_DEVICES=7 python sample_answers.py --model_path /data/jx/pretrained_models/Qwen/Qwen2.5-VL-7B-Instruct --judge_model_path /data/jx/pretrained_models/Qwen/Qwen2.5-7B-Instruct --num_preds 10 --data_path /data/jx/reasoning/llm_reasoning_math.jsonl --output_dir /data/jx/reasoning/ --current_chunk 0 --total_chunks 8 --batch_size 16


def extract_last_boxed_answer(text):
    """Extracts the last boxed answer from a LaTeX-style solution, handling nested braces correctly."""
    pattern = r'(\\boxed\s*\{)'  
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        return None  # No \boxed{ found
    
    last_match = matches[-1]
    brace_count = 1
    extracted = []
    
    # Move right after the entire matched pattern (\boxed{ plus optional spaces)
    i = last_match.end()
    
    while i < len(text):
        char = text[i]
        
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                # Done: we've found the matching closing brace
                break
        extracted.append(char)
        i += 1
    
    return "".join(extracted).strip()


## params
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model path",
        required=True,
    )
    parser.add_argument(
        "--judge_model_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model path for judging correctness",
        required=True,
    )        
    parser.add_argument(
        "--num_preds",
        type=int,
        default=10,
        help="Number of completions to generate for each question",
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/raid/longhorn/jiaxi/data/reasoning/llm_reasoning_math.jsonl",
        help="Path to the data file",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/raid/longhorn/jiaxi/data/reasoning/",
        help="Path to save the predictions",
        required=True,
    )
    parser.add_argument(
        "--current_chunk",
        type=int,
        default=0,
        help="The current chunk number",
        required=True,
    )
    parser.add_argument(
        "--total_chunks",
        type=int,
        default=8,
        help="The total number of chunks",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size",
        required=True,
    )
    args = parser.parse_args()
    return args


# ======================
# Phase 1: sample predictions for each question
# ======================

def generate_predictions(args):
    model = LLM(
        model=args.model_path,
        tensor_parallel_size=1,  # one GPU
        trust_remote_code=True,
        enforce_eager=True,
        max_num_seqs=16,
        max_model_len=32768,
    )
    
    # sampling params, 10 samples per question
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
        n=10  # 10 samples per question
    )
        
    # prompt format (qwen official)
    processor = AutoProcessor.from_pretrained(args.model_path)
    def build_prompt(question, processor=processor):
        messages = [
            {"role": "system", "content": "You are a helpful assistant in solving math problems. Please provide the answer to the following math question. Ensure your final answer is in \\boxed{...}."},
            {"role": "user", "content": f"Question: {question}"},
            {"role": "assistant"}
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt


    def process_batch(batch):
        prompts = [build_prompt(item["problem"]) for item in batch]
        outputs = model.generate(prompts, sampling_params)
        
        output_path = f"{args.output_dir}/predictions_{args.current_chunk}_{args.total_chunks}.jsonl"
        with open(output_path, "a") as f:
            for data_item, output in zip(batch, outputs):
                f.write(json.dumps({
                    "problem": data_item["problem"],
                    "short_form_gt": data_item["short_form_gt"],
                    "predictions": [o.text for o in output.outputs]
                }) + "\n")
                
    
    with open(args.data_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    dataset = [sample for sample in dataset if sample["short_form_gt"]]
    total = len(dataset)
    total_to_process_data = dataset[args.current_chunk * total // args.total_chunks: (args.current_chunk + 1) * total // args.total_chunks]
    batch = []
    for idx, item in tqdm(enumerate(total_to_process_data), total=len(total_to_process_data), desc="Generating"):
        batch.append(item)
        if len(batch) >= args.batch_size:
            process_batch(batch)
            batch = []
            gc.collect()
    if batch:
        process_batch(batch)
    

# ======================
# Phase 2: LLM as a judge
# ======================
def judge_predictions(args):
    # use a language model to judge correctness
    judge_model = LLM(
        model=args.judge_model_path,
        tensor_parallel_size=1,
        trust_remote_code=True
    )
    
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=50
    )    

    tokenizer = AutoTokenizer.from_pretrained(args.judge_model_path)
    def build_judge_prompt(question, prediction, ground_truth, tokenizer=tokenizer):
        
        extracted_prediction = extract_last_boxed_answer(prediction)
        if extracted_prediction is None:
            ## take the last three lines
            extracted_prediction = prediction.split("\n")[-3:]
            extracted_prediction = "\n".join(extracted_prediction)
            
        prompt = """You are a math answer judge. Please strictly compare the predicted answer with the reference answer to determine if they are equivalent. Only answer "correct" or "incorrect".
        Consider the following cases as correct:
        1. Same numerical value (e.g. 42 and 42.0)
        2. Different reasonable representations (e.g. 1/2, \\frac{{1}}{{2}} and 0.5)
        3. Correct unit conversion (e.g. 0.1km and 100m)
        
        Question: {question}
        
        Predicted answer: {prediction}
        
        Reference answer: {ground_truth}
        
        Please judge if the prediction is correct. Only answer "correct" or "incorrect".
        """
        formatted_prompt = prompt.format(question=question, prediction=extracted_prediction, ground_truth=ground_truth)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
    with open(f"{args.output_dir}/predictions_{args.current_chunk}_{args.total_chunks}.jsonl", "r") as f:
        dataset = [json.loads(line) for line in f]
        
    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc="Judging"):
        prompts_item = [build_judge_prompt(item["problem"], pred, item["ground_truth_solution"]) for pred in item["predictions"]]
        outputs = judge_model.generate(prompts_item, sampling_params)
        dataset[idx]["judgements"] = [output.outputs[0].text.strip().lower() for output in outputs]
        
    with open(f"{args.output_dir}/judgements_{args.current_chunk}_{args.total_chunks}.jsonl", "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
            
           
# ======================
# Overall Pipeline
# ======================
if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path
    data_path = args.data_path
    output_dir = args.output_dir
    current_chunk = args.current_chunk
    total_chunks = args.total_chunks
    num_preds = args.num_preds

    generate_predictions(args)
    judge_predictions(args)
    
    print("Done!", f"Chunk {current_chunk}/{total_chunks}")