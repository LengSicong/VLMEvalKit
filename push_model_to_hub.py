#!/usr/bin/env python3
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


# Dictionary mapping between relative paths and target model names
PATH_TO_MODEL_NAME = {
    "projects/long-doc/llama-3.1-instruct-single-100k-multi-50k-sw/model_lr5e-7_batch512_epochs2_gpus8_linearSchedule_gnorm": "llama-3.1_single-multi-150k_5e-7",
    "projects/long-doc/mistral_rope1e7_single_100k_multi_50k_sw/model_lr1e-6_batch512_epochs2_gpus8_linearSchedule_gnorm": "mistral_single-multi-150k_1e-6",
    "projects/long-doc/mistral-longalpaca/model_lr1e-6_batch512_epochs2_gpus8_linearSchedule_gnorm": "mistral_longalpaca_1e-6",
    "projects/long-doc/mistral-longalign/model_lr1e-6_batch512_epochs2_gpus8_linearSchedule_gnorm": "mistral_longalign_1e-6", 
    "projects/long-doc/mistral-longreward/model_lr1e-6_batch512_epochs2_gpus8_linearSchedule_gnorm": "mistral_longreward_1e-6",
    "projects/long-doc/llama-3.1-instruct-longalpaca/model_lr5e-7_batch512_epochs2_gpus8_linearSchedule_gnorm": "llama-3.1_longalpaca_5e-7",
    "projects/long-doc/llama-3.1-instruct-longalign/model_lr5e-7_batch512_epochs2_gpus8_linearSchedule_gnorm": "llama-3.1_longalign_5e-7",
    "projects/long-doc/llama-3.1-instruct-longreward/model_lr5e-7_batch512_epochs2_gpus8_linearSchedule_gnorm": "llama-3.1_longreward_5e-7",
    "projects/long-doc/mistral-single-multi-graph20k/model_lr1e-6_batch512_epochs2_gpus8_linearSchedule_gnorm": "mistral_single-multi-graph20k_1e-6",
    "projects/long-doc/mistral-multi-graph20k/model_lr1e-6_batch512_epochs2_gpus8_linearSchedule_gnorm": "mistral_multi-graph20k_1e-6",
    "projects/long-doc/mistral-single-graph20k/model_lr1e-6_batch512_epochs2_gpus8_linearSchedule_gnorm": "mistral_single-graph20k_1e-6",
    "projects/long-doc/llama-3.1-instruct-single-multi-graph20k/model_lr5e-7_batch512_epochs2_gpus8_linearSchedule_gnorm": "llama-3.1_single-multi-graph20k_5e-7",
    "projects/long-doc/llama-3.1-instruct-multi-graph20k/model_lr5e-7_batch512_epochs2_gpus8_linearSchedule_gnorm": "llama-3.1_multi-graph20k_5e-7",
    "projects/long-doc/llama-3.1-instruct-single-graph20k/model_lr5e-7_batch512_epochs2_gpus8_linearSchedule_gnorm": "llama-3.1_single-graph20k_5e-7",
    "projects/long-doc/mistral-wildchat20k/model_lr1e-6_batch512_epochs2_gpus8_linearSchedule_gnorm": "mistral_wildchat20k_1e-6",
    "projects/long-doc/mistral-simple20k/model_lr1e-6_batch512_epochs2_gpus8_linearSchedule_gnorm": "mistral_simple20k_1e-6",
    "projects/long-doc/llama-3.1-instruct-wildchat20k/model_lr1e-6_batch512_epochs2_gpus8_linearSchedule_gnorm": "llama-3.1_wildchat20k_1e-6",
    "projects/long-doc/llama-3.1-instruct-sp35-65-20k/model_lr3e-7_batch512_epochs2_gpus8_linearSchedule_gnorm": "llama-3.1_sp35-65-20k_3e-7",
    "projects/long-doc/llama-3.1-instruct-sp35-65-20k/model_lr5e-7_batch512_epochs2_gpus8_linearSchedule_gnorm": "llama-3.1_sp35-65-20k_5e-7",
}



def push_model_to_hub(model_path, repo_id, token):
    """Push model to HuggingFace Hub"""
    try:
        print(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print(f"Pushing model to {repo_id}")
        model.push_to_hub(repo_id=repo_id, token=token)
        tokenizer.push_to_hub(repo_id=repo_id, token=token)
        
        print(f"Successfully pushed {repo_id} to Hugging Face Hub")
        return True
    except Exception as e:
        print(f"Error pushing {repo_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Push local models to Hugging Face Hub")
    parser.add_argument("--blob_dir", type=str, required=True, help="Base directory containing models")
    parser.add_argument("--hub_token", type=str, required=True, help="Hugging Face Hub token")
    parser.add_argument("--hub_username", type=str, required=True, help="Hugging Face Hub username")
    parser.add_argument("--dry_run", action="store_true", help="Print actions without pushing")
    args = parser.parse_args()
    
    # Clean up paths
    model_paths = [path.strip("/") for path in PATH_TO_MODEL_NAME.keys()]
    
    for rel_path in model_paths:
        full_path = os.path.join(args.blob_dir, rel_path)
        model_name = PATH_TO_MODEL_NAME.get(rel_path)
        repo_id = f"{args.hub_username}/{model_name}"
        
        print(f"\nProcessing: {rel_path}")
        print(f"Full path: {full_path}")
        print(f"Model name: {model_name}")
        print(f"Repository ID: {repo_id}")
        
        if args.dry_run:
            print("Dry run - skipping actual push")
        else:
            push_model_to_hub(full_path, repo_id, args.hub_token)
    
    print("\nAll models processed!")


if __name__ == "__main__":
    main()

## python push_model_to_hub.py --blob_dir /home/msranlpblob/jiaxi/ --hub_token $HF_TOKEN --hub_username jxjessieli --dry_run