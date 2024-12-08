# for loop over dataset
toh1
for DATASET in A-OKVQA ChartQA_TEST DocVQA_VAL InfoVQA_VAL TextVQA_VAL AI2D_TEST MathVista_MINI
    # for loop over ckpt
    for CKPT in mm-o1/Qwen2VL-72B-sft-v1 mm-o1/Qwen2VL-72B-sft-v2
        for COT in 0 1
            orun -g 8 --name evmmc --upload_pdx --link outputs --cmd \
                CKPT=$CKPT DATASET=$DATASET ENABLE_COT=$COT bash scripts/eval/run_eval.sh
        end
    end
end