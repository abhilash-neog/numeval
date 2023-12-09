# NumEval@SemEval 2024

Mridul Khurana, Abhilash Neog, Aruj Nayak

CS 5624: Natural Language Processing

## Install the Requirements from
```
requirements.txt
```

## Model Running

### T5
Use `run_t5.py` to run the T5-large model.
Please adjust the `data` and `model` save path

Similarly, `run_t5-3b.py` for running T5-3B model.

## Llama 2 - 7B
Use `zero_shot_llama2.py` to run the zero-shot and few-shot performance for llama2.

Please adjust the `data` and `predictions` save path

## Evaluating the Predictions
Use `numhg_eval.py` for evaluating the predictions
```
python numhg_eval.py --tgt_path "path_to_labels.txt" --pre_path "path_to_predictions.txt" --num_gt_path "path_to_numerical_gt.txt"
```
