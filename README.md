## Things to do before running:
1. in command line: export OPENAI_API_KEY='....'

## Smoke test for evaluation (just to check if code works; small search depth and small time budget allocated for MCTS)
- `python3 -u evaluation/run_evaluation.py > nohup_evaluation_smoke_test.out  2>&1 &`

## Download conversation daily for Q function learning (offline)
1. Download the data: https://huggingface.co/datasets/daily_dialog/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet?download=true
2. Place the .parquet file into the main directory and rename it as `daily_dialogue.parquet`
3. Proceed to next section
   
## Pretraining a Q function (can use for termination heuristic in MCTS, or as a pretrained Q function to be finetuned with runtime MCTS)
- `nohup python3 -u evaluation/train_offline_q_function.py --data=daily_dialogue --number=200 > nohup_train_q_offline_small.out  2>&1 &` (note that "200" in command line means we sample 200 conversation starting point from `daily_dialogue.parquet` dataset (downloaded in previous step) to train the Q function).
- Note that the code checks if a model `trained_q_function_daily_dialogue` exists in the directory. If it already does, it will not train a new model.

# Evaluation
- ` python3 evaluation/run_evaluation.py --evaluation_data=evaluation_starters_simple.txt --evaluation_depth=5 --mcts_search_depth=8 --mcts_time=1000 --pretr
ained_q_function=model_pretrained_qfn`


