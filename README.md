
## How to run:
```
python3 -u evaluation/run_evaluation.py \
--evaluation_data=evaluation_starters_love.txt \
--evaluation_depth=5 \
--cuda_for_llm=5 \
--cuda_for_q_and_embedding=5 \
--cuda_for_transition=0 \
--transition_model_dir="models/deterministic/" \
--mcts_search_depth=8 \
--mcts_time=500 \
--reward_func=harmful \
--trials=5 \
--lr=0.0001 \
--pretrained_q_function=trained_q_function_daily_dialogueFULL \
--embedding=llama \
--agent=semantic_online \
--result_file=output \
--evaluation_mode=None
```

## Description of what each arg means:
```
python3 -u evaluation/run_evaluation.py \
--evaluation_data=evaluation_starters_love.txt \ ### conversation starter files
--evaluation_depth=5 \ ### how many evaluation decision steps (so, LLM makes 5 decisions during evaluation; aka a conversation of 10 turns will happen) 
--cuda_for_llm=5 \ ### cuda for create_human_and_llm() and Llama_2_Guard_Reward()
--cuda_for_q_and_embedding=5 \ ### cuda for Q function and embedding_model_llama(), I realised embedding_model_llama() uses Llama guard as well so i think we are initializing twice here, using unncessary GPUs oops.
--cuda_for_transition=0 \ ### cuda loading transition models
--transition_model_dir="models/deterministic/" \ ### directory containing the transition models
--mcts_search_depth=8 \ ### during MCTS how many conversation turns is searched until; note this counts the human turn as well (so, during mcts LLM makes 3-4 decisions)
--mcts_time=500 \ ### how many seconds of MCTS to run for
--reward_func=harmful \ ### either harmful or length.
--trials=5 \ ### repeat trials
--lr=0.00001 \ ### LR used in Q function, actually i used 0.00001 for length. for harmfulness not sure what is a good value.
--pretrained_q_function=trained_q_function_daily_dialogueFULL \ ### initialize Q function from an offline trained one. Only useful if agent is "offline_online_mixed"
--embedding=llama \ ### embedding model. use llama now.
--agent=semantic_online \ ### greedy, random, pure_online (vanilla MCTS), offline_online_mixed, semantic_online (ours). For some agents, some previous args don't matter.
--result_file=output \ ### a csv file will be created in directory after everything is finished.
--evaluation_mode=gpt ### this is optional. If I put None, it defaults to using llama for evaluation. you can put gpt to use chatgpt for evaluation
```

## Things to do before running:
1. in command line: export OPENAI_API_KEY='....'

## Download conversation daily for Q function learning (offline)
1. Download the data: `wget  https://huggingface.co/datasets/daily_dialog/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet?download=true -O daily_dialogue.parquet`
2. Proceed to next section
   
## Pretraining a Q function (can use for termination heuristic in MCTS, or as a pretrained Q function to be finetuned with runtime MCTS)
- `nohup python3 -u evaluation/train_offline_q_function.py --data=daily_dialogue --number=200 > nohup_train_q_offline_small.out  2>&1 &` (note that "200" in command line means we sample 200 conversation starting point from `daily_dialogue.parquet` dataset (downloaded in previous step) to train the Q function).
- Note that the code checks if a model `trained_q_function_daily_dialogue` exists in the directory. If it already does, it will not train a new model.

## Smoke test for evaluation (just to check if code works; small search depth and small time budget allocated for MCTS)
- Make sure we have trained a Q function locally in **Section: Pretraining a Q function** (some offline agent requires it to work)
- `python3 -u evaluation/run_evaluation.py > nohup_evaluation_smoke_test.out  2>&1 &`
  
## Evaluation
- Make sure we have trained a Q function locally in **Section: Pretraining a Q function** (some offline agent requires it to work)
- ` python3 evaluation/run_evaluation.py --evaluation_data=evaluation_starters_simple.txt --evaluation_depth=5 --mcts_search_depth=8 --mcts_time=1000 --pretr
ained_q_function=model_pretrained_qfn`

## Training transition model
- go to [https://drive.google.com/drive/u/0/folders/1DeiOgO01ljM7BoNFNnIJN9u0KkV2f1HG](https://drive.google.com/drive/folders/1DeiOgO01ljM7BoNFNnIJN9u0KkV2f1HG?usp=drive_link) and download the two pkl file to the directory.
- `python3 train_transition_model.py` to train the transition model.


