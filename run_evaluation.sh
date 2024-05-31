# greedy 
# nohup python3 -u evaluation/run_evaluation.py --evaluation_data=evaluation_starters_100.txt --evaluation_depth=5 --mcts_search_depth=8 --mcts_time=5 --pretrained_q_function=trained_q_function_daily_dialogueFULL --agent=greedy --result_file=50_starters_greedy_evaluation > output/50_starters_greedy_evaluation.out  2>&1 &

# # offline:
# nohup python3 -u evaluation/run_evaluation.py --evaluation_data=evaluation_starters_100.txt --evaluation_depth=5 --mcts_search_depth=8 --mcts_time=5 --pretrained_q_function=trained_q_function_daily_dialogueFULL --agent=pure_offline --result_file=50_starters_pure_offline_evaluation > output/50_starters_pure_offline_evaluation.out  2>&1 &

# # online 500s:
nohup python3 -u evaluation/run_evaluation.py --evaluation_data=evaluation_starters_100.txt --evaluation_depth=5 --mcts_search_depth=8 --mcts_time=500 --pretrained_q_function=trained_q_function_daily_dialogueFULL --agent=pure_online --result_file=50_starters_pure_online_search_time_500 > output/50_starters_pure_online_evaluation_500.out  2>&1 &

# online 1000s:
nohup python3 -u evaluation/run_evaluation.py --evaluation_data=evaluation_starters_100.txt --evaluation_depth=5 --mcts_search_depth=8 --mcts_time=1000 --pretrained_q_function=trained_q_function_daily_dialogueFULL --agent=pure_online --result_file=50_starters_pure_online_search_time_1000 > output/50_starters_pure_online_evaluation_1000.out  2>&1 &

# offline online 500s:
nohup python3 -u evaluation/run_evaluation.py --evaluation_data=evaluation_starters_100.txt --evaluation_depth=5 --mcts_search_depth=8 --mcts_time=500 --pretrained_q_function=trained_q_function_daily_dialogueFULL --agent=offline_online_mixed --result_file=50_starters_offline_online_mixed_search_time_500 > output/50_starters_offline_online_mixed_evaluation_500.out  2>&1 &

# offline online 1000s:
nohup python3 -u evaluation/run_evaluation.py --evaluation_data=evaluation_starters_100.txt --evaluation_depth=5 --mcts_search_depth=8 --mcts_time=1000 --pretrained_q_function=trained_q_function_daily_dialogueFULL --agent=offline_online_mixed --result_file=50_starters_offline_online_mixed_search_time_1000 > output/50_starters_offline_online_mixed_evaluation_1000.out  2>&1 &