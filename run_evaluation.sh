# greedy 
# nohup python3 -u evaluation/run_evaluation.py --evaluation_data=evaluation_starters_100.txt --evaluation_depth=5 --mcts_search_depth=8 --mcts_time=5 --pretrained_q_function=trained_q_function_daily_dialogueFULL --agent=greedy --result_file=50_starters_greedy_evaluation > output/50_starters_greedy_evaluation.out  2>&1 &

# random
#python3 -u evaluation/run_evaluation.py --evaluation_data=evaluation_starters_50.txt --evaluation_depth=5 --mcts_search_depth=5 --mcts_time=1000 --pretrained_q_function=trained_q_function_daily_dialogueFULL --agent=random --result_file=50_starters_random_agent

# # offline:
# nohup python3 -u evaluation/run_evaluation.py --evaluation_data=evaluation_starters_100.txt --evaluation_depth=5 --mcts_search_depth=8 --mcts_time=5 --pretrained_q_function=trained_q_function_daily_dialogueFULL --agent=pure_offline --result_file=50_starters_pure_offline_evaluation > output/50_starters_pure_offline_evaluation.out  2>&1 &

# # online 500s:
# nohup python3 -u evaluation/run_evaluation.py --evaluation_data=evaluation_starters_100.txt --evaluation_depth=5 --mcts_search_depth=8 --mcts_time=500 --pretrained_q_function=trained_q_function_daily_dialogueFULL --agent=pure_online --result_file=50_starters_pure_online_search_time_500 > output/50_starters_pure_online_evaluation_500.out  2>&1 &

# # online 1000s:
# nohup python3 -u evaluation/run_evaluation.py --evaluation_data=evaluation_starters_100.txt --evaluation_depth=5 --mcts_search_depth=8 --mcts_time=1000 --pretrained_q_function=trained_q_function_daily_dialogueFULL --agent=pure_online --result_file=50_starters_pure_online_search_time_1000 > output/50_starters_pure_online_evaluation_1000.out  2>&1 &

# # offline online 500s:
# nohup python3 -u evaluation/run_evaluation.py --evaluation_data=evaluation_starters_100.txt --evaluation_depth=5 --mcts_search_depth=8 --mcts_time=500 --pretrained_q_function=trained_q_function_daily_dialogueFULL --agent=offline_online_mixed --result_file=50_starters_offline_online_mixed_search_time_500 > output/50_starters_offline_online_mixed_evaluation_500.out  2>&1 &

# # offline online 1000s:
# nohup python3 -u evaluation/run_evaluation.py --evaluation_data=evaluation_starters_100.txt --evaluation_depth=5 --mcts_search_depth=8 --mcts_time=1000 --pretrained_q_function=trained_q_function_daily_dialogueFULL --agent=offline_online_mixed --result_file=50_starters_offline_online_mixed_search_time_1000 > output/50_starters_offline_online_mixed_evaluation_1000.out  2>&1 &

#nohup python3 -u evaluation/run_evaluation.py --evaluation_data=evaluation_starters_test.txt --evaluation_depth=5 --cuda=1 --mcts_search_depth=8 --mcts_time=100 --pretrained_q_function=trained_q_function_daily_dialogueFULL --agent=pure_online --result_file=test_depth_8_time_100_resetQ_LR_1 > output/test_depth_8_time_100_resetQ_LR_1.out 2>&1 &
#nohup python3 -u evaluation/run_evaluation.py --evaluation_data=evaluation_starters_test.txt --evaluation_depth=5 --cuda=2 --mcts_search_depth=8 --mcts_time=500 --pretrained_q_function=trained_q_function_daily_dialogueFULL --agent=pure_online --result_file=test_depth_8_time_500_resetQ_LR_1 > output/test_depth_8_time_500_resetQ_LR_1.out 2>&1 &
#nohup python3 -u evaluation/run_evaluation.py --evaluation_data=evaluation_starters_test.txt --evaluation_depth=5 --cuda=3 --mcts_search_depth=4 --mcts_time=100 --pretrained_q_function=trained_q_function_daily_dialogueFULL --agent=pure_online --result_file=test_depth_4_time_100_resetQ_LR_1 > output/test_depth_4_time_100_resetQ_LR_1.out 2>&1 &
nohup python3 -u evaluation/run_evaluation.py --evaluation_data=evaluation_starters_test.txt --evaluation_depth=3 --cuda=4 --mcts_search_depth=4 --mcts_time=50 --pretrained_q_function=trained_q_function_daily_dialogueFULL --agent=pure_online --result_file=testing_2_actions > output/testing_2_actions.out 2>&1 &