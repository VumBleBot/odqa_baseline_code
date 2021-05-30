# debug     : if true, change length of train_data to 100 and epoch to 1
#    - default: false
# report    : if true, report results to slack channel 
#    - default: false
#    - You must have a slack app, and fill out '../input/secrets.json' according to set form in README.md.
# data_path : the path for your data directory.
#    - default: ../input
# run_cnt   : a variable that determines the how many it runs
#    - default: 1
#    - if run_cnt is 2, if will be run twice with a different seed value

# python -m run --strategies RET_BM25 --debug False --report False --data_path {YOUR_DATA_PATH}

python -m run --strategies RET_BM25 --debug false --report false --run_cnt 1
