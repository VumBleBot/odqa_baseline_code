# debug, if true, change length of train_data to 100 and epoch to 1
# report, if true, report results to slack channel 
#     You must have a slack app, and fill out '../input/secrets.json' according to set form in README.md.
# data_path, the path for your data directory.
#     default: ../input

# if there is no '{YOUR_DATA_PATH}/data/test_dataset',
#     This script will not work

python -m predict --strategies RET_BM25 --debug False --report False --data_path {YOUR_DATA_PATH}
