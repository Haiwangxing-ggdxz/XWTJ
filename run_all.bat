@echo on
call conda env list
@REM 替换自己的环境名称
call conda activate Tensorflow
call python ./Recall/Recall_itemcf.py
call python ./Recall/DSSM_recall.py
call python ./Recall/merge.py
call python ./Rank/Feat_Eng.py
call python ./Rank/din_rank.py
call python ./Rank/DCN_rank.py


pause