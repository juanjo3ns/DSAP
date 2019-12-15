testing:
	python main.py config_test
deltas:
	python main.py config_deltas
all:
	python main.py config_all
	python main.py config_all23
	python main.py config_mfcc23
	python main.py config_deltas23
	python main.py config_nmf23

train:
	python main.py config_metricheck
	python main.py config_deltas
	python main.py config_mixup2_mfcc
	python main.py config_mixup2_deltas
	python main.py config_mixup2_nmf

weights:
	python main.py config_no
	python main.py config_yes
mixup:
	python main.py config_rate2
	python main.py config_rate4
	python main.py config_rate6
nmf:
	python main.py config_yes
	python main.py config_no
