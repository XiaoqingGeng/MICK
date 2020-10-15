# MICK

## Prerequisites
* pytorch >= 0.4.1
* pytorch = 0.3.1 for SNAIL

## Getting Started
* Train MLMAN
```
cd MLMAN_Proto
python3 train_demo.py --N_for_test "#relation classes in test set" --N_for_train 5 --K_for_train 15 --K_for_test 15 --Q 5 --batch 1 --max_length 80 --train_or_test train --dataset "data_source_folder" --use_sup_cost 1 --language chn --pretrain "pretrained model if any"

```

* Train Prototypical networks
```
cd MLMAN_Proto
python3 train_demo.py --N_for_test "#relation classes in test set" --N_for_train 5 --K_for_train 15 --K_for_test 15 --Q 5 --batch 1 --max_length 80 --train_or_test train --dataset "data_source_folder" --use_sup_cost 1 --language chn --pretrain "pretrained model if any"

```

* Train BertPair
```
cd BertPair_GNN
python3 train_demo.py --N "#relation classes in test set" --trainN 5 --K 1 --Q 1 --model pair --encoder bert --max_length 80 --train "data_source_folder"/train --test "data_source_folder"/test --val "data_source_folder"/test  --pair --hidden_size 768 --val_step 1000 --batch_size 1 --save_ckpt "save checkpoint" --language chn --sup_cost 1 --load_ckpt "pretrained model if any"  

```

* Train GNN
```
cd BertPair_GNN
python3 train_demo.py --N "#relation classes in test set" --trainN 5 --K 1 --Q 1 --model gnn --encoder cnn --max_length 80 --train "data_source_folder"/train --test "data_source_folder"/test --val "data_source_folder"/test --val_step 1000 --batch_size 1 --save_ckpt "save checkpoint" --language chn --sup_cost 1 --load_ckpt "pretrained model if any"  

```

* Train SNAIL
```
cd SNAIL
python3 train_demo.py --N "#relation classes in test set" --trainN 5 --K 1 --Q 1 --model gnn --encoder cnn --max_length 80 --train "data_source_folder"/train --test "data_source_folder"/test --val "data_source_folder"/test --val_step 1000 --batch_size 1 --save_ckpt "save checkpoint" --language chn --sup_cost 1 --load_ckpt "pretrained model if any"  
```

## Data
* Our proposed TinyRel-CM dataset in data/.
* To run models on dataset, the dataset folder should contain a "train.json", a "test.json" and a "rel2id.json".
  "train.json" and "test.json" files should look like:
```
  {"relation class name":[
	{"h":["head_entity_name", "head_entity_id", "head_entity_pos"],
	 "t":["tail_entity_name", "tail_entity_id", "tail_entity_pos"],
	 "tokens": [tokens in the sentence]},
	...
	],
  ...
  }
```  
  An example is shown in data/DISEASE\_DISEASE\_example, where group DISEASE-DISEASE is regarded as test set.
