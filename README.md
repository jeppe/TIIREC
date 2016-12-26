# Introduction
TIIREC is a tensor approach for tag-driven item recommendation. This repository holds an implementation with a map-reduce evaluation module. The format of input data should be [user_id, tag_id, item_id] with \t as the separation symbolic.

You could type "python TIIREC.py -h" to check the meaning of each paramters.

Type following commands in the terminal to run this program. You could find the evaluation result at "logs" director.
> python TIIREC.py --trainpath ./data/train.data --testpath ./data/test.data --dimension 10

