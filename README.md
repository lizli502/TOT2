## TOT

We implement our code in Tensorflow and the code is tested under a server with 40-core Intel Xeon E5-2630 
v4 @ 2.20GHz CPU, 256G RAM and Nvidia GTX 1080 GPUs (with TensorFlow 1.13 and Python 3).

## Datasets
The preprocessed datasets are in the `data` directory (`e.g. data/amazon_book.txt`). Each line of the `txt` format data contains a `user id` and an `item id`, where both user id and item id are indexed from 1 consecutively. Each line represents one interaction between the user and the item. For every user, their interactions were sorted by timestamp. For each interaction, the corresponding review of the user to the item is in the file named  `amazon_book_reviews`. Due to GitHub's file size limit of 100.00 MB, we put the review file [here](https://drive.google.com/open?id=1NoaKua1dGFuMESTI5GZag5m910MCBPnw). You can download the file and put it inside the `data` directory if you want to run our code.

## Options
The training of the TOT model is handled by the main.py script that provides the following command line arguments.
```
--dataset            STR           Name of dataset.                  Default is "ml1m".
--train_dir          STR           Train directory.                  Default is "default".
--batch_size         INT           Batch size.                       Default is 128.    
--lr                 FLOAT         Learning rate.                    Default is 0.001.
--maxlen             INT           Maximum length of sequence.       Default is 50.
--user_hidden_units  INT           Hidden units of user.             Default is 50.
--item_hidden_units  INT           Hidden units of item.             Default is 50.
--num_blocks         INT           Number of blocks.                 Default is 2.
--num_epochs         INT           Number of epochs to run.          Default is 2001.
--num_heads          INT           Number of heads.                  Default is 1.
--dropout_rate       FLOAT         Dropout rate value.               Default is 0.5.
--threshold_user     FLOAT         SSE probability of user.          Default is 1.0.
--threshold_item     FLOAT         SSE probability of item.          Default is 1.0.
--num_blocks_nlp     INT           Number of blocks for nlp side.    Default is 2.
--num_heads_nlp      INT           Number of heads for nlp side.     Default is 1.
--text_maxlen        INT           Maximum length of user review.    Default is 64.
--glove_emb_dim      INT           Dimention of glove embeddings.    Default is 100.
--loss_coef_nlp      FLOAT         Loss coefficient of nlp.          Default is 1.0.
--l2_emb             FLOAT         L2 regularization value.          Default is 0.0.
--gpu                INT           Name of GPU to use.               Default is 0.
--print_freq         INT           Print frequency of evaluation.    Default is 10.
--k                  INT           Top k for NDCG and Hits.          Default is 10.
```
## Commands
To train a TOT model on `amazon_book` data.

```
python3 main.py --dataset=amazon_book --batch_size=32 --train_dir="default" --user_hidden_units 50 --item_hidden_units 50 --num_blocks=2 --maxlen=50 --dropout_rate 0.6 --num_heads=2 --num_heads_nlp=2 --lr=0.001 --num_epochs 4001 --gpu=5 --threshold_user=0.5 --threshold_item 0.99 --print_freq 10 --loss_coef_nlp 0.4 --num_blocks_nlp=2 --dropout_rate_nlp=0.4
```



