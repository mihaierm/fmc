Files in this package:

nn/*: contains 2 saved Tensorflow networks. "good1" is the one used, "bad1" is just an example used in writing the notebook
tf_rec.py: the main script
user_book.csv: the user/book dataset
user_char.csv: the user traits dataset
fmc_challenge_presentation.ipynb: notebook containing some exploratory analysis and explanation of the solution

Requirements:
The code was developed and tested on Python 3.
Packages needed to run tf_rec.py: tensorflow, numpy, pandas
Packages needed to run the notebook: tensorflow, numpy, pandas, matplotlib, sklearn
Alternatively, the notebook can be viewed as gist here: https://gist.github.com/mihaierm/295f76098f618eaf12c70aa2e9e42691

How to run:

1. To try building a new model, use:

python tf_rec.py build

This will load and process the dataset, perform training on 10 networks in a 9+1 fold splitting of data and finally calculate the hit rate on a 9 rows validation dataset
You will be asked at the end if you want to save the network (will replace nn/good1)
The progress of the process is shown throughout in the command line


2. To see the results on the existing model, use:

python tf_rec.py check


3. To see the results on the test model, use:

python tf_rec.py test

Note that test_user_book.csv must be in the same folder in this case