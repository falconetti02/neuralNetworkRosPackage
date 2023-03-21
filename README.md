this is a ros package that has a neural network
`src/beginner_tutorials/scripts` in this directory there are python files
file named `matris_train.py` trains a model using train_dataset.csv file and writes the weight and bias values in different csv files
then you can run rospackage
there are listener.py and talker.py 
talker.py is a publisher and publishes what is in test_dataset.csv file 
listener.py is a subscriber and subscribes what talker.py publishes

you can run your ros in a terminal with the code `roscore`
in a new terminal write `rosrun beginner_tutorials listener.py`
in a new terminal write `rosrun beginner_tutorials talker.py`
