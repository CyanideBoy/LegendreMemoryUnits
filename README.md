# LegendreMemoryUnits
Exploring LMUs as part of CS726
# Directory structure:
* dataloaders/: contains files for returning data in desired format. One file per task.
* results/: Contains saved models, plots and loss arrays. One subfodler per task.
* ()_exp: () stands for the task e.g. mg/ptb_char. Main file which loads data and model, carries out training and stores plots/models
* utils.py: Store all common functions such as merging the LMU and LSTM plots to get one single plot.
* models.py: Contains LMU and LSTM cell models.
