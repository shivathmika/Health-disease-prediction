"""import pandas as pd
from sklearn import model_selection
import config 

if __name__ == "__main__":
    # Training data is in a csv file called train.csv
    df = pd.read_csv(config.TRAINING_FILE)
    
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    
    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # fetch targets
    y = df.target.values
    
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f
    
    # save the new csv with kfold column
    df.to_csv(config.TRAINING_FILE_WITH_FOLDS, index=False)"""
    
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import config

def generate_folds():
    # Load the training dataset
    data = pd.read_csv(config.TRAINING_FILE)

    # Add a new column for fold assignment and initialize with -1
    data["kfold"] = -1

    # Shuffle the dataset
    data = data.sample(frac=1).reset_index(drop=True)

    # Extract the target labels
    targets = data["Outcome"].values

    # Set up StratifiedKFold for maintaining label distribution across folds
    skf = StratifiedKFold(n_splits=5)

    # Assign fold numbers to the "fold" column
    for fold_number, (train_idx, val_idx) in enumerate(skf.split(X=data, y=targets)):
        data.loc[val_idx, "kfold"] = fold_number

    # Export the dataset with fold information
    data.to_csv(config.TRAINING_FILE_WITH_FOLDS, index=False)
    print("file saved in destination")

if __name__ == "__main__":
    generate_folds()
