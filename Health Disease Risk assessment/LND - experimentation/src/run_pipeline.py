from data_preprocessing import preprocess_data
from train_model import train_model
import config

def main():
    x, y = preprocess_data()

    # Train the model
    trained_model = train_model(x, y)
    trained_model.save(config.SAVE_MODEL_PATH + "bestModel.h5")    

if __name__ == "__main__":
    main()