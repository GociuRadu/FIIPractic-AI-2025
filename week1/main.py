from src.utils import load_dataset

def main():
    data = load_dataset("diabetes_dataset.csv")
    target="diabet"
    train_size=int(0.8*len(data))
    train_data=data[:train_size]
    test_data=data[train_size:]
    print("train data: ",train_data.shape)
    print("test data: ",test_data.shape)

if __name__ == "__main__":
    main()
