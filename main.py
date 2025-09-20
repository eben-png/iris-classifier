import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

def get_accuracy_split(train_X, val_X, train_y, val_y):
    model = RandomForestClassifier(random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    accuracy = accuracy_score(val_y, preds_val)
    return(accuracy)

def get_accuracy_full(X,val_x,y,val_y):
    model = RandomForestClassifier(random_state=0)
    model.fit(X, y)
    preds_val = model.predict(val_X)
    accuracy = accuracy_score(val_y, preds_val)
    return(accuracy)

if __name__ == "__main__":
    # Load data
    iris_file_path = "iris_synthetic_data.csv"
    iris_data = pd.read_csv(iris_file_path)

    # Filter data
    y = iris_data["label"]
    iris_features = ["sepal length","sepal width","petal length","petal width"]
    X = iris_data[iris_features]
    X = pd.get_dummies(iris_data[iris_features])
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    answer = ""
    while answer != "0":
        print("1. Test custom inputs")
        print("2. Get accuracy (half data)")
        print("3. Get accuracy (full data)")
        print("0. Exit")

        answer = input("What will you do? ")

        match(answer):
            case "1":
                sepal_length = float(input("Sepal Length: "))
                sepal_width = float(input("Sepal Width: "))
                petal_length = float(input("Petal Length: "))
                petal_width = float(input("Petal Width: "))

                model = RandomForestClassifier(random_state=0)
                model.fit(X, y)

                custom_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                        columns=iris_features)

                # Predict
                prediction = model.predict(custom_df)
                print("Predicted label:", prediction[0])
            case "2":
                print(get_accuracy_split(train_X, val_X, train_y, val_y))
            case "3":
                print(get_accuracy_full(X, val_X, y, val_y))
        
    print("Goodbye")