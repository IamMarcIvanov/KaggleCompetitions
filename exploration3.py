import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor as knn_regressor
from sklearn.neighbors import KNeighborsClassifier as knn_classifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as ss

def make_sex_hot(df):
    one_hot_sex = pd.get_dummies(df['Sex'], prefix='Sex')
    df['Male'] = one_hot_sex['Sex_male'].values
    df['Female'] = one_hot_sex['Sex_female'].values
    df.drop(columns=['Sex'], inplace=True)
    return df

def get_train_test(df):
    train_df, test_df = train_test_split(df, test_size=0.2)
    print('train size', len(train_df.index))
    print('test size', len(test_df.index))
    print('train age is na', train_df['Age'].isna().sum())
    print('test age is na', test_df['Age'].isna().sum())
    return train_df, test_df

def fill_age_knn(df, n_neighbors=2):
    df_filled = df[df['Age'].notna()]
    X = df_filled.drop(columns=['Age']).to_numpy()
    y = df_filled['Age'].to_numpy()
    regressor = knn_regressor(n_neighbors=n_neighbors)
    regressor.fit(X, y)

    df_na = df[df['Age'].isna()]
    X_pred_df = df_na.drop(columns=['Age'])
    y_pred = regressor.predict(X_pred_df.to_numpy())
    print('number of predictions', len(y_pred))
    X_pred_df['Age'] = pd.Series(y_pred).values
    return pd.concat([df_filled, X_pred_df], ignore_index=True), regressor

def fill_test_age_knn(df, regressor):
    df_filled = df[df['Age'].notna()]
    X = df_filled.drop(columns=['Age']).to_numpy()
    y = df_filled['Age'].to_numpy()

    df_na = df[df['Age'].isna()]
    X_pred_df = df_na.drop(columns=['Age'])
    y_pred = regressor.predict(X_pred_df.to_numpy())
    print('number of predictions', len(y_pred))
    X_pred_df['Age'] = pd.Series(y_pred).values
    return pd.concat([df_filled, X_pred_df], ignore_index=True)

def get_survivors_knn(X_train, y_train, X_pred, n_neighbors=3):
    classifier = knn_classifier(n_neighbors=n_neighbors)
    classifier.fit(X_train, y_train)
    return classifier.predict(X_pred)

def get_best_n_neighbors(X_train, y_train, X_test, y_test, low=3, high=50):
    x, y = list(range(low, high + 1)), []
    for n_neighbors in x:
        y_pred = get_survivors_knn(X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), n_neighbors=n_neighbors)
        y.append(accuracy_score(y_test, y_pred))
    plt.plot(x, y)
    plt.xlabel('acc')
    plt.ylabel('num neighbors')
    plt.title(f'best num neighbors = {y.index(max(y)) + 1} ; max acc = {max(y)}')
    plt.show()
    return y.index(max(y)) + 1

def preprocess(path):
    df = pd.read_csv(path)
    df.drop(columns=['Ticket', 'Embarked', 'Name', 'Cabin'], inplace=True)
    print('dropped columns of Ticket, Embarked, Name, Cabin')
    df = make_sex_hot(df)
    print('removed sex column and added male and female as one hot')
    return df

def main():
    train_path = r'/mnt/windows_d/Svalbard/Data/KaggleCompetitions/titanic/train.csv'
    train_df, test_df = get_train_test(preprocess(train_path))
    X_train, y_train = train_df.drop(columns=['Survived']), train_df['Survived']
    scaler = ss()
    X_test, y_test = test_df.drop(columns=['Survived']), test_df['Survived']
    X_train, train_regressor = fill_age_knn(X_train)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = fill_test_age_knn(X_test, regressor=train_regressor)
    X_test = scaler.transform(X_test)
    n_neighbors = get_best_n_neighbors(X_train, y_train, X_test, y_test)

    test_path = r'/mnt/windows_d/Svalbard/Data/KaggleCompetitions/titanic/test.csv'
    ans_path = r'/mnt/windows_d/Svalbard/Data/KaggleCompetitions/titanic/ans.csv'
    train_df = preprocess(train_path)
    X_train, y_train = train_df.drop(columns=['Survived']), train_df['Survived']
    X_train, regressor = fill_age_knn(X_train)
    test_df = fill_test_age_knn(preprocess(test_path), regressor)
    test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mean())
    print(test_df.isna().sum())
    ans_df = pd.DataFrame()
    ans_df['PassengerId'] = test_df['PassengerId'].values
    ans_df['Survived'] = get_survivors_knn(
            X_train.to_numpy(), 
            y_train.to_numpy(), 
            test_df.to_numpy(),
            n_neighbors = n_neighbors
    )
    ans_df.to_csv(ans_path, index=False)

if __name__ == '__main__':
    main()
