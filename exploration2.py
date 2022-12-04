import pandas as pd
import matplotlib.pyplot as plt

def make_sex_hot(df):
    one_hot_sex = pd.get_dummies(df['Sex'], prefix='Sex')
    df['Male'] = one_hot_sex['Sex_male'].values
    df['Female'] = one_hot_sex['Sex_female'].values
    df.drop(columns=['Sex'], inplace=True)
    return df

def preprocess(path):
    df = pd.read_csv(path)
    df.drop(columns=['Ticket', 'Embarked', 'Name', 'Cabin'], inplace=True)
    print('dropped columns of Ticket, Embarked, Name, Cabin')
    df = make_sex_hot(df)
    print('removed sex column and added male and female as one hot')
    return df

def main():
    df = preprocess(TRAIN_PATH)
    plt.matshow(df.corr())
    plt.xticks(range(df.shape[1]), df.columns, rotation=45)
    plt.yticks(range(df.shape[1]), df.columns)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    TRAIN_PATH = r'/mnt/windows_d/Svalbard/Data/KaggleCompetitions/titanic/train.csv'
    TEST_PATH = r'/mnt/windows_d/Svalbard/Data/KaggleCompetitions/titanic/test.csv'
    main()
