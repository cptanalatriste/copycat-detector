import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from copycat_detector import calculate_containment_value

CATEGORY_COLUMN = 'Category'
TASK_COLUMN = 'Task'
DATATYPE_COLUMN = 'Datatype'
CLASS_COLUMN = 'Class'


def pre_process_data_file(csv_file):
    plagiarism_df = pd.read_csv(csv_file)
    plagiarism_df['Class'] = plagiarism_df[CATEGORY_COLUMN].map(get_class_from_category)

    category_mapping = {'non': 0,
                        'heavy': 1,
                        'light': 2,
                        'cut': 3,
                        'orig': -1}

    plagiarism_df = plagiarism_df.replace({CATEGORY_COLUMN: category_mapping})

    return plagiarism_df


def calculate_containment_from_df(plagiarism_df, ngram_size, target_file):
    count_vectorizer = CountVectorizer(analyzer='word', ngram_range=(ngram_size, ngram_size))
    ngram_array = count_vectorizer.fit_transform(raw_documents=plagiarism_df['Text']).toarray()

    answer_index, source_index = get_answer_and_source(plagiarism_df=plagiarism_df, target_file=target_file)

    return calculate_containment_value(ngram_array=ngram_array, answer_index=answer_index, source_index=source_index)


def get_answer_and_source(plagiarism_df, target_file):
    answer_index = plagiarism_df.index[plagiarism_df['File'] == target_file]
    answer_task = plagiarism_df.at[answer_index[0], TASK_COLUMN]

    source_index = plagiarism_df.index[
        (plagiarism_df[TASK_COLUMN] == answer_task) & (plagiarism_df[CLASS_COLUMN] == -1)]

    return answer_index[0].item(), source_index[0].item()


def generate_train_test_data(original_dataframe, features_dataframe, selected_features):
    merged_dataframe = pd.concat([original_dataframe, features_dataframe], axis=1)

    train_dataframe = merged_dataframe.loc[merged_dataframe[DATATYPE_COLUMN] == 'train']
    test_dataframe = merged_dataframe.loc[merged_dataframe[DATATYPE_COLUMN] == 'test']

    train_x = train_dataframe.loc[:, selected_features].to_numpy()
    test_x = test_dataframe.loc[:, selected_features].to_numpy()

    train_y = train_dataframe[CLASS_COLUMN].to_numpy()
    test_y = test_dataframe[CLASS_COLUMN].to_numpy()

    return train_x, train_y, test_x, test_y


def get_class_from_category(category):
    if category == 'non':
        return 0
    elif category == 'orig':
        return -1
    else:
        return 1
