import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from copycat_detector import calculate_containment_value

CATEGORY_COLUMN = 'Category'


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


def calculate_containment_value(plagiarism_df, ngram_size, target_file):
    count_vectorizer = CountVectorizer(analyzer='word', ngram_range=(ngram_size, ngram_size))
    ngram_array = count_vectorizer.fit_transform(raw_documents=plagiarism_df['Text']).toarray()

    answer_index, source_index = get_answer_and_source(plagiarism_df=plagiarism_df, task_value=answer_task)

    return calculate_containment_value(ngram_array=ngram_array, answer_index=answer_index, source_index=source_index)


def get_answer_and_source(plagiarism_df, target_file):
    answer_index = plagiarism_df.index[plagiarism_df['File'] == target_file]
    answer_task = plagiarism_df.at[answer_index[0], 'Task']

    source_row = plagiarism_df[(plagiarism_df['Task'] == answer_task) & (plagiarism_df['Class'] == -1)]

    return answer_index[0], source_row.index[0]


def get_class_from_category(category):
    if category == 'non':
        return 0
    elif category == 'orig':
        return -1
    else:
        return 1
