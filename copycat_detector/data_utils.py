import pandas as pd

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


def get_class_from_category(category):
    if category == 'non':
        return 0
    elif category == 'orig':
        return -1
    else:
        return 1
