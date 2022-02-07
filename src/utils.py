import os


def get_models_path(path, keywords=None):
    file_list = os.listdir(path)
    models_path = []
    for i in file_list:
        if keywords is not None:
            for word in keywords:
                if word in i and i not in models_path:
                    models_path.append(f'{path}/{i}')

    return models_path
