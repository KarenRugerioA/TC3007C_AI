import os, shutil

def configure_storage(dataset_path, original_name_dataset):
    try:
        shutil.rmtree(f'../data/{original_name_dataset}')
    except:
        pass
    try:
        shutil.rmtree(f'fragments/joblibs/{original_name_dataset}')
    except:
        pass

    # Creating the main folder in data
    os.mkdir(f'../data/{original_name_dataset}')

    # Saving the dataset in its correspondent folder
    with open(dataset_path, 'r') as origin, open(f'../data/{original_name_dataset}/{original_name_dataset}.csv', 'w') as dest:
        results = origin.read()
        dest.write(results)

    # Creating the files for the train and test
    os.mkdir(f'../data/{original_name_dataset}/train')
    os.mkdir(f'../data/{original_name_dataset}/test')

    # Creating the main folder for the joblibs
    os.mkdir(f'fragments/joblibs/{original_name_dataset}')
    os.mkdir(f'fragments/joblibs/{original_name_dataset}/model')
    os.mkdir(f'fragments/joblibs/{original_name_dataset}/etl')
    os.mkdir(f'fragments/joblibs/{original_name_dataset}/model/mlp')
    os.mkdir(f'fragments/joblibs/{original_name_dataset}/model/decission_tree')
    os.mkdir(f'fragments/joblibs/{original_name_dataset}/model/benchmark')
    os.mkdir(f'fragments/joblibs/{original_name_dataset}/model/random_forest')
    os.mkdir(f'fragments/joblibs/{original_name_dataset}/model/cnn')
