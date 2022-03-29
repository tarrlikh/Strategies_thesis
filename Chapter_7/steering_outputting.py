import settings
import random
import os
import json


def load(name_of_the_dataset):
    '''
    Loads the duration data from a dataset file into the local duration_datasets variable
    
    Uses global variables slur_id, duration_datasets
    '''

    
    with open('data/'+name_of_the_dataset+f'/{settings.slur_id}.json', 'r') as file:
        settings.duration_datasets[name_of_the_dataset]=json.load(file)

def clear(name_of_the_dataset):
    
    '''
    Deletes the dataset file and the duration_datasets variable
    
    Uses global variables duration_datasets
    '''
    

    
    if os.path.isfile('data/'+name_of_the_dataset+'.json'):
        os.remove('data/'+name_of_the_dataset+'.json')
        
    if name_of_the_dataset in settings.duration_datasets:
        del settings.duration_datasets[name_of_the_dataset]
        
def update(name_of_the_dataset, additional_durations):
    
    '''
    If needed, creates a directory for the dataset file;
    
    Overwrites the duration_dataset entry with that file
    
    Adds additional_durations to the duration_datasets
    
    Updates the file with the duration_dataset entry
    
    Uses global variables slur_id, duration_datasets
    
    Assumes all files are under data/{name_of_the_dataset}
    '''
    
    
    
    if not os.path.exists('data/'+name_of_the_dataset):
        print('the path didnt exist yet')
        os.makedirs('data/'+name_of_the_dataset)
    else:
        print('the path does exist already!')
    
    if os.path.isfile('data/'+name_of_the_dataset+f'/{settings.slur_id}.json'):
        
        with open('data/'+name_of_the_dataset+f'/{settings.slur_id}.json', 'r') as file:
            settings.duration_datasets[name_of_the_dataset]=json.load(file)
    
    else:        
        print(f'file {name_of_the_dataset}/{settings.slur_id}.json didnt exist yet!')
    
    if not name_of_the_dataset in settings.duration_datasets:
        
        settings.duration_datasets[name_of_the_dataset]=additional_durations
        
        print(f'entry {name_of_the_dataset} in duration_datasets didnt exist yet!')
    
    else:

        settings.duration_datasets[name_of_the_dataset]+=additional_durations
    
    
    with open('data/'+name_of_the_dataset+f'/{settings.slur_id}.json', 'w') as file:
            json.dump(settings.duration_datasets[name_of_the_dataset],file)