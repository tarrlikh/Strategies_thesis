
# coding: utf-8

# In[2]:


import numpy as np
import random
from scipy import linalg
from scipy.optimize import minimize
from itertools import product
import json
import os
import sys
import time
from string import *

import matplotlib.pyplot as plt


# In[90]:

#System size: a crucial step influencing the rest of the code

system_size=3

type_of_protocol=f'W_state_blind_99%_durations'

number_of_experiments=eval(sys.argv[1])

slur_id=sys.argv[2]

iterations=100000

fit_function_threshold=0.99

spin_dim=2
duration_datasets=dict()

print(f'W-state steering, in total {number_of_experiments} simulations')


###### Data management #######

def load(name_of_the_dataset):
    '''
    Loads the duration data from a dataset file into the local duration_datasets variable
    
    Uses global variables slur_id, duration_datasets
    '''
    
    with open('data/'+name_of_the_dataset+f'/{slur_id}.json', 'r') as file:
        duration_datasets[name_of_the_dataset]=json.load(file)

def clear(name_of_the_dataset):
    
    '''
    Deletes the dataset file and the duration_datasets variable
    
    Uses global variables duration_datasets
    '''
    
    
    if os.path.isfile('data/'+name_of_the_dataset+'.json'):
        os.remove('data/'+name_of_the_dataset+'.json')
        
    if name_of_the_dataset in duration_datasets:
        del duration_datasets[name_of_the_dataset]
        
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
        os.mkdir('data/'+name_of_the_dataset)
        print('the path didnt exist yet')
    else:
        print('the path does exist already!')
    
    if os.path.isfile('data/'+name_of_the_dataset+f'/{slur_id}.json'):
        
        with open('data/'+name_of_the_dataset+f'/{slur_id}.json', 'r') as file:
            duration_datasets[name_of_the_dataset]=json.load(file)
    
    else:        
        print(f'file {name_of_the_dataset}/{slur_id}.json didnt exist yet!')
    
    if not name_of_the_dataset in duration_datasets:
        
        duration_datasets[name_of_the_dataset]=additional_durations
        
        print(f'entry {name_of_the_dataset} in duration_datasets didnt exist yet!')
    
    else:

        duration_datasets[name_of_the_dataset]+=additional_durations
    
    
    with open('data/'+name_of_the_dataset+f'/{slur_id}.json', 'w') as file:
            json.dump(duration_datasets[name_of_the_dataset],file)


# In[15]:


def chained_kron(list_of_matrices):
    output=list_of_matrices[0]
    for matrix in list_of_matrices[1:]:
        output=np.kron(output,matrix)
    return(output)    


# In[56]:


'''
Actions on the state: never_clicked action, ever_clicked action
'''

def never_clicked(state, coupling):
    return(operator_action(state, coupling[0]))

def ever_clicked(state, coupling):        
    return(operator_action(state, coupling[1]))

# def operator_action(state, operator):
    
#     new_state=operator@state
#     norm=np.linalg.norm(new_state)
#     probability=norm**2
    
#     if probability!=0:
#         new_state=new_state/norm
        
#     return(new_state,probability)

'''
Same op action but now with DMs. 
'''

def operator_action(state, operator):
    
    new_state=operator@state@operator.T
    probability=np.trace(new_state)
    
    if probability!=0:
        new_state=new_state/probability
        
    return(new_state,probability)


# In[43]:




# def glob_fit(state):
#     return((mb_rho@np.outer(state,state)).trace() )

'''
Same glob_fit but now with DMs. 
'''

def glob_fit(state):
    return((mb_rho@state).trace() )



# In[44]:



    


# In[45]:


sigma_minus=np.array([[0,0],[1,0]])
sigma_plus=np.array([[0,1],[0,0]])
projector_up=np.array([[1,0],[0,0]])
projector_down=np.array([[0,0],[0,1]])
identity=np.array([[1,0],[0,1]])

up=np.array([1,0])
down=np.array([0,1])

gamma=0.1
def P_func(V):
    return(chained_kron([identity for i in range(system_size)])-V.T@V)


# In[46]:


W1=gamma*(chained_kron([sigma_plus, identity, identity])-chained_kron([identity, identity, sigma_plus]))
R1=P_func(W1)

W2=gamma*(chained_kron([sigma_minus, identity, sigma_minus]))
R2=P_func(W2)

W3=gamma*(chained_kron([sigma_minus,sigma_plus,identity])-chained_kron([projector_down,projector_up,identity]))
R3=P_func(W3)

W4=gamma*(chained_kron([identity, sigma_plus, sigma_minus])-chained_kron([identity, projector_up, projector_down]))
R4=P_func(W4)

total_coupling_list=[[R1,W1], [R2,W2], [R3,W3], [R4, W4]]


up_up_up=chained_kron([up,up,up])

up_up_down=chained_kron([up, up, down])
up_down_up=chained_kron([up, down, up])
down_up_up=chained_kron([down, up, up])

psi_plus=chained_kron([up,down,down])+chained_kron([down,up,down])+chained_kron([down,down,up])
psi_plus=psi_plus/np.linalg.norm(psi_plus)

psi_minus=chained_kron([up,down,down])-chained_kron([down,down,up])
psi_minus=psi_minus/np.linalg.norm(psi_minus)

psi_plusminus=chained_kron([up,down,down])+chained_kron([down,down,up])-2*chained_kron([down,up,down])
psi_plusminus=psi_plusminus/np.linalg.norm(psi_plusminus)

down_down_down=chained_kron([down, down, down])

mb_rho=np.outer(psi_plus,psi_plus)




# In[48]:


def coupling_strategy_function(scenario, strategies):

    if str(scenario) in strategies[0]:
        coupling_strategy=strategies[0][str(scenario)]
    else:
        coupling_strategy=strategies[1]
    
    return(coupling_strategy)


# In[49]:


strategies_W_state_feedback=(
    {"[]" : [0],
    "[0]": [2,3]},
[2,3])

strategies_W_state_blind=({},
[0,1,2,3])


# In[50]:


# starting_state=(np.outer(up_up,up_up)+np.outer(down_down,down_down)+np.outer(phi_plus,phi_plus)+np.outer(phi_minus,phi_minus))/4.
starting_state=np.outer(down_down_down,down_down_down)


# In[ ]:

global_performance=[]
durations=[]


stopwatch=time.time()

for j in range(number_of_experiments):
    
    current_state=starting_state
    
    current_glob_fit=glob_fit(current_state)
    
    steering_scenario=[]
    
    
    
    state_overlaps=[]
    
    for i in range(iterations):
        
    
        
        '''For scenario-based steering (start):'''
        coupling_strategy=coupling_strategy_function(steering_scenario, strategies_W_state_blind)
        
        
        
        available_couplings=[total_coupling_list[coupling_id] for coupling_id in coupling_strategy]
        
        
        coupling=available_couplings[np.mod(i,len(available_couplings))]
        
        ever_click_decided=ever_clicked(current_state,coupling)
        never_click_decided=never_clicked(current_state,coupling)
        
        '''For scenario-based steering (end)'''
        
        
       
    
        coin=random.random()

        
        if (never_click_decided[1]<coin):       
            
            
            steering_scenario+=[coupling_strategy[np.mod(i,len(available_couplings))]]

            
            current_state=ever_click_decided[0]
            
        else:
            current_state=never_click_decided[0]       
            
        
        
        current_glob_fit=glob_fit(current_state)

        global_performance+=[np.log(1-current_glob_fit)]

    

        state_overlaps+=[[down_down_down@current_state@down_down_down,
                          psi_plus@current_state@psi_plus,
                          psi_plusminus@current_state@psi_plusminus,
                          psi_minus@current_state@psi_minus,
                          up_up_down@current_state@up_up_down,
                          up_down_up@current_state@up_down_up,
                          down_up_up@current_state@down_up_up,
                          up_up_up@current_state@up_up_up]]
        
        

        if (current_glob_fit>fit_function_threshold):
            break
    
    if np.mod(j,100)==0:
            print(f'\n run={j}')
    

    print(f'attempting update with {[i]}...')
    
    update(type_of_protocol, [i])
    
    print(f'updated successfully')


print(f'All complete. Average compute time per simulation={(time.time()-stopwatch)/number_of_experiments}')


# In[92]:





