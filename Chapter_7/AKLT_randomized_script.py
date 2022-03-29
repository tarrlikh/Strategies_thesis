import numpy as np
import random
from scipy import linalg
from scipy.optimize import minimize
from itertools import product
import json
import os
import time
import sys
from string import *

import matplotlib.pyplot as plt


#System size: a crucial step influencing the rest of the code

system_size=eval(sys.argv[1])

type_of_protocol=f'AKLT_{sys.argv[1]}_randomcoupling_durations'

number_of_experiments=eval(sys.argv[2])

slur_id=sys.argv[3]

iterations=1000


size=system_size
spin_dim=3
duration_datasets=dict()


def load(name_of_the_dataset):
    
    
    with open('data/'+name_of_the_dataset+f'/{slur_id}.json', 'r') as file:
        duration_datasets[name_of_the_dataset]=json.load(file)

def clear(name_of_the_dataset):
    if os.path.isfile('data/'+name_of_the_dataset+'.json'):
        os.remove('data/'+name_of_the_dataset+'.json')
        
    if name_of_the_dataset in duration_datasets:
        del duration_datasets[name_of_the_dataset]
        
def update(name_of_the_dataset, additional_durations):
    
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
    
def kron_to_tensor(matrix_operator, size=system_size):
    
    index_ranges=tuple(np.ones(2*size,int)*spin_dim)
    tensor_operator=matrix_operator.reshape(index_ranges)
    
    
    return(tensor_operator)

def tensor_to_kron(tensor_operator, size=system_size):
    
    index_ranges=(spin_dim**size,spin_dim**size)
    matrix_operator=tensor_operator.reshape(index_ranges)
    
    
    return(tensor_operator)


def permute (operator, new_label_list, size=system_size):
    
    operator_tensor=kron_to_tensor(operator)
    
    shuffle=[new_label_list+[label+size for label in new_label_list]]
    
    operator_tensor_transposed=operator_tensor.transpose(shuffle)
    
    kron_shape=(spin_dim**size, spin_dim**size)
    
    operator_permuted=operator_tensor_transposed.reshape(kron_shape)
    
    return(operator_permuted)
    
def partial_trace(matrix_operator, spins_to_traceout, size=system_size):
    tensor_operator=kron_to_tensor(matrix_operator, size=size)
    spins_ids=list(range(size))
    spins_to_remain=[spin_id for spin_id in spins_ids if spin_id not in spins_to_traceout]
    shuffle=tuple(spins_to_traceout+[spin_id+size for spin_id in spins_to_traceout]+spins_to_remain+[spin_id+size for spin_id in spins_to_remain])
    reshuffled_tensor=tensor_operator.transpose(shuffle)
    bipartite_shape=(spin_dim**len(spins_to_traceout),spin_dim**len(spins_to_traceout),spin_dim**(len(spins_to_remain)),spin_dim**(len(spins_to_remain)))
    bipartite_operator=reshuffled_tensor.reshape(bipartite_shape)
    return(np.trace(bipartite_operator))

def chained_kron(list_of_matrices):
    output=list_of_matrices[0]
    for matrix in list_of_matrices[1:]:
        output=np.kron(output,matrix)
    return(output) 

'''
Creating many-body operators from few-body terms
'''

def simplify(operator,size):
    return (operator.ptrace( list( range(1,size+1) ) ))


def permutation (i, j, size):
    
    output=list(range(size))
    output[i]=j
    output[j]=i
    return(output)


def kron_permute(kron_operator, new_spins_list, size=system_size):
    
    tensor_operator=kron_to_tensor(kron_operator, size)
    
#     print(f"tensor_op_shape={tensor_operator.shape}, new_spins_list={new_spins_list}")
    
    shuffle=tuple(new_spins_list+[spin_label+size for spin_label in new_spins_list])
    
    tensor_operator_permuted=tensor_operator.transpose(shuffle)
    
    kron_operator_permuted=tensor_to_kron(tensor_operator_permuted, size)
    
    return(kron_operator_permuted)
    

def manybody_term_new(term, term_labels):
    
    mbterm=np.kron(manybody_identity, term)
    
    old_spins_list=list(range(system_size+len(term_labels) ))
    
    new_spins_list=[]
    n=0
    
    disposable_term_labels=term_labels
    
    for system_label in range(system_size):
        
        nontrivial_label=False
        
        for term_label_number in range(len(disposable_term_labels)):
            if disposable_term_labels[term_label_number]==system_label:
                new_spins_list+=[system_size+term_label_number]
                nontrivial_label=True
        if not nontrivial_label:
            new_spins_list+=[system_label]
            
    new_spins_list+=term_labels
    
    mbterm=kron_permute(mbterm,new_spins_list, size=system_size+len(term_labels))
    
    spins_to_traceout=list(range(system_size,system_size+len(term_labels)))
    
    mbterm=partial_trace(mbterm, spins_to_traceout, size=system_size+len(term_labels))/(spin_dim**len(term_labels))
    
    
    return(mbterm)
    
    

    
def manybody_term(term, i, j):
    
    mbterm=tensor(manybody_identity,term)
    
    
    
    
    mbterm=mbterm.permute(permutation (system_size, i, system_size+2))
    mbterm=mbterm.permute(permutation (system_size+1, j, system_size+2))
    mbterm=mbterm.ptrace( list( range(system_size) ) )/9
    
    return(mbterm)


    
def AKLT(system_size):
    return(sum([manybody_term_new(AKLT_term, [i, np.mod(i+1,system_size)]) for i in range(system_size)] ) )


'''
Actions on the state: never_clicked action, ever_clicked action
'''

def never_clicked(state, coupling):
    return(operator_action(state, coupling[0]))

def ever_clicked(state, coupling):        
    return(operator_action(state, coupling[1]))

def operator_action(state, operator):
    
    new_state=operator@state
    norm=np.linalg.norm(new_state)
    probability=norm**2
    
    if probability!=0:
        new_state=new_state/norm
        
    return(new_state,probability)

# '''
# Same op action but now with DMs. 
# '''

# def operator_action(state, operator):
    
#     new_state=operator@state@operator.T
#     probability=np.trace(new_state)
    
#     if probability!=0:
#         new_state=new_state/probability
        
#     return(new_state,probability)


'''
RDM and distance measures
'''

def RDM(rho,i,j):
    lis=list(range(system_size))
    lis.remove(i)
    lis.remove(j)
    return(partial_trace(rho,lis))

def dist(dm1,dm2):
    return(((dm1-dm2)@(dm1-dm2)).trace())

def glob_fit(state):
    return((mb_rho@np.outer(state,state)).trace() )

# '''
# Same glob_fit but now with DMs. 
# '''

# def glob_fit(state):
#     return((mb_rho@state).trace() )

def loc_fit(state):
    
    fit=0
    for i in range(system_size):
        state_rdm=RDM(np.outer(state,state), i, np.mod(i+1,system_size))
        mb_rho_rdm=RDM(mb_rho, i, np.mod(i+1,system_size))
        fit-=dist(state_rdm,mb_rho_rdm)
    
    return(fit)


def loc_fit_partitioned(state, partition_location):
    '''
    partition_location: integer from 0 to system_size-1, explaining where the partition starts
    '''
    
    state_rdm=RDM(np.outer(state,state), partition_location, np.mod(partition_location+1,system_size))
    mb_rho_rdm=RDM(mb_rho, partition_location, np.mod(partition_location+1,system_size))
    fit-=dist(state_rdm,mb_rho_rdm)
    
    return(fit)

'''
Expected benefits, benefit-based decider, blind decider
'''





# def expected_local_fits(ever_click,never_click, coupling_list):
#     output=[]
#     for i in range(len(coupling_list)):
#         click_benefit=loc_fit(ever_click[i][0])*ever_click[i][1]
#         noclick_benefit=loc_fit(never_click[i][0])*never_click[i][1]
#         output+=[click_benefit+noclick_benefit]        
#     return(output)

# def maximal_local_fits(ever_click,never_click, coupling_list, current_state):
#     output=[]
#     for i in range(len(coupling_list)):
#         click_benefit=(loc_fit(ever_click[i][0])-loc_fit(current_state))*ever_click[i][1]
#         noclick_benefit=(loc_fit(never_click[i][0])-loc_fit(current_state))*never_click[i][1]
#         output+=[max(click_benefit,noclick_benefit)]        
#     return(output)

# def maximal_local_risks(ever_click,never_click, coupling_list, current_state):
#     output=[]
#     for i in range(len(coupling_list)):
#         click_benefit=(loc_fit(ever_click[i][0])-loc_fit(current_state))*ever_click[i][1]
#         noclick_benefit=(loc_fit(never_click[i][0])-loc_fit(current_state))*never_click[i][1]
#         output+=[min(click_benefit,noclick_benefit)]        
#     return(output)

def expected_global_fit(ever_click,never_click):
    
    click_benefit=glob_fit(ever_click[i][0])*ever_click[i][1]
    noclick_benefit=glob_fit(never_click[i][0])*never_click[i][1]
    output=click_benefit+noclick_benefit
        
    return(output)


def expected_global_fits(ever_click,never_click, coupling_list):
    output=[]
    for i in range(len(coupling_list)):
        click_benefit=glob_fit(ever_click[i][0])*ever_click[i][1]
        noclick_benefit=glob_fit(never_click[i][0])*never_click[i][1]
        output+=[click_benefit+noclick_benefit]
        
    return(output)


# def maximal_global_fits(ever_click,never_click, coupling_list, current_state):
#     output=[]
#     for i in range(len(coupling_list)):
#         click_benefit=(glob_fit(ever_click[i][0])-glob_fit(current_state))*ever_click[i][1]
#         noclick_benefit=(glob_fit(never_click[i][0])-glob_fit(current_state))*never_click[i][1]
#         output+=[max(click_benefit,noclick_benefit)]
        
#     return(output)

# def maximal_global_risks(ever_click,never_click, coupling_list):
#     output=[]
#     for i in range(len(coupling_list)):
#         click_benefit=(glob_fit(ever_click[i][0])-glob_fit(current_state))*ever_click[i][1]
#         noclick_benefit=(glob_fit(never_click[i][0])-glob_fit(current_state))*never_click[i][1]
#         output+=[min(click_benefit,noclick_benefit)]
        
#     return(output)

def fresh_coupling(recent_history, indices):
    
    options=[x for x in indices if x not in recent_history]
    certified_fresh=False
    
    if options!=[]:
        certified_fresh=True
        coupling=random.choice(options)
    else:
        coupling=random.choice(indices)
    
    
    return(coupling, certified_fresh, len(options))

# def risk_based_decider(history,risks):
    
#     recent_history=history[-(system_size-1):]
    
    
#     indices=[ind for ind, val in enumerate(risks) if val==min(risks)]

    
# #     output, certified_fresh, fresh_options=fresh_coupling(recent_history,indices)

#     output=random.choice(indices)
        
#     return (output)

def benefit_based_decider(history,benefits):
    
    recent_history=history[-(system_size-1):]
    
    
    indices=[ind for ind, val in enumerate(benefits) if val==max(benefits)]

    
#     output, certified_fresh, fresh_options=fresh_coupling(recent_history,indices)

    output=random.choice(indices)
        
    return (output)

def blind_decider(history, couplings):
    
    indices=list(range(len(couplings) ) )
    
    recent_history=history[-(system_size-1):]
    
    output, certified_fresh, fresh_options=fresh_coupling(recent_history,indices)

#     output=random.choice(indices)
        
    return (output)
    
'''
Local states & operators
'''


#1-spin operators
S0=np.array([[1,0,0],[0,1,0],[0,0,1]])
S1=(1/np.sqrt(2))*np.array([[0,1,0],[1,0,1],[0,1,0]])
S2=(1/np.sqrt(2))*np.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]])
S3=np.array([[1,0,0],[0,0,0],[0,0,-1]])

#2-spin operators
Heis_term=np.kron(S1,S1)+np.kron(S2,S2)+np.kron(S3,S3)
identity_term=np.kron(S0,S0)
AKLT_term=identity_term/3.+Heis_term/2+Heis_term@Heis_term/6



#2-spin states
psi=np.linalg.eigh(AKLT_term)[1].T[0:4]
phi=np.linalg.eigh(AKLT_term)[1].T[4:9]

#2-spin coupling
V=np.outer(psi[0],phi[0])+np.outer(psi[1],phi[1])+np.outer(psi[2],phi[2])+np.outer(psi[3],phi[3])+np.outer(phi[3],phi[4])
P=np.outer(psi[0],psi[0])+np.outer(psi[1],psi[1])+np.outer(psi[2],psi[2])+np.outer(psi[3],psi[3])

V1=np.outer(psi[0],phi[0])+np.outer(psi[1],phi[1])+np.outer(psi[2],phi[2])+np.outer(phi[3],phi[4])
P1=np.outer(psi[0],psi[0])+np.outer(psi[1],psi[1])+np.outer(psi[2],psi[2])+np.outer(phi[3],phi[3])

V2=np.outer(psi[1],phi[1])+np.outer(psi[2],phi[2])+np.outer(psi[3],phi[3])+np.outer(phi[3],phi[4])
P2=np.outer(phi[0],phi[0])+np.outer(psi[1],psi[1])+np.outer(psi[2],psi[2])+np.outer(psi[3],psi[3])


def mb_V_func(a,coupling_location):
    '''
    a - antisym matrix elements, defining a unitary rotation from phi to psi
    phsi_coef - 4 x 4 matrix, phi->psi
    '''
    
    phsi_coef=linalg.expm(antisym_fourdim(a))
    
    V=np.outer(phi[3],phi[4])
    
    for i in range(4):
        for j in range(4):
            V+=phsi_coef[i,j]*np.outer(psi[i],phi[j])
    
    
#     V_phi_terms=[phi_coef[i]*np.outer(phi[i],phi[4]) for i in range(4)]
#     V_phi=sum(V_phi_terms)
    

    
    mb_V=manybody_term_new(V,[coupling_location,np.mod(coupling_location+1,system_size)])
    
    return(mb_V)


def antisym_fourdim(a):    
    A=np.array([[0,a[0],a[1],a[2]],[-a[0],0,a[3],a[4]],[-a[1],-a[3],0,a[5]],[-a[2],-a[4],-a[5],0]])
    return(A)


'''
Many-body stuff: system size, states, operators
'''


#Many-body identity

manybody_identity=chained_kron([S0 for i in range(system_size)])

#Many-body states
triv_state=manybody_identity[0]

mb_psi=(np.linalg.eigh(AKLT(system_size))[1].T)[0]
#(AKLT(size).eigenstates()[1])[0:4] - if 4-fold degenerate

mb_rho=np.outer(mb_psi,mb_psi)
# psi[0]*psi[0].dag()+psi[1]*psi[1].dag()+psi[2]*psi[2].dag()+psi[3]*psi[3].dag() - if degenerate

#Coupling lists
mb_P_list=[manybody_term_new(P,[coupling_location,np.mod(coupling_location+1,system_size)]) for coupling_location in range(system_size)]
mb_V_list=[manybody_term_new(V,[coupling_location,np.mod(coupling_location+1,system_size)]) for coupling_location in range(system_size)]

# mb_P1_list=[manybody_term_new(P1,[coupling_location,np.mod(coupling_location+1,system_size)]) for coupling_location in range(system_size)]
# mb_V1_list=[manybody_term_new(V1,[coupling_location,np.mod(coupling_location+1,system_size)]) for coupling_location in range(system_size)]

# mb_P2_list=[manybody_term_new(P2,[coupling_location,np.mod(coupling_location+1,system_size)]) for coupling_location in range(system_size)]
# mb_V2_list=[manybody_term_new(V2,[coupling_location,np.mod(coupling_location+1,system_size)]) for coupling_location in range(system_size)]

total_coupling_list=[[mb_P_list[i],mb_V_list[i]] for i in range(len(mb_P_list)) ]

# total_coupling_list_12=[[ [mb_P1_list[i],mb_V1_list[i] ], [mb_P2_list[i],mb_V2_list[i] ] ] for i in range(len(mb_P_list)) ]

# total_coupling_list_12= [item for sublist in total_coupling_list_12 for item in sublist]

starting_state=triv_state
# starting_state=(np.outer(up_up,up_up)+np.outer(down_down,down_down)+np.outer(phi_plus,phi_plus)+np.outer(phi_minus,phi_minus))/4.
# starting_state=np.outer(down_down_down,down_down_down)
global_feedback_durations=[]
local_feedback_durations=[]
blind_durations=[]
coupling_alternatives_active_durations=[]
coupling_alternatives_passive_durations=[]




global_performance=[]
local_performance=[]
durations=[]



stopwatch=time.time()

for j in range(number_of_experiments):
    
    current_state=starting_state
    
#     current_loc_fit=loc_fit(current_state)
    current_glob_fit=glob_fit(current_state)
    
    steering_scenario=[]
    
    coupling_history=[]
    
    
    state_overlaps=[]
    amount_of_clicks=[0]
    
    for i in range(iterations):
        
        coupling_locator=np.mod(i,system_size)
        
        '''For continuous optimization steering (start)'''
#         available_couplings=[total_coupling_list[np.mod(i,system_size)]]
        '''For continuous optimization steering (end)'''
        
        '''For scenario-based steering (start):'''
#         coupling_strategy=coupling_strategy_function(steering_scenario, strategies_W_state_blind)
#         available_couplings=[total_coupling_list[coupling_id] for coupling_id in coupling_strategy]
        
        
#         coupling=available_couplings[np.mod(i,len(available_couplings))]
        
#         ever_click_decided=ever_clicked(current_state,coupling)
#         never_click_decided=never_clicked(current_state,coupling)
        
        '''For scenario-based steering (end)'''
        
        
        '''For benefit-based local-choice steering (start):'''
#         available_couplings=total_coupling_list


        
#         ever_click=[ever_clicked(current_state,coupling) for coupling in available_couplings]        
#         never_click=[never_clicked(current_state,coupling) for coupling in available_couplings]

#         global_progress=False
#         local_progress=False
        
        
#         local_benefits=expected_local_fits(ever_click,never_click, available_couplings)

#         if len(local_performance)>1:
#             local_progress=(max(local_benefits)>0.99*current_loc_fit)
#     #         print(f"exp local change: {loc_fit(new_state,system_size)}->{max(local_benefits)}")
#         else:
#             local_progress=True
        

#         global_benefits=expected_global_fits(ever_click,never_click, available_couplings)

#         if len(global_performance)>1:
#             global_progress=(max(global_benefits)>1.01*current_glob_fit)
#     #         print(f"exp global change: {glob_fit(new_state,system_size)}->{max(global_benefits)}")
#         else:
#             global_progress=True       



#         if global_progress:
#             coupling_id = risk_based_decider(coupling_history, global_max_risks)
# #             coupling_id = benefit_based_decider(coupling_history, global_benefits)
#         else:
#             if local_progress:
#                 coupling_id = benefit_based_decider(coupling_history, local_benefits)
#             else:
#                 coupling_id = blind_decider(coupling_history,available_couplings)
        
#         never_click_decided=never_click[coupling_id]
#         ever_click_decided=ever_click[coupling_id]

        '''For benefit-based local-choice steering (end)'''
        
    
        '''(start) continuous optimization steering'''
        
        never_click_decided=operator_action(current_state, mb_P_list[coupling_locator])
        
#         def global_fit_to_optimize(a):                    
#             return(-np.real(glob_fit(operator_action(current_state, mb_V_func(a, coupling_locator))[0])))
        
#         a=minimize(global_fit_to_optimize, np.zeros(6), method='SLSQP', options={'maxiter':2}).x     
    
#         ever_click_decided=operator_action(current_state, mb_V_func(a, coupling_locator))

        '''(end) continuous optimization steering'''
    
        coin=random.random()

        
        if (never_click_decided[1]<coin):       
            
            '''(start) continuous optimization steering'''
    
#             def global_fit_to_optimize(a):                    
#                 return(-np.real(glob_fit(operator_action(current_state, mb_V_func(a, coupling_locator))[0])))

#             a=minimize(global_fit_to_optimize, np.zeros(6), method='SLSQP', options={'maxiter':10}).x     
            
            a=np.array([random.random() for parameter_label in range(6)])
        
            ever_click_decided=operator_action(current_state, mb_V_func(a, coupling_locator))
            
#             TO COMPARE, here's a version without continuous optimization

#             ever_click_decided=operator_action(current_state, mb_V_list[coupling_locator])
            
            '''(end) continuous optimization steering'''
            
          
            
            '''discrete strategy steering (start)'''
            
#             steering_scenario+=[coupling_strategy[np.mod(i,len(available_couplings))]]
            
            '''discrete strategy steering (end)'''
            
            amount_of_clicks+=[amount_of_clicks[-1]+1]
            current_state=ever_click_decided[0]
            
        else:
            current_state=never_click_decided[0]        
            amount_of_clicks+=[amount_of_clicks[-1]]
            
#         current_loc_fit=loc_fit(current_state)
        
        
        current_glob_fit=glob_fit(current_state)

#         local_performance+=[np.log(-current_loc_fit)]
        global_performance+=[np.log(1-current_glob_fit)]

    
#         For W-state scenario-based optimization:

#         state_overlaps+=[[down_down_down@current_state@down_down_down,
#                           psi_plus@current_state@psi_plus,
#                           psi_plusminus@current_state@psi_plusminus,
#                           psi_minus@current_state@psi_minus,
#                           up_up_down@current_state@up_up_down,
#                           up_down_up@current_state@up_down_up,
#                           down_up_up@current_state@down_up_up,
#                           up_up_up@current_state@up_up_up]]
        
#         coupling_history+=[coupling_id]
        

        if (current_glob_fit>0.5):
            break
    
    durations+=[i]
#     if i>9000:
#         break
    
    if np.mod(j,10)==0:
            print(f'\n run={j}')
    
    update(type_of_protocol, [i])
    
plt.plot(state_overlaps)

# plt.plot(global_performance)

# plt.plot(local_performance)    

print(f'AKLT steering for {system_size} spins, in total {number_of_experiments} simulations')

print(f'computation time per simulation={(time.time()-stopwatch)/number_of_experiments}')

