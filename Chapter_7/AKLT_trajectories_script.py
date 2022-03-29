import numpy as np
import random
from scipy import linalg
from scipy.optimize import minimize
from itertools import product
from functools import *
import json
import os
import time
import sys
from string import *




import matplotlib.pyplot as plt


#System size: a crucial step influencing the rest of the code

slur_id=sys.argv[1]

system_size=5


number_of_experiments=10


iterations=10000

fit_function_threshold=0.99

size=system_size
spin_dim=3

print(f'AKLT steering for {system_size} spins, in total {number_of_experiments} simulations')

    
#############################
# Tensor/kronecker products #
#############################    
    
    
def kron_to_tensor(matrix_operator, size=system_size):
    '''
    Turns a kronecker product into tensor product
    
    Uses global variables system_size, spin_dim
    '''
    
    
    
    index_ranges=tuple(np.ones(2*size,int)*spin_dim)
    tensor_operator=matrix_operator.reshape(index_ranges)
    
    
    return(tensor_operator)

def tensor_to_kron(tensor_operator, size=system_size):
    '''
    Turns a tensor product into kronecker product
    
    Uses global variables system_size, spin_dim
    '''
    
    
    index_ranges=(spin_dim**size,spin_dim**size)
    matrix_operator=tensor_operator.reshape(index_ranges)
    
    
    return(tensor_operator)



    
def partial_trace(matrix_operator, spins_to_traceout, size=system_size):
    
    '''
    Traceout the spins with labels spins_to_traceout
    
    Uses global variables system_size, spin_dim
    '''
    
    tensor_operator=kron_to_tensor(matrix_operator, size=size)
    
    spins_ids=list(range(size))
    
    spins_to_remain=[spin_id for spin_id in spins_ids if spin_id not in spins_to_traceout]
    
    shuffle=tuple(spins_to_traceout+[spin_id+size for spin_id in spins_to_traceout]+spins_to_remain+[spin_id+size for spin_id in spins_to_remain])
    
    reshuffled_tensor=tensor_operator.transpose(shuffle)
    
    bipartite_shape=(spin_dim**len(spins_to_traceout),spin_dim**len(spins_to_traceout),spin_dim**(len(spins_to_remain)),spin_dim**(len(spins_to_remain)))
    
    bipartite_operator=reshuffled_tensor.reshape(bipartite_shape)
    
    return(np.trace(bipartite_operator))

def chained_kron(list_of_matrices):
    '''
    Returns a kronecker product of a list of matrices, from 1 to last
    
    Uses no global variables
    '''
    return(reduce(np.kron, list_of_matrices)) 

####################################################
# Creating many-body operators from few-body terms #
####################################################


def permutation (i, j, size):
    '''
    Returns a range(size) with j and i permuted 
    
    Uses no global variables
    '''
    
    
    output=list(range(size))
    output[i]=j
    output[j]=i
    return(output)


def kron_permute(kron_operator, new_spins_list, size=system_size):
    
    '''
    Inputs a kronecker product operator and a relabelling of spins, outputs a kronecker product operator with spins relabelled 
    
    
    Uses global variables sysem_size
    '''
    
    tensor_operator=kron_to_tensor(kron_operator, size)
    
#     print(f"tensor_op_shape={tensor_operator.shape}, new_spins_list={new_spins_list}")
    
    shuffle=tuple(new_spins_list+[spin_label+size for spin_label in new_spins_list])
    
    tensor_operator_permuted=tensor_operator.transpose(shuffle)
    
    kron_operator_permuted=tensor_to_kron(tensor_operator_permuted, size)
    
    return(kron_operator_permuted)
    

def manybody_term(term, term_labels):
    
    
    '''
    Inserts a k-spin term into a k-spin slot term_labels (which lists the participating spin labels)
    
    Global variables: system_size, spin_dim
    '''
    
    mbterm=np.kron(manybody_identity, term)    
    
    disposable_term_labels=term_labels
    
    new_spins_list=[system_size+term_labels.index(spin_label) if spin_label in term_labels else spin_label for spin_label in range(system_size)]
    
    new_spins_list+=term_labels   
    
    mbterm=kron_permute(mbterm,new_spins_list, size=system_size+len(term_labels))
    
    spins_to_traceout=list(range(system_size,system_size+len(term_labels)))
    
    mbterm=partial_trace(mbterm, spins_to_traceout, size=system_size+len(term_labels))/(spin_dim**len(term_labels))
    
    
    return(mbterm)
    
    




'''
Actions on the state: never_clicked action, ever_clicked action
'''


def operator_action(state, operator):
    
    '''
    Inputs a (generally non-unitary) operator and a starting state, outputs the resulting state and the probability to arrive there
    
    Global variables: none
    '''
    
    new_state=operator@state
    norm=np.linalg.norm(new_state)
    probability=norm**2
    
    if probability!=0:
        new_state=new_state/norm
        
    return(new_state,probability)

def never_clicked(state, coupling):
    
    '''
    Assuming that coupling[0] is a never-click operator, act on the state with that
    '''
    
    
    return(operator_action(state, coupling[0]))

def ever_clicked(state, coupling):
    '''
    Assuming that coupling[0] is a click operator, act on the state with that
    '''
    
    return(operator_action(state, coupling[1]))



'''
RDM and distance measures
'''

def RDM(rho,i,j):
    
    '''
    Returns the RDM on spins i, j
    
    global variables: system_size
    '''
    
    lis=list(range(system_size))
    lis.remove(i)
    lis.remove(j)
    
    return(partial_trace(rho,lis))

def dist(dm1,dm2):
    
    '''
    Trace-distance between two density matrices
    '''
    
    return(((dm1-dm2)@(dm1-dm2)).trace())

def glob_fit(state):
    
    '''
    The global fit with the target many-body rho state
    
    Global variables: target_rho
    '''
    
    return((target_rho@np.outer(state,state)).trace() )



def loc_fit_partitioned(state, partition_location):
    '''
    partition_location: integer from 0 to system_size-1, explaining where the partition starts;
    
    assumes periodic boundary conditions
    
    Global variables: target_rho, system_size
    '''
    
    state_rdm=RDM(np.outer(state,state), partition_location, np.mod(partition_location+1,system_size))
    target_rho_rdm=RDM(target_rho, partition_location, np.mod(partition_location+1,system_size))
    fit=-dist(state_rdm,target_rho_rdm)
    
    return(fit)


def loc_fit(state):
    
    fit=0
    for i in range(system_size):
        state_rdm=RDM(np.outer(state,state), i, np.mod(i+1,system_size))
        target_rho_rdm=RDM(target_rho, i, np.mod(i+1,system_size))
        fit-=dist(state_rdm,target_rho_rdm)
    
    return(fit)




    
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


def AKLT(system_size):
    return(sum([manybody_term(AKLT_term, [i, np.mod(i+1,system_size)]) for i in range(system_size)] ) )


#2-spin states
psi=np.linalg.eigh(AKLT_term)[1].T[0:4]
phi=np.linalg.eigh(AKLT_term)[1].T[4:9]

#2-spin coupling
V=np.outer(psi[0],phi[0])+np.outer(psi[1],phi[1])+np.outer(psi[2],phi[2])+np.outer(psi[3],phi[3])+np.outer(phi[3],phi[4])
P=np.outer(psi[0],psi[0])+np.outer(psi[1],psi[1])+np.outer(psi[2],psi[2])+np.outer(psi[3],psi[3])


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
    
    

    

    
    mb_V=manybody_term(V,[coupling_location,np.mod(coupling_location+1,system_size)])
    
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

target_psi=(np.linalg.eigh(AKLT(system_size))[1].T)[0]
#(AKLT(size).eigenstates()[1])[0:4] - if 4-fold degenerate

target_rho=np.outer(target_psi,target_psi)
# psi[0]*psi[0].dag()+psi[1]*psi[1].dag()+psi[2]*psi[2].dag()+psi[3]*psi[3].dag() - if degenerate

#Coupling lists
mb_P_list=[manybody_term(P,[coupling_location,np.mod(coupling_location+1,system_size)]) for coupling_location in range(system_size)]
mb_V_list=[manybody_term(V,[coupling_location,np.mod(coupling_location+1,system_size)]) for coupling_location in range(system_size)]


total_coupling_list=[[mb_P_list[i],mb_V_list[i]] for i in range(len(mb_P_list)) ]

starting_state=triv_state
# starting_state=(np.outer(up_up,up_up)+np.outer(down_down,down_down)+np.outer(phi_plus,phi_plus)+np.outer(phi_minus,phi_minus))/4.
# starting_state=np.outer(down_down_down,down_down_down)



local_performance=[]



stopwatch=time.time()

for j in range(number_of_experiments):
    
    current_state=starting_state
    
#     current_loc_fit=loc_fit(current_state)
    current_glob_fit=glob_fit(current_state)
    
    steering_scenario=[]
    
    coupling_history=[]
    
    global_performance=[]
    
    state_overlaps=[]
    amount_of_clicks=[0]
    
    for i in range(iterations):
        
        if i>210:
            break
        
        coupling_locator=np.mod(i,system_size)
        
        
        
           
        available_couplings=[total_coupling_list[np.mod(i,system_size)]]
        
        never_click_decided=operator_action(current_state, mb_P_list[coupling_locator])

    
        coin=random.random()

        
        if (never_click_decided[1]<coin):       
            
            '''(start) continuous optimization steering'''
    
            def global_fit_to_optimize(a):                    
                return(-np.real(glob_fit(operator_action(current_state, mb_V_func(a, coupling_locator))[0])))            
            
#             def local_fit_to_optimize(a):                    
#                 return(-np.real(loc_fit(operator_action(current_state, mb_V_func(a, coupling_locator))[0])))

#             a=minimize(global_fit_to_optimize, np.zeros(6), method='SLSQP', options={'maxiter':10}).x     
            
            a=np.array([random.random() for parameter_label in range(6)])
        
            ever_click_decided=operator_action(current_state, mb_V_func(a, coupling_locator))
            

            
            '''(end) continuous optimization steering'''
            
            
            amount_of_clicks+=[amount_of_clicks[-1]+1]
            current_state=ever_click_decided[0]
            
        else:
            current_state=never_click_decided[0]        
            amount_of_clicks+=[amount_of_clicks[-1]]
            
#         current_loc_fit=loc_fit(current_state)
        
        
        current_glob_fit=glob_fit(current_state)

#         local_performance+=[np.log(-current_loc_fit)]
        global_performance+=[np.real(1-current_glob_fit)]

    

        

        if (current_glob_fit>fit_function_threshold):
            break
    
    
    if np.mod(j,10)==0:
            print(f'\n run={j}')
    
    
    output_file_name=f'trajectories_AKLT/{slur_id}.json'
    
    
    
    
    if os.path.isfile(output_file_name):
        
      
    
        with open(output_file_name, 'r') as fl:

            stored_trajectories=json.load(fl)


        with open(output_file_name, 'w') as fl:

            json.dump(stored_trajectories+[global_performance], fl)
            
    else:
        
        
        with open(output_file_name, 'w') as fl:

            json.dump([global_performance], fl)
 

