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
import ecumene as ecu
from string import *

import matplotlib.pyplot as plt









##########################

MDN=ecu.Ecumene('MDN', 'load')

##########################

time_stamp=eval(sys.argv[1])

number_of_experiments=eval(sys.argv[2])

slur_id=sys.argv[3]



##########################


sim_filename=f'simulation_history/{time_stamp}_sim'

postfix=f'{time_stamp}_{slur_id}'

with open(sim_filename, 'r') as sfl:
    sim_id=json.loads(json.load(sfl))

  

fit_function_threshold=1-sim_id['target_accuracy']
system_size, target, strategy_type, datatype=[sim_id[key] for key in ['system_size', 'target', 'strategy_type', 'datatype']]

size=system_size

if target=='AKLT':
    spin_dim=3
    purity=True
elif target=='W':
    spin_dim=2
    purity=False


# if target=='W':
    
#     raise Exception ('This is only for AKLT!')
    
# fixed constant, just for fool-proofness
    
iterations=100000

###########################





















    
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
    
    if purity:
    
        new_state=operator@state
        norm=np.linalg.norm(new_state)
        probability=norm**2

        if probability!=0:
            new_state=new_state/norm

        return(new_state,probability)

    else:
        
        new_state=operator@state@operator.T
        probability=np.trace(new_state)

        if probability!=0:
            new_state=new_state/probability

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
    if purity:
        return((target_rho@np.outer(state,state)).trace() )
    else:
        return((target_rho@state).trace() )


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













if target=='AKLT':

    '''
    States & operators for AKLT
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


    #Many-body identity

    manybody_identity=chained_kron([S0 for i in range(system_size)])


    target_psi=(np.linalg.eigh(AKLT(system_size))[1].T)[0]
    target_rho=np.outer(target_psi,target_psi)


    #Coupling lists
    mb_P_list=[manybody_term(P,[coupling_location,np.mod(coupling_location+1,system_size)]) for coupling_location in range(system_size)]

    starting_state=manybody_identity[0]

    ###########################


    




    global_performance=[]

    stopwatch=time.time()


    for j in range(number_of_experiments):

        current_state=starting_state

        current_glob_fit=glob_fit(current_state)



        for i in range(iterations):

            coupling_locator=np.mod(i,system_size)

            never_click_decided=operator_action(current_state, mb_P_list[coupling_locator])


            coin=random.random()

            if (never_click_decided[1]<coin):       


                def global_fit_to_optimize(a):                    
                    return(-np.real(glob_fit(operator_action(current_state, mb_V_func(a, coupling_locator))[0])))            

                if strategy_type=='passive':
                    a=np.array([random.random() for parameter_label in range(6)])

                elif strategy_type=='active':
                    a=minimize(global_fit_to_optimize, np.zeros(6), method='SLSQP', options={'maxiter':10}).x     


                ever_click_decided=operator_action(current_state, mb_V_func(a, coupling_locator))

                current_state=ever_click_decided[0]

            else:

                current_state=never_click_decided[0]        


            current_glob_fit=glob_fit(current_state)    


            if (current_glob_fit>fit_function_threshold):
                break

        if np.mod(j,10)==0:
                print(f'\n run={j}')

        if datatype=='durations':
            MDN.store_data([i], sim_id, postfix=postfix)


        print(f'updated successfully')













elif target=='W':




    '''
    States & operators for W
    '''




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

    target_rho=np.outer(psi_plus,psi_plus)




    # In[48]:


    def coupling_strategy_function(scenario, strategies):

        if str(scenario) in strategies[0]:
            coupling_strategy=strategies[0][str(scenario)]
        else:
            coupling_strategy=strategies[1]

        return(coupling_strategy)


    # In[49]:

    if strategy_type=='scenario_based_active':
        strategies=(
            {"[]" : [0],
            "[0]": [2,3]},
        [2,3])
    
    elif strategy_type=='passive':
    
        strategies=({},
        [0,1,2,3])

    starting_state=np.outer(down_down_down,down_down_down)


    durations=[]


    stopwatch=time.time()

    for j in range(number_of_experiments):

        current_state=starting_state

        current_glob_fit=glob_fit(current_state)

        steering_scenario=[]



        state_overlaps=[]

        for i in range(iterations):


            coupling_strategy=coupling_strategy_function(steering_scenario, strategies)



            available_couplings=[total_coupling_list[coupling_id] for coupling_id in coupling_strategy]


            coupling=available_couplings[np.mod(i,len(available_couplings))]

            ever_click_decided=ever_clicked(current_state,coupling)
            never_click_decided=never_clicked(current_state,coupling)

       


            coin=random.random()


            if (never_click_decided[1]<coin):       


                steering_scenario+=[coupling_strategy[np.mod(i,len(available_couplings))]]


                current_state=ever_click_decided[0]

            else:
                current_state=never_click_decided[0]       



            current_glob_fit=glob_fit(current_state)




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
        
        if datatype=='durations':
        
            MDN.store_data([i], sim_id, postfix=postfix)
            
        elif datatype=='state_trajectories':
            
            state_overlaps=np.array(state_overlaps)
            
            subspace_densities=np.array(np.round(np.array([state_overlaps.T[0].T, 
                  np.sum(state_overlaps.T[1:4].T,axis=1),
                  np.sum(state_overlaps.T[4:7].T,axis=1),
                  state_overlaps.T[7].T]).T),dtype=int)

            subspace_trajectory=[list(time_snapshot).index(1) for time_snapshot in subspace_densities]
            
            MDN.store_data([subspace_trajectory], sim_id, postfix=postfix)

        print(f'updated successfully')




print(f'All complete. Computation time per simulation={(time.time()-stopwatch)/number_of_experiments}')

            