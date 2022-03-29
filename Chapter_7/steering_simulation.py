import numpy as np
from tensor_product_tools import *


'''
Actions on the state: never_clicked action, ever_clicked action
'''


def operator_action(state, operator):
    
    '''
    Inputs a (generally non-unitary) operator and a starting state, outputs the resulting state and the probability to arrive there
    
    Global variables: none
    '''
    
#     For pure states
    
    if settings.pure_state_simulation:
    
        new_state=operator@state
        norm=np.linalg.norm(new_state)
        probability=norm**2

        if probability!=0:
            new_state=new_state/norm
        
        
        
        
#     For mixed state simulation

    else:

        new_state=operator@state@operator.T
        probability=np.trace(new_state)

        if probability!=0:
            new_state=new_state/probability


        
    return(new_state,probability)






'''
RDM and distance measures
'''

def RDM(rho,i,j):
    
    '''
    Returns the RDM on spins i, j
    
    global variables: system_size
    '''
    
    lis=list(range(settings.system_size))
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
    
    return((settings.target_rho@np.outer(state,state)).trace() )

# '''
# Same glob_fit but now with DMs. 
# '''

# def glob_fit(state):
#     return((target_rho@state).trace() )



def loc_fit_partitioned(state, partition_location):
    '''
    partition_location: integer from 0 to system_size-1, explaining where the partition starts;
    
    assumes periodic boundary conditions
    
    Global variables: target_rho, system_size
    '''
    
    state_rdm=RDM(np.outer(state,state), partition_location, np.mod(partition_location+1,settings.system_size))
    target_rho_rdm=RDM(settings.target_rho, partition_location, np.mod(partition_location+1,settings.system_size))
    fit=-dist(state_rdm,target_rho_rdm)
    
    return(fit)

# def loc_fit(state):
    
#     '''
#     The integrated local-partitioned fit with the target many-body rho state
    
#     assumes periodic boundary conditions
    
#     Global variables: target_rho, system_size
#     '''
    
#     return(sum([loc_fit_partitioned(state, partition_location) for partition_location in range(system_size)]) )

def loc_fit(state):
    
    fit=0
    for i in range(settings.system_size):
        state_rdm=RDM(np.outer(state,state), i, np.mod(i+1,settings.system_size))
        target_rho_rdm=RDM(settings.target_rho, i, np.mod(i+1,settings.system_size))
        fit-=dist(state_rdm,target_rho_rdm)
    
    return(fit)



'''
Expected benefits, benefit-based decider, blind decider
'''


# def expected_global_fit(ever_click,never_click):
    
#     '''
    
#     '''
    
#     click_benefit=glob_fit(ever_click[i][0])*ever_click[i][1]
#     noclick_benefit=glob_fit(never_click[i][0])*never_click[i][1]
#     output=click_benefit+noclick_benefit
        
#     return(output)


# def expected_global_fits(ever_click,never_click, coupling_list):
#     output=[]
#     for i in range(len(coupling_list)):
#         click_benefit=glob_fit(ever_click[i][0])*ever_click[i][1]
#         noclick_benefit=glob_fit(never_click[i][0])*never_click[i][1]
#         output+=[click_benefit+noclick_benefit]
        
#     return(output)


# def fresh_coupling(recent_history, indices):
    
#     options=[x for x in indices if x not in recent_history]
#     certified_fresh=False
    
#     if options!=[]:
#         certified_fresh=True
#         coupling=random.choice(options)
#     else:
#         coupling=random.choice(indices)
    
    
#     return(coupling, certified_fresh, len(options))


# def benefit_based_decider(history,benefits):
    
#     recent_history=history[-(settings.system_size-1):]
    
    
#     indices=[ind for ind, val in enumerate(benefits) if val==max(benefits)]

    
# #     output, certified_fresh, fresh_options=fresh_coupling(recent_history,indices)

#     output=random.choice(indices)
        
#     return (output)

# def blind_decider(history, couplings):
    
#     indices=list(range(len(couplings) ) )
    
#     recent_history=history[-(settings.system_size-1):]
    
#     output, certified_fresh, fresh_options=fresh_coupling(recent_history,indices)

# #     output=random.choice(indices)
        
#     return (output)