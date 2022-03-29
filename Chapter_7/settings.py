# Variable list: system_size, spin_dim, target_rho, pure_state_simulation, type_of_system, type_of_protocol, starting_state

system_size, spin_dim, target_psi, target_rho, pure_state_simulation, type_of_system, type_of_protocol, starting_state_name, starting_state, target_fidelity, duration_cutoff, number_of_experiments = (None for variables in range(12))

from tensor_product_tools import *

import numpy as np
from scipy import linalg

def init():
    
    global pure_state_simulation, duration_cutoff
    
    pure_state_simulation=True
    
    duration_cutoff=1000000

def system_properties():
    
    
#     In this part, we define the spin_dim, the target_psi, target_rho, the starting_state - as a function of the type_of_system and the starting_state_name; we use 
    
    if type_of_system=='AKLT':
        
        global spin_dim, target_psi, target_rho, starting_state, mb_P_list, phi, psi
        
        
        
        spin_dim=3
        
        
        #1-spin operators
        S0=np.array([[1,0,0],[0,1,0],[0,0,1]])
        S1=(1/np.sqrt(2))*np.array([[0,1,0],[1,0,1],[0,1,0]])
        S2=(1/np.sqrt(2))*np.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]])
        S3=np.array([[1,0,0],[0,0,0],[0,0,-1]])

        #2-spin operators
        Heis_term=np.kron(S1,S1)+np.kron(S2,S2)+np.kron(S3,S3)
        identity_term=np.kron(S0,S0)
        AKLT_term=identity_term/3.+Heis_term/2+Heis_term@Heis_term/6


        def AKLT(a_system_size):
            return(sum([manybody_term(AKLT_term, [i, np.mod(i+1,a_system_size)]) for i in range(a_system_size)] ) )
        
        psi=np.linalg.eigh(AKLT_term)[1].T[0:4]
        phi=np.linalg.eigh(AKLT_term)[1].T[4:9]
        
        P=np.outer(psi[0],psi[0])+np.outer(psi[1],psi[1])+np.outer(psi[2],psi[2])+np.outer(psi[3],psi[3])

        
        mb_P_list=[manybody_term(P,[coupling_location,np.mod(coupling_location+1,system_size)]) for coupling_location in range(system_size)]
        
        

#         Defining target state        
        
        print(f'system_size={system_size}')
    
        target_psi=(np.linalg.eigh(AKLT(system_size))[1].T)[0]
        
        target_rho=np.outer(target_psi,target_psi)
        
        manybody_identity=chained_kron([S0 for i in range(system_size)])
        
        
        
#         Defining starting state
        
        if starting_state_name=='all_down': 

            starting_state=manybody_identity[0]        
        
        elif starting_state_name=='fully_mixed':
            if pure_state_simulation:
                raise Exception('Can"t do pure state simulation with fully mixed starting state!')      
                
            starting_state=manybody_identity  
            
        else:            
            raise Exception('Can"t parse starting_state_name!')
        
        
        
    
    
    elif type_of_system=='W_state':
        
        if system_size!=3:
            raise Exception('Wrong system size!')
        
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

        psi_minus=chained_kron([up,down,down])-chained_kron([down,down,up])
        psi_minus=psi_minus/np.linalg.norm(psi_minus)

        psi_plusminus=chained_kron([up,down,down])+chained_kron([down,down,up])-2*chained_kron([down,up,down])
        psi_plusminus=psi_plusminus/np.linalg.norm(psi_plusminus)
        
#         Defining target state
        
        target_psi=chained_kron([up,down,down])+chained_kron([down,up,down])+chained_kron([down,down,up])
        target_psi=target_psi/np.linalg.norm(target_psi)
        
        target_rho=np.outer(target_psi,target_psi)

        down_down_down=chained_kron([down, down, down])
        
#         Defining starting state
        if starting_state_name=='all_down': 

            starting_state=down_down_down      
            
        else:            
            raise Exception('Can"t parse starting_state_name!')
        
        
    else:
        
        raise Exception('Don"t recognise system type!')
        
        
    if ( (not pure_state_simulation) and (starting_state.shape==(system_size,) ) ):

        starting_state=np.outer(starting_state,starting_state)
        
        
        
        
        
        
def antisym_fourdim(a):    
    A=np.array([[0,a[0],a[1],a[2]],[-a[0],0,a[3],a[4]],[-a[1],-a[3],0,a[5]],[-a[2],-a[4],-a[5],0]])
    return(A)

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
    

    
    mb_V=manybody_term(V,[coupling_location,np.mod(coupling_location+1,system_size)])
    
    return(mb_V)
