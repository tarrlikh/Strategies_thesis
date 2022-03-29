import sys
import random


import settings

from tensor_product_tools import *

from steering_simulation import *

from steering_outputting import *



settings.init()

settings.type_of_system=sys.argv[1]

settings.starting_state_name=sys.argv[2]

settings.type_of_protocol=sys.argv[3]

settings.target_fidelity=eval(sys.argv[4])

settings.system_size=eval(sys.argv[5])

name_of_the_dataset=f'{settings.type_of_system}_from_{settings.starting_state_name}_with_{settings.type_of_protocol}_to_fidelity={settings.target_fidelity}/N={settings.system_size}'

settings.number_of_experiments=eval(sys.argv[6])

settings.slur_id=sys.argv[7]

# Defining settings.spin_dim, settings.target_psi, settings.mixed_state_simulation, settings.target_rho

settings.system_properties()

print(f'{name_of_the_dataset} - {settings.number_of_experiments}, {settings.slur_id}')

settings.duration_datasets={name_of_the_dataset: []}

durations=[]

for j in range(settings.number_of_experiments):
    
    current_state=settings.starting_state

    for i in range(settings.duration_cutoff):
        
        coupling_locator=np.mod(i,settings.system_size)
        
        '''For scenario-based steering (start):'''
#         coupling_strategy=coupling_strategy_function(steering_scenario, strategies_W_state_blind)
#         available_couplings=[total_coupling_list[coupling_id] for coupling_id in coupling_strategy]
        
        
#         coupling=available_couplings[np.mod(i,len(available_couplings))]
        
#         ever_click_decided=operator_action(current_state,coupling[0])
#         never_click_decided=operator_action(current_state,coupling[1])
        
        '''For scenario-based steering (end)'''
        
       
        
    
        '''(start) continuous optimization steering'''
        
#         available_couplings=[total_coupling_list[np.mod(i,settings.system_size)]]
        
        never_click_decided=operator_action(current_state, settings.mb_P_list[coupling_locator])
        
        def global_fit_to_optimize(a):                    
            return(-np.real(glob_fit(operator_action(current_state, settings.mb_V_func(a, coupling_locator))[0])))            

        def local_fit_to_optimize(a):                    
            return(-np.real(loc_fit(operator_action(current_state, settings.mb_V_func(a, coupling_locator))[0])))

#         rotation_parameters=minimize(local_fit_to_optimize, np.zeros(6), method='SLSQP', options={'maxiter':10}).x     

#             TO COMPARE, here are some versions without continuous optimization

        rotation_parameters=np.array([random.random() for parameter_label in range(6)])
    
#         rotation_parameters=np.array([0 for parameter_label in range(6)])

        ever_click_decided=operator_action(current_state, settings.mb_V_func(rotation_parameters, coupling_locator))



        '''(end) continuous optimization steering'''
    
    
    
    
        coin=random.random()

        
        if (never_click_decided[1]<coin):      
            
            current_state=ever_click_decided[0]   
            
        else:
            current_state=never_click_decided[0]        

        current_glob_fit=glob_fit(current_state)

        if (current_glob_fit>settings.target_fidelity):
            break
    
    durations+=[i]
    
    if np.mod(j,2)==0:
            print(f'\n run={j}')
  


print(f'attempting update with {durations}...')

update(name_of_the_dataset, durations)

print(f'updated successfully')
    
print(durations)
    