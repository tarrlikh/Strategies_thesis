import elementaries as el
import ansatzes as ans
import hamiltonians as ham
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import basinhopping


def evaluate_expectation(ansatz_id, observable, ansatz_parameters):
    ansatz=el.list_of_strings(ansatz_id)
    
    return(ans.ansatz_state(ansatz, ansatz_parameters) .dot(observable.dot(np.transpose([ans.ansatz_state(ansatz, ansatz_parameters)])) ) )

def expectation_def(ansatz_parameters, ansatz, observable, in_state=el.default_state):
    state=ans.ansatz_state(ansatz, ansatz_parameters, in_state=in_state)
    return(np.real( state.dot(observable.dot(np.transpose([state]) ) ) )[0])



def adaptive_generator(ansatz_parameters, ansatz, hamiltonian, ansatz_ids):
    
    estimated_advantage=[]
    
    expectation=lambda observable: expectation_def(ansatz_parameters, ansatz, observable)
    
    for string_id in ansatz_ids:
        string_operator=el.pauli_string_operator(string_id)
        commutator=1j*(hamiltonian.dot(string_operator)-string_operator.dot(hamiltonian))
        double_commutator=1j*(hamiltonian.dot(commutator)-string_operator.dot(commutator))
        if (np.abs(expectation(double_commutator))<0.01):
            estimated_advantage+=[0.]
        else:
            estimated_advantage+=[expectation(commutator)*expectation(commutator)/np.abs(expectation(double_commutator))]
    logs=estimated_advantage
    
    return(ansatz_ids[estimated_advantage.index(max(estimated_advantage))], logs)

def optimize_expectation(ansatz_ids, observable, method=None, niter=2, starting_state=np.array([None]),tol=0.00000000001,min_tol=None, mult_order=None, adaptive=False, global_search=False, in_state=el.default_state, maxgates=None,progressbar=False,post_rotation=None):
    
    if min_tol==None:
        min_tol=tol
    
    expectation = lambda ansatz_parameters: expectation_def(ansatz_parameters, ansatz, observable, in_state=in_state)
    
    if starting_state.any()==None:
        starting_state=np.zeros(1)
    
    minimizer_kwargs = dict(method = method)
    
    new_starting_state_list=[]
    ansatz_id=[]
    ordered_ansatz_id=[]
    ansatz=el.list_of_strings(ordered_ansatz_id)
    
    minimizations=[]
    
    logs_1=[]
    logs_2=[]
    
    if maxgates==None:
        maxgates=len(ansatz_ids)
    
    for i in range(maxgates):
        if (progressbar&(i%5==0)):
            print('gates number:', i)
        
        logs_1+=[[ansatz_ids[i],adaptive_generator(np.array(new_starting_state_list), ansatz, observable, ansatz_ids)[0]]]    
        logs_2+=[adaptive_generator(np.array(new_starting_state_list), ansatz, observable, ansatz_ids)[1] ]
        
        if adaptive==False:
            ansatz_id+=[ansatz_ids[i]]            
        else:
            ansatz_id+=[adaptive_generator(np.array(new_starting_state_list), ansatz, observable, ansatz_ids)[0]]
            
        

        if mult_order==None:
            ordered_ansatz_id=ansatz_id
            
        else:
            indices=[]
            for generator in ansatz_id:
                indices+=[mult_order.index(generator)]
            
            permutation=list(np.argsort(indices))            
            
            ordered_ansatz_id=[ansatz_id[index] for index in permutation]
        
        if mult_order==None:
            starting_state=np.array(new_starting_state_list+[0.])
        else:      
            starting_state_list=new_starting_state_list
            starting_state_list.insert(permutation.index(i),0)
            starting_state=np.array(starting_state_list)
                
       
            
        ansatz=el.list_of_strings(ordered_ansatz_id)
                
#         expectation func to be inserted?
        renormalized_tolerance=np.exp(np.log(tol)*i/maxgates+np.log(min_tol)*(1-i/maxgates))
#         minimizations+=[minimize(expectation, starting_state , method=method, tol=renormalized_tolerance)]
        if global_search:
            minimizations+=[basinhopping(expectation, starting_state, niter=niter, minimizer_kwargs = minimizer_kwargs)]
        else:
            minimizations+=[minimize(expectation, starting_state,tol=tol, method=method)]
        new_starting_state_list=(minimizations[i].x).tolist()
        
        if type(new_starting_state_list)==float:
            new_starting_state_list=[new_starting_state_list]
            
        
        
            
    
    return( minimizations,logs_1,logs_2)

def optimize_fidelity(ansatz_ids, target_state, method=None, starting_state=np.array([None])):
    
    fidelity = lambda ansatz_parameters: -np.real( ( ans.ansatz_state(ansatz, ansatz_parameters) ).dot(target_state) )**2
    
    if starting_state.any()==None:
        starting_state=np.zeros(len(ansatz_ids[0]))
    

    
    ansatz_id=[]
    
    minimizations=[]
    
    for i in range(len(ansatz_ids)-1):
        ansatz_id+=ansatz_ids[i]
        ansatz=el.list_of_strings(ansatz_id)
        
#         fidelity = lambda ansatz_parameters: -np.real( ( ans.ansatz_state(ansatz, ansatz_parameters) ).dot(target_state) )**2
        
        minimizations+=[minimize( fidelity, starting_state , method=method)]
        
        new_starting_state=minimizations[i].x
        
        starting_state=np.array(list(new_starting_state)+[0. for j in range(len(ansatz_ids[i+1]))])
            
    ansatz_id+=ansatz_ids[len(ansatz_ids)-1]
    
    ansatz=el.list_of_strings(ansatz_id)
    
#     fidelity = lambda ansatz_parameters: -np.real( ( ans.ansatz_state(ansatz, ansatz_parameters) ).dot(target_state) )**2
    
    minimizations+=[minimize( fidelity, starting_state , method=method)]
    
    return(minimizations)


def optimize_fidelity_hessianmethods(ansatz_id, target_state, method=None):
    
    ansatz=el.list_of_strings(ansatz_id)
    
    fidelity = lambda ansatz_parameters: -np.real( ( ans.ansatz_state(ansatz, ansatz_parameters) ).dot(target_state) )**2 
    
    observable=-np.transpose([target_state]).dot(np.array([target_state]))
    
    first, sec = ans.hessian_preparation(ansatz_id)
    
    return( minimize( fidelity, 0.*np.ones(len(ansatz_id)), jac=lambda thetas: ans.jacobian_computation(observable, ansatz, thetas, first), method=method) )
#     return( minimize( fidelity, np.ones(len(ansatz)), jac=lambda thetas: ans.jacobian_computation(observable, ansatz, thetas, first), hess=lambda thetas: ans.hessian_computation(observable, ansatz, thetas, first, sec) , method=method) )