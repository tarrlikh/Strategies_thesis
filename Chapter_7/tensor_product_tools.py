import settings
import numpy as np
from functools import *

def kron_to_tensor(matrix_operator, size=settings.system_size):
    '''
    Turns a kronecker product into tensor product
    
    Uses global variables system_size, spin_dim
    '''
    
    
    
    index_ranges=tuple(np.ones(2*size,int)*settings.spin_dim)
    tensor_operator=matrix_operator.reshape(index_ranges)
    
    
    return(tensor_operator)

def tensor_to_kron(tensor_operator, size=settings.system_size):
    '''
    Turns a tensor product into kronecker product
    
    Uses global variables system_size, spin_dim
    '''
    
    
    index_ranges=(settings.spin_dim**size,settings.spin_dim**size)
    matrix_operator=tensor_operator.reshape(index_ranges)
    
    
    return(tensor_operator)



    
def partial_trace(matrix_operator, spins_to_traceout, size=settings.system_size):
    
    '''
    Traceout the spins with labels spins_to_traceout
    
    Uses global variables system_size, spin_dim
    '''
    
    tensor_operator=kron_to_tensor(matrix_operator, size=size)
    
    spins_ids=list(range(size))
    
    spins_to_remain=[spin_id for spin_id in spins_ids if spin_id not in spins_to_traceout]
    
    shuffle=tuple(spins_to_traceout+[spin_id+size for spin_id in spins_to_traceout]+spins_to_remain+[spin_id+size for spin_id in spins_to_remain])
    
    reshuffled_tensor=tensor_operator.transpose(shuffle)
    
    bipartite_shape=(settings.spin_dim**len(spins_to_traceout),settings.spin_dim**len(spins_to_traceout),settings.spin_dim**(len(spins_to_remain)),settings.spin_dim**(len(spins_to_remain)))
    
    bipartite_operator=reshuffled_tensor.reshape(bipartite_shape)
    
    return(np.trace(bipartite_operator))

def chained_kron(list_of_matrices):
    '''
    Returns a kronecker product of a list of matrices, from 1 to last
    
    Uses no global variables
    '''
    return(reduce(np.kron, list_of_matrices)) 



def permutation (i, j, size):
    '''
    Returns a range(size) with j and i permuted 
    
    Uses no global variables
    '''
    
    
    output=list(range(size))
    output[i]=j
    output[j]=i
    return(output)


def kron_permute(kron_operator, new_spins_list, size=settings.system_size):
    
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
    
    manybody_identity=np.eye(settings.spin_dim**settings.system_size)
    
    mbterm=np.kron(manybody_identity, term)    
    
    disposable_term_labels=term_labels
    
    new_spins_list=[settings.system_size+term_labels.index(spin_label) if spin_label in term_labels else spin_label for spin_label in range(settings.system_size)]
    
    new_spins_list+=term_labels   
    
    mbterm=kron_permute(mbterm,new_spins_list, size=settings.system_size+len(term_labels))
    
    spins_to_traceout=list(range(settings.system_size,settings.system_size+len(term_labels)))
    
    mbterm=partial_trace(mbterm, spins_to_traceout, size=settings.system_size+len(term_labels))/(settings.spin_dim**len(term_labels))
    
    
    return(mbterm)



