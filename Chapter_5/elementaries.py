import numpy as np

trivial_operator=np.array([1])

data_type='complex128'

sx=np.zeros((2,2),dtype=data_type)
sx[0,1]=1
sx[1,0]=1

sy=np.zeros((2,2),dtype=data_type)
sy[0,1]=-1j
sy[1,0]=1j

sz=np.zeros((2,2),dtype=data_type)
sz[0,0]=1
sz[1,1]=-1

unity=np.eye(2,dtype=data_type)

pauli=[unity,sx,sy,sz]

basis=np.eye(2,dtype=data_type)

number_of_qubits=6

identity=np.eye(2**number_of_qubits, dtype=data_type)

default_state=identity[0]

x_default_state=np.ones(2**number_of_qubits)/np.sqrt(2**number_of_qubits)

def kron_delta(x,y):
    if (x==y):
        output=1
    else:
        output=0
    return(output)

def pauli_operator(position_of_pauli, type_of_pauli):
    '''
    Calculates a single pauli operator.
    '''    
   
    u=trivial_operator
    for i in range(number_of_qubits):
        if i==position_of_pauli:
            u=np.kron( u , pauli[type_of_pauli] )
        else:
            u=np.kron( u , pauli[0] )
            
    return(u)

# def pauli_string_operator(string_identifier):
#     '''
#     Calculates a pauli string operator.
    
#     Inputs a string identifier, i.e. a list of string's paulis. List elements are pairs: [position of a pauli, type of a pauli].
#     '''    
    
#     string_operator=identity()
    
#     #String identifier contains pairs of elements: 0 - position of a pauli, 1 - type of a pauli
#     for i in range(len(string_identifier)):
#         string_operator=string_operator.dot( pauli_operator(string_identifier[i][0] , string_identifier[i][1]) )
    
#     return(string_operator)

def pauli_string_operator(string_identifier):
    '''
    Calculates a pauli string operator.
    
    Inputs a string identifier, i.e. a list of string's paulis. List elements are pairs: [position of a pauli, type of a pauli].
    '''    
    
    string_operator=identity
    
    for element in string_identifier:
        string_operator=string_operator.dot(pauli_operator(element[0],element[1]))

            
    return(string_operator)

def list_of_strings(list_of_string_ids):
    '''
    Calculates a list of Pauli string operators.
    
    Inputs a list of string identifiers.
    '''
    
    lis=[]
    
    #list of strings contains elements which are lists of paulis
    
    for i in range(len(list_of_string_ids)):
        lis+=[pauli_string_operator(list_of_string_ids[i])]
    
    return(lis)


def string_iexp(string, theta):
    '''
    Calculates an imaginary exponential of a Pauli string, multiplied by parameter theta.
    
    Inputs a string operator and the angle theta, outputs an exponential operator.
    '''
    return(np.cos(theta)*identity+1j*np.sin(theta)*string)


def state_from_bitstring(bitstring, prefactor=1):
    '''
    Creates a 2^N dimensional computational basis state vector. 

    Inputs a corresponding bitstring [s1,s2..s_N] with s_i=0,1, and a prefactor c:
    
    [s1,s2,..s_N], c ---> c*|s1,s2,s3,..s_N>
    '''
    
    state=trivial_operator*prefactor
    
    for element in bitstring:
        state=np.kron(state,basis[element])
    
    return(state)

def partial_trace(state, cut=number_of_qubits//2, number_of_qubits=number_of_qubits):
    density_matrix=np.transpose([state]).dot(np.array([state]))
    density_matrix=density_matrix.reshape((2**cut, 2**(number_of_qubits-cut), 2**cut, 2**(number_of_qubits-cut) ))
    density_matrix=np.transpose(density_matrix, (0,2,1,3))
    red_matrix=np.trace(density_matrix)
    
    return(red_matrix)

def entropy(density_matrix):
    eigenvalues=np.linalg.eig(density_matrix)[0]
    S=0
    for element in eigenvalues:
        S-=element*np.log(element)
    return(S)
