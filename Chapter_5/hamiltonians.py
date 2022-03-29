import numpy as np
import elementaries as el
from numpy import random


def single_qubit_paulis (alpha):
    
    pauli_ids=[[[i, alpha]] for i in range(el.number_of_qubits)]
    
    return(pauli_ids)

def two_qubit_strings (alpha, beta):
    
    string_ids=[ [[i, alpha], [j, beta]] for i in range(el.number_of_qubits) for j in range(el.number_of_qubits)]
    
    return(string_ids)

def nearest_neighbour_1D (alpha,beta):
    
    string_ids=[ [[i, alpha], [i+1, beta]] for i in range(el.number_of_qubits-1)]
    
    return(string_ids)

def constant_subhamiltonian (magnitude, generator_ids):
    
    hamiltonian=0.*el.identity
    
    for element in generator_ids:
        hamiltonian+=el.pauli_string_operator(element)
    
    hamiltonian*=magnitude
    
    return(hamiltonian)

def random_subhamiltonian (disorder_strength, generator_ids):
    hamiltonian=0.*el.identity
    
    for element in generator_ids:      
        hamiltonian=hamiltonian+random.normal(0,disorder_strength)*el.pauli_string_operator(element)
    
    return(hamiltonian)

# def custom_subhamiltonian(filter, generators):
    