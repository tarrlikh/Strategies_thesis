import numpy as np
import elementaries as el
import hamiltonians as ham
import itertools

'''
This module assumes that the hamiltonian of interest is of a generalized transverse Ising type
'''

def connectivity_check(S, coupling):
    return( ((S[coupling[0]]!=0)|(S[coupling[1]]!=0))&(not((S[coupling[0]]%2!=0)&(S[coupling[1]]%2!=0))) )

def perturbative_diagrams(order, interactions):
    
    #interaction filter is of the following form: list of pairs of vertices, which have a nonzero coupling between them
    
    diagram_list=[]
    
    
    
    diagrams=[[[0 for i in range(el.number_of_qubits)]]]
    
    
    top_diagrams=diagrams[0]
        
    for n in range(order):
        new_diagrams=[]
        for diagram in top_diagrams:
            for coupling in interactions:
                if (connectivity_check(diagram,coupling)|(n==0)):
                    new_diagram=diagram.copy()
                    for qubit in coupling:
                        new_diagram[qubit]+=1
                    new_diagrams+=[new_diagram]
        new_diagrams=[list(map(lambda element: element%2+(int(np.sign(element-1)+1)//2)*((element+1)%2)*2, diagram)) for diagram in new_diagrams]
        new_diagrams.sort()
        new_diagrams=list(k for k,_ in itertools.groupby(new_diagrams))
        diagrams+=[new_diagrams]
        y=[]
            
        top_diagrams=new_diagrams
    
    return(diagrams)
    #first-order perturbations: all interactions included

def states_from_diagrams(diagram_list):
    
    states=[]
    
    for diagram in diagram_list:
        states+=[list(map(lambda x: x%2, diagram))]
    
    return(states)
   