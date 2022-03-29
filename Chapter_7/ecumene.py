import pickle
import os
from functools import *
import re

class Ecumene:
    
    def __init__(self, context_name, method, target_directory=os.getcwd()):

    
        '''
        method - str, 'load' or 'new'
        '''

        self.name=context_name
        
        self.work_dir=target_directory+f'/ecumene/{self.name}'
        
        if method == 'new':

            if os.path.isdir(self.work_dir):
                raise ValueError('This context already has a dedicated directory; consider using load method')
            else:
                os.makedirs(self.work_dir)
                os.mkdir(self.work_dir+'/data')

            self.glossary={'meta': [['datatype', []], ['store_mode',['write', 'update_list', 'update_dictionary']]], 
                            'data': []}

            self.update_glossary_file()
                
            
        elif method == 'load':
            
            
            
            if os.path.isdir(self.work_dir+'/data') and os.path.isfile(self.work_dir+'/glossary'):
                pass
            else:
                raise ValueError('Could not load the instance!')

            with open(self.work_dir+'/glossary', 'rb') as fl:
                self.glossary=pickle.load(fl)
                
        else:
            
            raise Exception('Unknown method!')
                
        
            
    def update_glossary_file(self):
        
        with open(self.work_dir+'/glossary', 'wb') as fl:
                pickle.dump(self.glossary, fl)
                
                
    def parameter_list(self, parameter_type):
        '''
        parameter_type='meta', 'data' or 'all'
        '''
        if parameter_type=='all':
            param_list=[parameter[0] for parameter in self.glossary['meta']]
            param_list+=[parameter[0] for parameter in self.glossary['data']]
        elif parameter_type=='meta' or parameter_type=='data':
            param_list=[parameter[0] for parameter in self.glossary[parameter_type]]
        else:
            raise ValueError('The parameter type should be meta or data or all!')


        return param_list


    def parameter_values_list(self, parameter_name):

        if parameter_name in self.parameter_list('meta'):
            parameter_type='meta'
        elif parameter_name in self.parameter_list('data'):
            parameter_type='data'
        else:
            raise ValueError('No such parameter!')

        parameter_index=self.parameter_list(parameter_type).index(parameter_name)

        values_list=self.glossary[parameter_type][parameter_index][1]


        return values_list
    
    def add_parameter(self, parameter_type, parameter_name, default_parameter_value):
    
        '''
        parameter_type - str, only options 'meta' and 'data': type of the parameter
        parameter_name - str, the name of the parameter
        default_parameter_value - the parameter value you've been implicitly using in previous data; 
                                    if doesn't apply, input None
        '''
        
        
        if parameter_type=='data' or parameter_type=='meta':
            pass
        else:
            raise ValueError('The parameter type should be meta or data!')

        if parameter_name in self.parameter_list('all'):
            raise ValueError('The parameter is already present!')
        else:
            pass

        self.glossary[parameter_type]+=[[parameter_name, [default_parameter_value]]]
        
        
        
        
        
        # Changing all the data file names since a new parameter was introduced:
        
        all_filenames = os.listdir(self.work_dir+'/data')
        
        filename_regex=re.compile(".+\d__\d.+")

        our_filenames = [name for name in all_filenames if filename_regex.match(name)]
        
        # Now that all file names are stored, we can update the glossary file and let it be used for other processes
        # WARNING: This is if the glossary file was the key method; instead, right now we use the local object to do so -- has to be changed
        
        self.update_glossary_file()

#         Procedure:

# * Find all files with a pattern digit|digit in them, list their names
# * Create substituion names, edited in an appropriate fashion
# * Rename all files accordingly

        if parameter_type=='meta':

            new_filenames = [re.sub("__", "_0__", name) for name in our_filenames]

        elif parameter_type=='data':

            new_filenames = [re.sub("$", "_0", name) for name in our_filenames]

        else:

            raise Exception('Unknown parameter type!')


        for number in range(len(our_filenames)):

            os.rename(f'{self.work_dir}/data/{our_filenames[number]}', f'{self.work_dir}/data/{new_filenames[number]}')
        
        
    def add_parameter_values(self, parameter_name, parameter_values):
    
        if parameter_name in self.parameter_list('meta'):
            parameter_type='meta'
        elif parameter_name in self.parameter_list('data'):
            parameter_type='data'
        else:
            raise ValueError('No such parameter!')

        values_list=self.parameter_values_list(parameter_name)

        parameter_index=self.parameter_list(parameter_type).index(parameter_name)

        for value in parameter_values:
            if value in values_list:
                pass
            else:
                self.glossary[parameter_type][parameter_index][1]+=[value]
                
                
        self.update_glossary_file()
        
    def expand_glossary(self, parameter_type, extra_parameters):
        
        for parameter_name, parameter_values in extra_parameters.items():
            if parameter_name not in self.parameter_list('all'):
                self.add_parameter(parameter_type, parameter_name, parameter_values[0])
                self.add_parameter_values(parameter_name,parameter_values[1:])
            elif parameter_name in self.parameter_list(parameter_type):
                self.add_parameter_values(parameter_name,parameter_values[0:])
            else:
                raise ValueError(f'This parameter is not type {parameter_type}!')
            
        self.update_glossary_file()
                
    
    def store_data(self, data, parameters, postfix=None, encoded=False):
        
        if not encoded:
            
            encoded_parameters=self.parameters_encoder(parameters)
            
        else:
            
            encoded_parameters=parameters
        
        if type(postfix)==type(None):
        
            file_name=f'{self.work_dir}/data/{encoded_parameters}'
        
        else:
            
            file_name=f'{self.work_dir}/data/{encoded_parameters}____'+postfix
        
        if parameters['store_mode'] == 'write':

            #Assumes this data is stored only once

            if os.path.isfile(file_name):
                # This logging method should probably be replaced by something more elegant
                print('This file already exists!')
            else:
                with open(file_name, 'wb') as fl:
                    pickle.dump(data,fl)
                    
        elif parameters['store_mode'] == 'overwrite':       
            with open(file_name, 'wb') as fl:
                    pickle.dump(data,fl)

        elif parameters['store_mode'] == 'update_list':

            # Assumes the data is a list, to be added to an existing list

            if os.path.isfile(file_name):
                with open(file_name, 'rb') as fl:
                    existing_list=pickle.load(fl)
            else:
                print('List does not exist yet!')
                existing_list=[]

            with open(file_name, 'wb') as fl:
                pickle.dump(existing_list+data, fl)

        elif parameters['store_mode'] == 'update_dictionary':

            # Assumes the data is a dictionary, and the existing dictionary is to be updated with that

            if os.path.isfile(file_name):
                with open(file_name, 'rb') as fl:
                    existing_dict=pickle.load(fl)
            else:
                print('Dict does not exist yet!')
                existing_dict=dict()

            with open(file_name, 'wb') as fl:
                existing_dict.update(data)
                pickle.dump(existing_dict, fl)

        else:
            raise ValueError('Unknown storing mode!')
            
            
            
    def collect_postfixes(self, parameters):
        
        all_filenames = os.listdir(self.work_dir+'/data')
        
        
        filename_regex=re.compile(f"{self.parameters_encoder(parameters)}____.+")

        postfixed_filenames = sorted([name for name in all_filenames if filename_regex.match(name)])
        
        if (parameters['store_mode'] == 'update_dictionary') | (parameters['store_mode'] == 'update_list'):
            
            for number, filename in enumerate(postfixed_filenames):
                
                self.store_data(self.load_data(filename, encoded=True), parameters)
                
                print(f'Collected item {number} from postfixed data! Removing the postfixed data...')
                
                self.remove_data(filename, encoded=True)
            
        else:
            
            raise Exception('This store mode isn"t supposed to be used with postfixes!')
            
        print(f'Done! Collected {len(postfixed_filenames)} items of postfixed data.')
        
                
                
        
        
        return postfixed_filenames
            
        
    def exists_already(self, parameters):
        
        file_name=f'{self.work_dir}/data/{self.parameters_encoder(parameters)}'
        
        if os.path.isfile(file_name):
            return True
        else:
            return False
    
    def load_data(self, parameters, encoded=False):
        
        #note: this can be used with postfix, just use the filename directly with encoded set to True
        
        if not encoded:
        
            file_name=f'{self.work_dir}/data/{self.parameters_encoder(parameters)}'
        
        else:
            
            file_name=f'{self.work_dir}/data/{parameters}'


        if not os.path.isfile(file_name):
            # This logging method should probably be replaced by something more elegant
            raise Exception(f'This file doesnt exist yet: {parameters}')
        else:
 
                
                
            
            try:
                with open(file_name,'rb') as fl:
                    data=pickle.load(fl)
            except:
           
                raise Exception(f'File {parameters} may be corrupt! Consider removing.')
                    
        return(data)
    
    
    def remove_data(self,parameters, encoded=False):
        
        if not encoded:
        
            encoded_parameters=self.parameters_encoder(parameters)
            
        else:
            
            encoded_parameters=parameters
        
        file_name=f'{self.work_dir}/data/{encoded_parameters}'
        
        if os.path.isfile(file_name):
            os.remove(file_name)
    
    def parameters_encoder(self, parameters):



        #This is just to check that the set of parameters is valid:

        current_parameter_list=self.parameter_list('all')

        if set(current_parameter_list)==set(parameters):
            pass
        else:
            raise ValueError(f'Incommensurate set of parameters: input \n {parameters} \n while it should be \n {current_parameter_list} ')

        #This is to produce the code:    
        code_meta=[]

        for parameter_type in self.glossary['meta']:


            if parameters[parameter_type[0]] in parameter_type[1]:
                code_meta+=[parameter_type[1].index(parameters[parameter_type[0]])]
            else:
                raise Exception(f'Unknown value {parameters[parameter_type[0]]} of parameter {parameter_type[0]}!')


        code_data=[]

        for parameter_type in self.glossary['data']:
            if parameters[parameter_type[0]] in parameter_type[1]:
                code_data+=[parameter_type[1].index(parameters[parameter_type[0]])]
            else:
                raise Exception(f'Unknown value {parameters[parameter_type[0]]} of parameter {parameter_type[0]}!')




        # code_meta = [m1, m2, .., mM] and code_data = [d1, d2, ..dD]
        # converted to a list of strings ['m1_', 'm2_', ..'mM', '__', 'd1', 'd2', .. 'dD']
        outfile_name=[str(code_meta[k])+'_' if k<(len(code_meta)-1) else str(code_meta[k]) 
                        for k in range(len(code_meta))]+['__']+[str(code_data[k])+'_' 
                        if k<(len(code_data)-1) else str(code_data[k]) for k in range(len(code_data))]


        # through an abuse of notation, converts this to a string of type 'm1_m2_.._mM__d1_d2_..dD'
        outfile_name=reduce(lambda x, y: x+y, outfile_name)
        
        
        return (outfile_name)  
    
    
    def which_type(self, parameter_name):
        
        if parameter_name in self.parameter_list('meta'):
            param_type='meta'
        elif parameter_name in self.parameter_list('data'):
            param_type='data'
        else:
            raise ValueError('Parameter name is not in the glossary!')
                                         
        return param_type
    
    def param_value_description(self, mode, parameter_name, parameter_value, parameter_description=dict()):
    
        '''
        stores a dictionary of descriptions for the parameter value
        mode - str, 'update', 'read'
        '''
        
        parameter_type=self.which_type(parameter_name)
        
        type_id=0 if parameter_type=='meta' else 1
        
        parameter_id=self.parameter_list(parameter_type).index(parameter_name)
        
        parameter_value_id=self.glossary[parameter_type][parameter_id][1].index(parameter_value)
        
        path_name=f'{self.work_dir}/parameter_descriptions/'
        
        file_name=path_name+f'{type_id}_{parameter_id}_{parameter_value_id}'
               
        if not os.path.isdir(path_name):
            os.makedirs(path_name)
        
        
        if mode=='update':
            if os.path.isfile(file_name):
                with open(file_name, 'rb') as fl:
                    existing_dict=pickle.load(fl)
            else:
                existing_dict=dict()

            with open(file_name, 'wb') as fl:
                existing_dict.update(parameter_description)
                pickle.dump(existing_dict, fl)
                
        elif mode == 'read':
            
            if os.path.isfile(file_name):
                with open(file_name, 'rb') as fl:
                    output=pickle.load(fl)
            else:
                raise ValueError('Description does not exist!')
            
            return(output)
        
        else: 
            
            raise ValueError('Unknown mode of operation!')
            
            