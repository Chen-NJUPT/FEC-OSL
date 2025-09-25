__author__ = 'HPC'

import random
import gzip
import os
import dgl
import pickle
import torch as th
from construct_graph_my import build_graphs_from_flowcontainer_json,mtu,mtime, concurrent_time_threshold
import json

root_dir  =  "dataset/"
random.seed(100)
time_period= 1

class FlowContainerJSONDataset:
    def __init__(self, graph_json_directory=root_dir+"tor_session_JSON/",mode='clear',dumpData=False,usedumpData=False,dumpFilename="dataset_builder.pkl_know.gzip",cross_version=False,test_split_rate=0.1, all_classes = 8, known_classes = 6):

        self.dumpFileName = dumpFilename
        if usedumpData==True and os.path.exists(dumpFilename):
            fp = gzip.GzipFile(dumpFilename,"rb")
            data=pickle.load(fp)
            fp.close()
            self.labelName = data['labelName']
            self.labelNameSet=data['labelNameSet']
            self.graphs = data['graphs']
            self.labelId = data['labelId']
            self.known_train_index = data['known_train_index']
            self.known_test_index = data['known_test_index']
            self.known_valid_index = data['known_valid_index']
            self.unknown_train_index = data['unknown_train_index']
            self.unknown_test_index = data['unknown_test_index']
            self.unknown_valid_index = data['unknown_valid_index']
            info ='Load dump data from {0}'.format(dumpFilename)
            
        else:
            if os.path.isdir(graph_json_directory)== False:
                info = '{0} is not a directory'.format(graph_json_directory)
                raise BaseException(info)
            assert mode in ['clear','noise','all']
            self.labelName = []
            self.labelNameSet = {}
            self.labelId = []
            self.graphs = []
            
            _labelNameSet = []  
            for _root,_dirs,_files in os.walk(graph_json_directory):
                
                if _root == graph_json_directory or len(_files) == 0:
                    continue
                _root =_root.replace("\\","/")
                packageName = _root.split("/")[-1]
                labelName=packageName
                _labelNameSet.append(labelName)
            _labelNameSet.sort()
            for i in range(len(_labelNameSet)):
                self.labelNameSet.setdefault(_labelNameSet[i], len(self.labelNameSet))

            for labelName in _labelNameSet:
                folder_path = os.path.join(graph_json_directory, labelName).replace("\\", "/")

                if not os.path.isdir(folder_path):  
                    continue

                _files = os.listdir(folder_path)  
                for index in range(len(_files)):
                    file = _files[index]
                    if file == ".DS_Store":  
                        continue

                    json_fname = os.path.join(folder_path, file).replace("\\", "/")
                    
                    gs = build_graphs_from_flowcontainer_json(json_fname, time_period=time_period, all_classes = all_classes, known_classes = known_classes)
                    if len(gs) < 1 or gs[0] is None:
                        continue
                    
                    self.graphs += gs
                    self.labelName += [labelName] * len(gs)  
                    self.labelId += [self.labelNameSet[labelName]] * len(gs)  

                    assert self.labelId[-1] in range(len(self.labelNameSet))
            assert len(self.graphs) == len(self.labelId)
            assert len(self.graphs) == len(self.labelName)
            info = "Build {0} graph over {1} classes, {2} graph per class. {3} flow.".format(len(self.graphs),len(self.labelNameSet),len(self.graphs)//len(self.labelNameSet),self.flowCounter)
            
            self.known_train_index = []
            self.known_valid_index = []
            self.known_test_index =  []
            self.unknown_train_index = []
            self.unknown_valid_index = []
            self.unknown_test_index =  []

            with open('dataset/byte_data/data_index_' + str(known_classes) + "_" + str(all_classes - known_classes) +  '.json', 'r') as json_file:
                temp = json.load(json_file)  
                self.known_train_index = temp['known_train_indices']
                self.known_valid_index = temp['known_val_indices']
                self.known_test_index =  temp['known_val_indices']
                
                self.unknown_train_index = temp['aunknown_train_indices']
                self.unknown_valid_index = temp['unknown_val_indices']
                self.unknown_test_index =  temp['unknown_val_indices']
                
            if dumpData :
                self.dumpData()    
        self.class_aliasname ={}
        labelNameSet = list(self.labelNameSet)
        
        labelNameSet.sort()             
        for i in range(len(labelNameSet)):
            self.class_aliasname.setdefault(i,labelNameSet[i])
        
        self.train_watch = 0
        self.test_watch =  0
        self.valid_watch = 0
        self.epoch_over = False

    def dumpData(self,dumpFileName=None):
        if dumpFileName == None:
            dumpFileName = self.dumpFileName
        fp = gzip.GzipFile(dumpFileName,"wb")
        pickle.dump({
                'graphs':self.graphs,
                'flowCounter':self.flowCounter,
                'labelName':self.labelName,
                'labelNameSet':self.labelNameSet,
                'labelId':self.labelId,
                'known_train_index':self.known_train_index,
                'known_valid_index':self.known_valid_index,
                'known_test_index':self.known_test_index,
                'unknown_train_index':self.unknown_train_index,
                'unknown_valid_index':self.unknown_valid_index,
                'unknown_test_index':self.unknown_test_index
                
            },file=fp,protocol=-1)
        fp.close()

    def reflesh(self):
        self.train_watch = 0
        
    def __next_batch(self, name, batch_size):
        graphs = []
        labels = []

        if name == 'train':
            
            remaining = len(self.train_index) - self.train_watch
            current_batch_size = min(batch_size, remaining)  

            for i in range(current_batch_size):
                graphs.append(self.graphs[self.train_index[self.train_watch]])
                labels.append(self.labelId[self.train_index[self.train_watch]])
                self.train_watch += 1

            
            if self.train_watch >= len(self.train_index):
                self.epoch_over += 1
                self.train_watch = 0  

        elif name == 'valid':
            remaining = len(self.valid_index) - self.valid_watch
            current_batch_size = min(batch_size, remaining)

            for i in range(current_batch_size):
                graphs.append(self.graphs[self.valid_index[self.valid_watch]])
                labels.append(self.labelId[self.valid_index[self.valid_watch]])
                self.valid_watch += 1

            if self.valid_watch >= len(self.valid_index):
                self.valid_watch = 0

        else:
            remaining = len(self.test_index) - self.test_watch
            current_batch_size = min(batch_size, remaining)

            for i in range(current_batch_size):
                graphs.append(self.graphs[self.test_index[self.test_watch]])
                labels.append(self.labelId[self.test_index[self.test_watch]])
                self.test_watch += 1

            if self.test_watch >= len(self.test_index):
                self.test_watch = 0

        return dgl.batch(graphs), th.tensor(labels)

    def next_train_batch(self,batch_size):
        return self.__next_batch('train',batch_size)
    
    def next_valid_batch(self,batch_size):
        return self.__next_batch('valid',batch_size)
    
    def next_test_batch(self,batch_size):
        return self.__next_batch('test',batch_size)
    
    def __next_batch_know(self, name, batch_size):
        graphs = []
        labels = []

        if name == 'train':
            
            remaining = len(self.known_train_index) - self.train_watch
            current_batch_size = min(batch_size, remaining)  
            for i in range(current_batch_size):
                graphs.append(self.graphs[self.known_train_index[self.train_watch]])
                labels.append(self.labelId[self.known_train_index[self.train_watch]])
                self.train_watch += 1

            
            if self.train_watch >= len(self.known_train_index):
                self.epoch_over += 1
                self.train_watch = 0  

        elif name == 'valid':
            remaining = len(self.known_valid_index) - self.valid_watch
            current_batch_size = min(batch_size, remaining)

            for i in range(current_batch_size):
                graphs.append(self.graphs[self.known_valid_index[self.valid_watch]])
                labels.append(self.labelId[self.known_valid_index[self.valid_watch]])
                self.valid_watch += 1

            if self.valid_watch >= len(self.known_valid_index):
                self.epoch_over += 1
                self.valid_watch = 0

        else:
            remaining = len(self.known_test_index) - self.test_watch
            current_batch_size = min(batch_size, remaining)

            for i in range(current_batch_size):
                graphs.append(self.graphs[self.known_test_index[self.test_watch]])
                labels.append(self.labelId[self.known_test_index[self.test_watch]])
                self.test_watch += 1

            if self.test_watch >= len(self.known_test_index):
                self.epoch_over += 1
                self.valid_watch = 0

        return dgl.batch(graphs), th.tensor(labels)

    def next_train_batch_know(self,batch_size):
        return self.__next_batch_know('train', batch_size)
    
    def next_valid_batch_know(self,batch_size):
        return self.__next_batch_know('valid', batch_size)
    
    def next_test_batch_know(self,batch_size):
        return self.__next_batch_know('test', batch_size)
    
    def __next_batch_unknow(self, name, batch_size):
        graphs = []
        labels = []

        if name == 'train':
            
            remaining = len(self.unknown_train_index) - self.train_watch
            current_batch_size = min(batch_size, remaining)  

            for i in range(current_batch_size):
                graphs.append(self.graphs[self.unknown_train_index[self.train_watch]])
                labels.append(self.labelId[self.unknown_train_index[self.train_watch]])
                self.train_watch += 1

            
            if self.train_watch >= len(self.unknown_train_index):
                self.epoch_over += 1
                self.train_watch = 0  

        elif name == 'valid':
            remaining = len(self.unknown_valid_index) - self.valid_watch
            current_batch_size = min(batch_size, remaining)

            for i in range(current_batch_size):
                graphs.append(self.graphs[self.unknown_valid_index[self.valid_watch]])
                labels.append(self.labelId[self.unknown_valid_index[self.valid_watch]])
                self.valid_watch += 1

            if self.valid_watch >= len(self.unknown_valid_index):
                self.epoch_over += 1
                self.valid_watch = 0

        else:
            remaining = len(self.unknown_test_index) - self.test_watch
            current_batch_size = min(batch_size, remaining)

            for i in range(current_batch_size):
                graphs.append(self.graphs[self.unknown_test_index[self.test_watch]])
                labels.append(self.labelId[self.unknown_test_index[self.test_watch]])
                self.test_watch += 1

            if self.test_watch >= len(self.unknown_test_index):
                self.epoch_over += 1
                self.test_watch = 0

        return dgl.batch(graphs), th.tensor(labels)

    def next_train_batch_unknow(self,batch_size):
        return self.__next_batch_unknow('train',batch_size)
    
    def next_valid_batch_unknow(self,batch_size):
        return self.__next_batch_unknow('valid',batch_size)
    
    def next_test_batch_unknow(self,batch_size):
        return self.__next_batch_unknow('test',batch_size)
    
    
