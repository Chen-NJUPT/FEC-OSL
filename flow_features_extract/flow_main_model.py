__author__ = 'dk'
import os
import dataset_builder
import random

device= "cuda:1"

class model:
    def __init__(self, dataset, randseed, splitrate, all_classes, known_classes):
        self.database = './data/'
        self.name = 'flow'
        self.rand = random.Random(x = randseed)
        self.data = None
        self.model = None
        self.full_rdata = []
        self.dataset = dataset
        if os.path.exists(self.database) == False:
            os.makedirs(self.database,exist_ok=True)
        self.splitrate = splitrate

        self.data_path = self.database + self.name + '_' + self.dataset
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.dumpFilename  = self.data_path + "/dataset_builder.pkl_" + str(known_classes) + "_" + \
            str(all_classes - known_classes) +  ".gzip"
        if not os.path.exists(self.dumpFilename):
            print(f"dumpFilename {self.dumpFilename} does not exist!")

    def parse_raw_data(self, all_classes, known_classes):
        self.full_rdata = os.path.join(".", "dataset", "flow_data", self.dataset)
        
        if self.dataset.startswith("USTC-TFC2016"):
            self.data_loader = dataset_builder.FlowContainerJSONDataset(mode='clear',
                dumpData=True,usedumpData=True,
                dumpFilename=self.dumpFilename,
                cross_version=False,
                test_split_rate=self.splitrate,
                graph_json_directory=self.full_rdata,
                all_classes = all_classes, known_classes = known_classes)
            
        elif self.dataset.startswith("tor_session"):
            self.data_loader = dataset_builder.FlowContainerJSONDataset(mode='clear',
                dumpData=True,usedumpData=True,
                dumpFilename=self.dumpFilename,
                cross_version=False,
                test_split_rate=self.splitrate,
                graph_json_directory=self.full_rdata,
                all_classes = all_classes, known_classes = known_classes)
        elif self.dataset.startswith("IDS"):
            self.data_loader = dataset_builder.FlowContainerJSONDataset(mode='clear',
                dumpData=True,usedumpData=True,
                dumpFilename=self.dumpFilename,
                cross_version=False,
                test_split_rate=self.splitrate,
                graph_json_directory=self.full_rdata,
                all_classes = all_classes, known_classes = known_classes)
        return self.data_loader

