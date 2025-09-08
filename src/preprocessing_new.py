import os
from config import *
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

sc=StandardScaler()
conf = config()

def fix_seeds(seed=61):
    # Function to set random seed to ensure reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def torch_num_threads(num_threads):
    torch.set_num_threads(num_threads)

def torch_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

class Preprocessing:
    def __init__(self,params):
        self.is_exp=params["exp"]
        if self.is_exp:
            self.exp_root = params['exp_root']
            self.exp_name = params['exp_name']
            self.strains = params['exp_strains']
            self.transform = params['exp_transform']
            self.gen_method = params['exp_gen_method']
            self.CV_method = params['exp_CV_method']
            self.Xstd = params['exp_Xstd']
            self.ystd = params['exp_ystd']
            self.batch_size = params['exp_batch_size']
            self.train_size = params['exp_train_size']
            self.random_state = params['exp_random_state']
            self.folder_path=self.exp_root+"/"+self.exp_name
            self.root_path=self.folder_path
            self.sampledata_files = [i for i in os.listdir(self.folder_path) if (i.endswith('.csv') == True) and (i.startswith("tbl") ==True)]
            if params.get("enzfilter_file") is not None:
                self.enzfilter_file = self.folder_path+params["enzfilter_file"]
            else:
                self.enzfilter_file = None

    def EXP_csv2df(self,index):
        df=pd.read_csv(self.folder_path+"/"+self.sampledata_files[index],header=0,index_col="KEGG IDs")
        return df.T
    
    def EXP_filter_enzymes(self,df):
        if self.enzfilter_file is None:
            reaction_file=open(self.folder_path+'/stoichiometry.txt')
        else:
            reaction_file=open(self.enzfilter_file)
        reactions = reaction_file.read()
        enzymes = reactions.split('\n')
        if self.enzfilter_file is None:
            for i in range(len(enzymes)):
                enzymes[i] = enzymes[i].split(':')[0]
        columns_to_drop = [col for col in df.columns if not col.startswith('C') and col not in enzymes]
        df = df.drop(columns_to_drop, axis=1)
        return df

    def EXP_enzmetname(self):
        df=self.EXP_csv2df(0)
        df_selected_filtered=self.EXP_filter_enzymes(df)
        enzyme_list=[]
        metabolite_list=[]
        for column_name in df_selected_filtered.columns:
            if column_name.startswith("C"):
                metabolite_list.append(column_name)
            else:
                enzyme_list.append(column_name)
        return enzyme_list, metabolite_list 

    def def_enzmetname(self):
        self.enzyme_name,self.metabolite_name=self.EXP_enzmetname()

    def str2float(self,matrix):
        """Function to convert string numbers to float"""
        researved=matrix
        matrix=np.array(matrix)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j]=='':
                    matrix[i][j]=0
                    researved.iloc[i,j]=matrix[i][j]
                else: 
                    matrix[i][j] = float(matrix[i][j])
                    researved.iloc[i,j]=matrix[i][j]
        return researved

    def EXP_enz_fillnan(self, df):
        self.def_enzmetname()
        for column_name in df.columns:
            if df[column_name].isna().all():
                print(f"All values are NA in column: {column_name}")                      
            if column_name in self.enzyme_name:
                df[column_name]=df[column_name].fillna(df[column_name].mean())
        return df

    def EXP_sampledata_generator(self,index):
        df=self.EXP_csv2df(index)
        df_filtered=self.EXP_filter_enzymes(df)
        df_filtered["index_col"]=index
        df_filtered_enzfilled=self.EXP_enz_fillnan(df_filtered)
        return df_filtered_enzfilled

    def allsampledata_generator(self):
        if self.is_exp:
            for i in range(len(self.sampledata_files)):
                strain_df=self.EXP_sampledata_generator(i)
                if i == 0:
                    df_concat=strain_df
                else:
                    df_concat=pd.concat([df_concat,strain_df],axis=0)        
        else:
            print("Simulation data processing is no longer supported.")
        return df_concat

    def split_enzmet(self,df):
        if "index_col" in df.columns:
            df_no_index = df.drop("index_col", axis=1)
        else:
            df_no_index = df

        if self.is_exp:
            met = df_no_index.loc[:, df_no_index.columns.str.startswith('C')]
            enz = df_no_index.loc[:, ~df_no_index.columns.str.startswith('C')]
        else:
            enz = df_no_index.iloc[:, 0:len(self.enzyme_name)]
            met = df_no_index.iloc[:, len(self.enzyme_name):len(self.enzyme_name) + len(self.metabolite_name)]

        return enz, met

    def rc_input(self,method,threshold=None):
        original_enzyme=[]
        sample_concat=self.allsampledata_generator()
        if method=="onesample":
            for i in range(len(self.strains.split(','))):
                for j in range(sample_concat.shape[0]):
                    if sample_concat.iloc[j]["index_col"]==i:
                        enz,_=self.split_enzmet(sample_concat)
                        if self.transform=="log":
                            enz = enz.applymap(lambda x: np.log(x))
                        for i in range(enz.shape[0]):
                            original_enzyme.append(np.array(enz.iloc[i]))
                        pass
        
        else:
            sample_mean_df=sample_concat.groupby("index_col").mean()
            #for i in range(len(self.strains.split(','))):
                #print(f"{i}th strain:", sample_mean_df.index[i])
            #print("for i and for j i-j")
            enz,_=self.split_enzmet(sample_mean_df)
            if self.transform=="log":
                enz = enz.applymap(lambda x: np.log(x) )
            for i in range(enz.shape[0]):
                original_enzyme.append(np.array(enz.iloc[i]))

        diflogenz=[]
        for i in range(len(self.strains.split(','))):
            for j in range(len(self.strains.split(','))):
                if i==j:
                    pass
                else:
                    diflogenz.append(original_enzyme[i]-original_enzyme[j])

        testset=[]
        for input_i in range(len(diflogenz)):
            for enz_i in range(len(self.enzyme_name)):
                input_enz_i =enz_i % len(self.enzyme_name)
                testvector=np.zeros(len(self.enzyme_name))

                if method == "lowfilter":
                    if np.abs(diflogenz[input_i][input_enz_i])<threshold: 
                        # If the absolute value of ln(FoldChange) is below threshold, set to (-)threshold
                        if diflogenz[input_i][input_enz_i]>=0:
                            testvector[input_enz_i]=threshold
                        elif diflogenz[input_i][input_enz_i]<0:
                            testvector[input_enz_i]=-threshold

                    else:
                        testvector[input_enz_i]=diflogenz[input_i][input_enz_i]                
                else:
                    testvector[input_enz_i]=diflogenz[input_i][input_enz_i]
                testset.append(testvector)
        return testset

    def rc_prediction_input(self):
        """
        Method that performs the same operation as Data_Preprocessing.RCmatrix_prediction_input.
        - Extract original enzyme amount as one vector (row) for each strain (first row or arbitrary row)
        - Log transformation (if transform=="log")
        - Calculate differences between all strains to create diflogenz
        - Convert diflogenz to testvector with only one component having a value and return
        """
        strain_list = self.strains.split(',')
        original_enzyme = []

        # Extract original enzyme amount vector for each strain
        for idx, strain in enumerate(strain_list):
            if self.is_exp:
                # For experimental data: read idx-th file and use enzyme values from first row
                df = self.EXP_sampledata_generator(idx)
                # Split into (enz, met) using split_enzmet
                enz, _ = self.split_enzmet(df)
                # Extract first row (iloc[0])
                row0 = enz.iloc[0].copy()

                if self.transform == "log":
                    row0 = row0.apply(lambda x: np.log(x))

                original_enzyme.append(np.array(row0))
            else:
                print("Simulation data processing is no longer supported.")
                
        # Calculate differences between strains (diflogenz)
        diflogenz = []
        for i in range(len(strain_list)):
            for j in range(len(strain_list)):
                if i == j:
                    continue
                diflogenz.append(original_enzyme[i] - original_enzyme[j])

        # Expand each diflogenz vector to "only one component" format and add to testset
        testset = []
        if len(original_enzyme) > 0:
            enzlen = len(original_enzyme[0])
        else:
            # Return empty list if nothing exists
            return testset

        for diff_vec in diflogenz:
            for k in range(enzlen):
                testvector = np.zeros(enzlen)
                testvector[k] = diff_vec[k]
                testset.append(testvector)

        return testset

    def calc_diff(self,df,i,j):
        # Ensure indices are integer type
        i = int(i)
        j = int(j)
        diff=df.drop("index_col",axis=1).iloc[i] - df.drop("index_col",axis=1).iloc[j]
        diff["index_col"]=df.iloc[i]["index_col"].astype(str)+df.iloc[j]["index_col"].astype(str)
        return diff
    
    def foldchange_generator(self):
        sample_concat=self.allsampledata_generator()

        if self.transform == "log":
            sample_concat_log=sample_concat.drop("index_col",axis=1).applymap(lambda x: np.log(x))
            sample_concat_log["index_col"]=sample_concat["index_col"]
            sample_concat=sample_concat_log

        sample_diffs=[]
        if self.gen_method == "allsamples_noreverse":
            for i in range(len(sample_concat)):
                for j in range(i + 1, len(sample_concat)):
                    sample_diff=self.calc_diff(sample_concat,i,j)
                    sample_diffs.append(sample_diff)

        elif self.gen_method == "allsamples_reverse":
            for i in range(len(sample_concat)):
                for j in range(len(sample_concat)):
                    if i == j:
                        continue
                    else:
                        sample_diff=self.calc_diff(sample_concat,i,j)
                        sample_diffs.append(sample_diff)
                        continue     
        
        elif self.gen_method == "allsamples_reverse_nosame":
            index_col = sample_concat["index_col"].values
            indices = np.arange(len(sample_concat))

            # Extract index pairs with different index_col for each row
            pairs = [(i, j) for i in indices for j in indices if index_col[i] != index_col[j]]

            # Calculate differences
            sample_diffs = [
                self.calc_diff(sample_concat, i, j) for i, j in pairs]

        diff_df=pd.DataFrame(sample_diffs,columns=sample_concat.columns)
        return  diff_df

    def sort_enzmet(self,enz,met):
        if self.is_exp:
            return np.concatenate([met,enz],1)
        else:
            return np.concatenate([enz,met],1)

    def cv_splitter(self,sample_diffs_df=None):
        self.def_enzmetname()
        if sample_diffs_df is None:
            sample_diffs_df=self.foldchange_generator()
        training_list, validation_list = [], []
        X,y=self.split_enzmet(sample_diffs_df)
        if self.Xstd=="All":
            # Standardize enzyme data
            X_std_np = sc.fit_transform(X)
            # Assign standardized data to original enzyme columns in sample_diffs_df
            for i, col_name in enumerate(self.enzyme_name):
                if col_name in sample_diffs_df.columns:
                    sample_diffs_df[col_name] = X_std_np[:, i]
                else:
                    print(f"Column {col_name} not found in sample_diffs_df during Xstd='All'")
        
        if self.ystd=="All":
            # Standardize metabolite data
            y_std_np = sc.fit_transform(y.to_numpy())
            # Assign standardized data to original metabolite columns in sample_diffs_df
            for i, col_name in enumerate(self.metabolite_name):
                if col_name in sample_diffs_df.columns:
                    sample_diffs_df[col_name] = y_std_np[:, i]
                else:
                    print(f"Column {col_name} not found in sample_diffs_df during ystd='All'")

        for i in range(len(self.strains.split(","))):  
            if self.transform == "log":
                if i==0:
                    validation=sample_diffs_df[sample_diffs_df["index_col"].str.contains("0.0")]
                    training=sample_diffs_df[~sample_diffs_df["index_col"].str.contains("0.0")]
                else:
                    validation=sample_diffs_df[sample_diffs_df["index_col"].str.contains(f"{i}")]
                    training=sample_diffs_df[~sample_diffs_df["index_col"].str.contains(f"{i}")]
            else:
                validation=sample_diffs_df[sample_diffs_df["index_col"].str.contains(f"{i}")]
                training=sample_diffs_df[~sample_diffs_df["index_col"].str.contains(f"{i}")]

            X_train,y_train=self.split_enzmet(training)
            X_val,y_val=self.split_enzmet(validation)
            if self.Xstd=="Each_fold":
                # Note: sc.fit should be done on X_train, and both X_train and X_val should be transformed
                sc_fold = StandardScaler() # Use new scaler for each fold
                X_train_std_np = sc_fold.fit_transform(X_train.to_numpy())
                X_val_std_np = sc_fold.transform(X_val.to_numpy())
                
                # Update enzyme columns in training DataFrame
                for idx_loop, col_name_loop in enumerate(self.enzyme_name):
                    if col_name_loop in training.columns:
                        training[col_name_loop] = X_train_std_np[:, idx_loop]
                
                # Update enzyme columns in validation DataFrame
                current_validation_enz_cols = X_val.columns
                for idx_loop, col_name_loop in enumerate(self.enzyme_name):
                    if col_name_loop in validation.columns:
                        validation[col_name_loop] = X_val_std_np[:, idx_loop]

            if self.ystd=="Each_fold":
                sc_fold_y = StandardScaler() # Use new scaler for each fold
                y_train_std_np = sc_fold_y.fit_transform(y_train.to_numpy())
                y_val_std_np = sc_fold_y.transform(y_val.to_numpy())

                for idx_loop, col_name_loop in enumerate(self.metabolite_name):
                    if col_name_loop in training.columns:
                        training[col_name_loop] = y_train_std_np[:, idx_loop]

                for idx_loop, col_name_loop in enumerate(self.metabolite_name):
                    if col_name_loop in validation.columns:
                        validation[col_name_loop] = y_val_std_np[:, idx_loop]

            training_list.append(training)
            validation_list.append(validation)

        return training_list, validation_list

    def loader_generator(self,diff_df=None):
        train_loader_list,val_loader_list=[], []
        def append_loader(X_train,y_train,X_val,y_val,train_loader_list,val_loader_list):
            train_dataset = TensorDataset(torch.from_numpy(np.array(X_train)).to(torch.float32), torch.from_numpy(np.array(y_train)).to(torch.float32))
            val_dataset = TensorDataset(torch.from_numpy(np.array(X_val)).to(torch.float32), torch.from_numpy(np.array(y_val)).to(torch.float32))
            train_loader=DataLoader(dataset=train_dataset,batch_size=self.batch_size,shuffle=True,drop_last=True)
            val_loader=DataLoader(dataset=val_dataset,batch_size=self.batch_size)
            train_loader_list.append(train_loader)
            val_loader_list.append(val_loader)  
            return train_loader_list, val_loader_list
        
        if self.CV_method=="leave_one_strain_out":
            training_list, validation_list = self.cv_splitter(diff_df)
            for i in range(len(training_list)):
                X_train,y_train=self.split_enzmet(training_list[i])
                X_val,y_val=self.split_enzmet(validation_list[i])
                train_loader_list, val_loader_list = append_loader(X_train,y_train,X_val,y_val,train_loader_list,val_loader_list)
        
        else:
            self.def_enzmetname()
            sample_diffs_df=self.foldchange_generator()
            X,y=self.split_enzmet(sample_diffs_df)
            if self.Xstd=="All":
                # Standardize enzyme data
                X_std_np = sc.fit_transform(X)
                # Assign standardized data to original enzyme columns in sample_diffs_df
                for i, col_name in enumerate(self.enzyme_name):
                    if col_name in sample_diffs_df.columns:
                        sample_diffs_df[col_name] = X_std_np[:, i]
                    else:
                        print(f"Column {col_name} not found in sample_diffs_df during Xstd='All'")
            
            if self.ystd=="All":
                # Standardize metabolite data
                y_std_np = sc.fit_transform(y.to_numpy())
                # Assign standardized data to original metabolite columns in sample_diffs_df
                for i, col_name in enumerate(self.metabolite_name):
                    if col_name in sample_diffs_df.columns:
                        sample_diffs_df[col_name] = y_std_np[:, i]
                    else:
                        print(f"Column {col_name} not found in sample_diffs_df during ystd='All'")

            X,y=self.split_enzmet(sample_diffs_df)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, train_size=self.train_size, random_state=42)
            if self.Xstd=="Each_fold":
                sc.fit(X_train)
                X_train=sc.transform(X_train)
                X_val=sc.transform(X_val)
            if self.ystd=="Each_fold":
                sc.fit(y_train)
                y_train=sc.transform(y_train)
                y_val=sc.transform(y_val)   
            train_loader_list, val_loader_list = append_loader(X_train,y_train,X_val,y_val,train_loader_list,val_loader_list)

        return train_loader_list, val_loader_list

    def loader_all(self):
        fc_df=self.foldchange_generator()
        X,y = self.split_enzmet(fc_df)
        if self.Xstd:
            X=sc.fit_transform(X)
        if self.ystd:
            y=sc.fit_transform(y)
        train_dataset = TensorDataset(torch.from_numpy(np.array(X)).to(torch.float32), torch.from_numpy(np.array(y)).to(torch.float32))
        train_loader=DataLoader(dataset=train_dataset,batch_size=self.batch_size,shuffle=True,drop_last=True)
        return train_loader

    def reaction_list(self):
        file=open(self.root_path+"/stoichiometry.txt").read()
        each_reaction = file.split('\n')
        for i in range(len(each_reaction)):
            each_reaction[i] = each_reaction[i].split(':')
        for i in range(len(each_reaction)):
            each_reaction[i][1] = each_reaction[i][1].split(' ')
        return each_reaction

    def enz_met_matrix(self,reaction_list):
        metabolite_dic={}
        for metabolite in self.metabolite_name:
            metabolite_dic[metabolite]={}
        for i in range(len(reaction_list)):
            enzyme = reaction_list[i][0]
            for met_candidate in reaction_list[i][1]:
                if met_candidate == '=':
                    continue
                else:
                    if metabolite_dic.get(met_candidate) is not None:
                        if metabolite_dic[met_candidate].get(enzyme) == None:
                            metabolite_dic[met_candidate][enzyme] = 1
                        else:
                            metabolite_dic[met_candidate][enzyme] += 1

        neighbor_matrix = [[0] * len(self.enzyme_name) for _ in range(len(self.metabolite_name))]
        total = 0
        for i in range((len(self.metabolite_name))):
            for j in range(len(self.enzyme_name)):
                if metabolite_dic[self.metabolite_name[i]].get(self.enzyme_name[j].split('.')[0]) is not None:
                    neighbor_matrix[i][j] = 1
                    total += 1
        return np.array(neighbor_matrix).T

    def met_met_matrix_eachenz(self,reaction_list):
        m_m_e_matrix = [[[0] * len(self.metabolite_name) for _ in range(len(self.metabolite_name))] for _ in range(len(self.enzyme_name))]

        for i in range(len(self.enzyme_name)):
            for j in range(len(reaction_list)):
                if self.enzyme_name[i].split('.')[0] == reaction_list[j][0]:
                    # the matrix do not consider the connection between the metabolite itself
                    for m1 in range(len(reaction_list[j][1])):
                        for m1i in range(len(self.metabolite_name)):
                            if self.metabolite_name[m1i] == reaction_list[j][1][m1]:
                                break
                        for m2 in range(len(reaction_list[j][1])):
                            for m2i in range(len(self.metabolite_name)):
                                if self.metabolite_name[m2i] == reaction_list[j][1][m2]:
                                    break
                            m_m_e_matrix[i][m1i][m2i] = 1
                            m_m_e_matrix[i][m2i][m1i] = 1
                    break    
        total = 0
        for i in range(len(m_m_e_matrix)):
            for j in range(len(m_m_e_matrix[0])):
                for k in range(len(m_m_e_matrix[0][0])):
                    if m_m_e_matrix[i][j][k] == 1:
                        total += 1
        return np.array(m_m_e_matrix)        

    def met_met_matrix(self,reaction_list):
        m_m_e=self.met_met_matrix_eachenz(reaction_list)
        m_m=np.sum(m_m_e,axis=0)
        m_m_matrix=np.zeros([len(self.metabolite_name),len(self.metabolite_name)])
        for i in range(len(self.metabolite_name)):
            for j in range(len(self.metabolite_name)):
                if m_m[i][j] !=0:
                    m_m_matrix[i][j]=1
        return m_m_matrix

    def matrices(self):
        reaction_list=self.reaction_list()
        EMmatrix=torch.Tensor(self.enz_met_matrix(reaction_list))
        MMmatrix=torch.Tensor(self.met_met_matrix(reaction_list))
        EMmatrix_rev=1-EMmatrix
        MMmatrix_rev=1-MMmatrix
        return EMmatrix,MMmatrix,EMmatrix_rev,MMmatrix_rev