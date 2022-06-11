#June 7th AWS version
#just commented out more_itertools and umap (numba issue)
#also one small edit to magic code when no data is supplied (log it)

import numpy as np
import scipy
import os
import pandas as pd
import seaborn as sns

from scipy.sparse import find, csr_matrix
import scipy.stats as stats

import magic
import palantir
import phenograph
import harmony
import bhtsne
#import umap

import matplotlib
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches

import sklearn
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

import functools
import pickle
import re
#from  more_itertools import unique_everseen
from copy import deepcopy
from textwrap import wrap 
import statsmodels


class sc:
    def __init__(self, species='mouse'):
        self.samples = []
        self.samples_names = []
        self.filtered_cells = []
        self.filtered_genes = []
        self.lib_size = None
        self.cell_ids = None
        self.sample_ids = None
        self.gene_expression = None
        self.gene_detection = None
        self.genes = None
        self.data = None
        self.data_normalized = None
        self.eig_vecs = None
        self.communities = None
                
        self.palette =  {0:"#000000",1:"#010067",2:"#D5FF00",3:"#FF0056",4:"#9E008E", 5:"#0E4CA1", 6:"#FFE502", 7:"#005F39", 8:"#00FF00",9:"#95003A",10:"#FF937E",11:"#A42400",12:"#001544",13:"#91D0CB",14:"#620E00",15:"#6B6882",16:"#0000FF",17:"#007DB5",18:"#6A826C",19:"#00AE7E",20:"#C28C9F",21:"#BE9970",22:"#008F9C",23:"#5FAD4E",24:"#FF0000",25:"#FF00F6",26:"#FF029D",27:"#683D3B",28:"#FF74A3",29:"#968AE8",30:"#98FF52",31:"#A75740",32:"#01FFFE",33:"#FFEEE8",34:"#FE8900",35:"#BDC6FF",36:"#01D0FF",37:"#BB8800",38:"#7544B1",39:"#A5FFD2",40:"#FFA6FE",41:"#774D00", 42:"#7A4782", 43:"#263400", 44:"#004754", 45:"#43002C", 46:"#B500FF", 47:"#FFB167", 48:"#FFDB66", 49:"#90FB92",50:"#7E2DD2", 51:"#BDD393", 52:"#E56FFE", 53:"#DEFF74", 54:"#00FF78", 55:"#009BFF", 56:"#006401", 57:"#0076FF", 58:"#85A900", 59:"#00B917",60:"#788231", 61:"#00FFC6", 62:"#FF6E41", 63:"#E85EBE"}
        self.palette[70] = "#FFFFFF"
        
    def load_samples(self, samples_paths_, samples_names_,verbose=False):
        
        '''
        samples_paths_: a list of samples paths to combine in this analysis
        samples_names_: a list of samples names to combine in this analysis
        '''
        
        self.sample_names = np.array(samples_names_)
        self.sample_paths = np.array(samples_paths_)
        
        self.samples = []
        for i in range(len(samples_paths_)):
            
            data = pd.read_csv(samples_paths_[i], 
                                         header=0, sep=',', 
                                         index_col=0).drop("CLUSTER",axis=1)
            

            #Handling duplicate genes within each sample individually
            tmp = open(samples_paths_[i],'r')
            genes_tmp = tmp.readline()
            genes_tmp = np.array(genes_tmp.split(",")[1:])
            genes_tmp = genes_tmp[np.sort(np.where(genes_tmp!='CLUSTER\n')[0])] #this will fail for samples straight from SEQC; works for my custom filtered samples
            dups = genes_tmp[pd.DataFrame(genes_tmp).duplicated()]
            data_genes_orig = np.array(list(data))
            if verbose:
                print(self.sample_names[i])
                print("Genes...",genes_tmp[0:10],"...")
                print("Number of genes...",genes_tmp.shape[0])
                print("Dimensions of data...",data.shape)
                print("Duplicated Genes...",dups)

            for dup in dups:
                dup_ = np.where(genes_tmp==dup)[0]
                if verbose:
                    print("Summing gene:",set(genes_tmp[dup_]))
                    print("Column names:",str(list(data.iloc[:,dup_])))
                dup_sum = data.iloc[:,dup_].sum(1)
                data = data.drop(list(data.iloc[:,dup_]),axis=1)
                data[dup] = dup_sum
                genes_tmp = np.concatenate((np.delete(genes_tmp,dup_),[dup]))
            self.samples.append(data)
            print("Genes removed: ", set(genes_tmp)-set(list(data)))
        
        
    def combine_samples(self,verbose=False):
        sample_dict = {self.sample_names[i]: self.samples[i] for i in range(len(self.samples))} #checked this gets all samples
        sample_genes = tuple([list(sample) for sample in self.samples])

        genes_all = functools.reduce(np.union1d,sample_genes)
        self.genes_all = genes_all.copy()
        if verbose:
            print("Total number of genes (union of all samples)...",len(genes_all))
        genes = functools.reduce(np.intersect1d,sample_genes)

        new_samples_allgenes,new_samples = [],[]
        for sample in self.sample_names:
            genes_me = list(sample_dict[sample])
            genes_needed = list(set(genes_all) - set(genes_me))
            if verbose:
                print("Missing genes: ", ", ".join(genes_needed))
            me = sample_dict[sample]
            me_new = pd.concat([me, pd.DataFrame(columns = genes_needed)],sort=False).fillna(0)
            new_samples.append(me_new[genes_all].copy())
            
            if verbose:
                print(sample)
                print("My genes missing from full set...",len(set(genes_me)-set(genes_all)))
                print("Full set genes missing from my genes...",len(genes_needed))
                print("My shape before adding genes needed...",me.shape)
                print("My shape after adding genes needed...",me_new.shape,me_new.shape[1]-len(genes_needed))
                print("Sum expression of my missing genes...",me_new[genes_needed].sum().sum())

        for i in range(len(self.sample_names)):
            new_samples_allgenes.append(deepcopy(new_samples[i][genes_all]))
        
        del new_samples

        full = np.vstack(new_samples_allgenes)
        cell_ids = [list(sample_dict[sample].index) for sample in self.sample_names]
        sample_ids = [np.repeat(sample,sample_dict[sample].shape[0]) for sample in self.sample_names]
             

        cell_ids = np.concatenate(cell_ids)
        sample_ids = np.concatenate(sample_ids)

        lib_size = full.sum(axis=1)
        geneExpression = (full > 0).sum(axis=0) #for each gene, how many cells is it detected in
        geneDetection = (full > 0).sum(axis=1) #for each cell, how many unique genes are detected

        self.lib_size = np.array(lib_size).copy()
        self.cell_ids = np.copy(cell_ids)
        self.sample_ids = np.copy(sample_ids)
        self.gene_expression = np.array(geneExpression).copy()
        self.gene_detection = np.array(geneDetection).copy()
        self.genes = np.array(genes_all).copy()
        self.data = pd.DataFrame(full, columns = genes_all, index=cell_ids)
        
        if verbose:
            print("Checking combined matrix matches originals...")
            for sample in self.sample_names:
                tmp = np.where(self.sample_ids==sample)[0]
                genes_me = list(self.samples[sample])
                print(sample,np.allclose(self.data.iloc[tmp][genes_me],self.samples[sample]))
    
    def filter_ribsomal(self, mitchondrial=True):

        tmp = np.array([gene[:3] != "MT-" for gene in self.genes])
        rb_suff = ['RP1-','RP2-','RP3-','RP4-','RP5-','RP6-','RP7-','RP8-','RP9-','RP10-','RP11-','RPS','RPL']
        tmp2 = np.array([gene[:3] not in rb_suff for gene in self.genes])

        if mitchondrial:
            print("Filtering Mitochondrial and Ribosomal Genes...")
            ribo = tmp * tmp2 
        else:
            print("Filtering Ribosomal Genes...")
            ribo = tmp2
        
        nonribo = np.where(ribo)[0]
        print("Filtering: ", ", ".join(self.genes[~ribo]))
                
        self.data = self.data[self.genes[nonribo]]
        self.lib_size_orig = self.lib_size.copy()
        self.lib_size = np.array(self.data.sum(1)) #do update library size after ribosomal filtering
        self.gene_expression = np.array(self.gene_expression[nonribo])
        self.gene_detection = np.array((self.data > 0).sum(axis=1))
        self.genes =np.array(self.genes)[nonribo]
        
        if self.data_normalized is not None:
            self.data_normalized = self.data_normalized[self.genes]


    def librarysize_normalize(self,scaling = None):
        full = deepcopy(self.data.values)
        if scaling is None:
            print("Global Library Size Normalizing to Median...")
            print("Median:", np.median(full.sum(1)))
            full_normalized = np.array(full) / full.sum(1)[:,None] * np.median(full.sum(1))
        else:
            print("Global Library Size Normalizing to user-defined scaling...")
            print("Scaling Factor:", scaling)
            full_normalized = np.array(full) / full.sum(1)[:,None] * float(scaling)     
        
        self.data_normalized = pd.DataFrame(full_normalized, columns = self.genes, index=self.cell_ids)
    
    
    def pca(self, npca = 1000, epsilon = .0001):
        
        if self.data_normalized is None:
            raise ValueError("No normalized data found - load and normalize your data first!")
        
        #WITH LOG TRANSFORMATION
        pc = PCA(n_components=npca, svd_solver='randomized') # fast random PCA
        self.pc_log = pc.fit_transform(np.log2(self.data_normalized+.1))
        self.pc_log_explained_var = pc.explained_variance_ratio_
        cms = np.cumsum(pc.explained_variance_ratio_)
        d1 = np.diff(pd.Series(cms).rolling(10).mean()[10:])
        d2 = np.diff(pd.Series(d1).rolling(10).mean()[10:])
        inflection_pt = np.min(np.where(np.abs(d2) < epsilon))
        
        fig,axes = plt.subplots(1,2,figsize=(10,5))
        
        print("PCA on Log-Transformed, Normalized Counts:")
        try:
            self.npca_log = int(inflection_pt)
            print("# PCs:", self.npca_log)
            print("% Variance Explained:", cms[self.npca_log])
            print(" ")
            axes[0].plot(np.cumsum(pc.explained_variance_ratio_))
            axes[0].vlines(self.npca_log,ymin=pc.explained_variance_ratio_[0],ymax=1)
            axes[0].set_title("Log Transformed")
        except:
            print("Set npca larger to get > .8 variance explained with log-transformation!")
            print(" ")
            
        
        #WITHOUT LOG TRANSFORMATION
        pc = PCA(n_components=npca, svd_solver='randomized') # fast random PCA
        self.pc = pc.fit_transform(self.data_normalized)
        self.pc_explained_var = pc.explained_variance_ratio_
        cms = np.cumsum(pc.explained_variance_ratio_)
        d1 = np.diff(pd.Series(cms).rolling(10).mean()[10:])
        d2 = np.diff(pd.Series(d1).rolling(10).mean()[10:])
        inflection_pt = np.min(np.where(np.abs(d2) < epsilon))
        self.npca = np.max([int(inflection_pt),np.min(np.where(cms>.8)[0])])
        
        print("PCA on Normalized (un-transformed) Counts:")
        print("# PCs:", self.npca)
        print("% Variance Explained:", cms[self.npca])

        
        axes[1].plot(np.cumsum(pc.explained_variance_ratio_))
        axes[1].vlines(self.npca,ymin=pc.explained_variance_ratio_[0],ymax=1)
        axes[1].set_title("Un-transformed")
            
    def diagnostics(self, log=False):
        rcParams['figure.figsize'] = (20,25)

        npca=100
        pca = PCA(n_components=npca, svd_solver='randomized') # fast random PCA
        if log:
            pc = pca.fit_transform(np.log2(self.data_normalized+.1))
        else:
            pc = pca.fit_transform(self.data_normalized)
        cat = LabelEncoder()

        #define color coding scheme
        norm = matplotlib.colors.Normalize(vmin=0, vmax=len(set(self.sample_ids)))
        sample_ids_cat = cat.fit_transform(self.sample_ids)

        cmap = cm.gist_rainbow
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        colors = m.to_rgba(sample_ids_cat)

        plt.subplot(3,2,1)
        for sample in set(self.sample_ids):
            sns.distplot(self.lib_size[np.where(self.sample_ids==sample)[0]], label=sample)
        if len(set(self.sample_ids))<8:
            plt.legend()
        plt.title("Library Size Distributions")

        plt.subplot(3,2,2)
        for sample in set(self.sample_ids):
            sns.distplot(np.log(self.lib_size[np.where(self.sample_ids==sample)[0]]+.1), label=sample)
        if len(set(self.sample_ids))<8:
            plt.legend()
        plt.title("log Library Size Distributions")


        plt.subplot(3,2,3)
        for sample in set(self.sample_ids):
            sns.distplot(np.log(self.gene_detection[np.where(self.sample_ids==sample)[0]]), label=sample)
        if len(set(self.sample_ids))<8:
            plt.legend()
        plt.title("Genes Detected Per Cell")

        plt.subplot(3,2,4)
        for sample in set(self.sample_ids):
            geneExpression1 = (np.array(self.data)[self.sample_ids==sample,:] > 0).sum(axis=0)
            sns.distplot(np.log(geneExpression1+.1), label=sample)
        if len(set(self.sample_ids))<8:
            plt.legend()
        plt.title("Cells with Gene Detected per Gene")

        plt.subplot(3,2,5)
        plt.scatter(pc[:,0],pc[:,1],c=colors)
        plt.xlabel("PC1")
        plt.ylabel("PC2")

        explained_var_log = pca.explained_variance_ratio_
        plt.subplot(3,2,6)
        plt.title(" Cumulative Proportion Variance Explained by PCA", fontsize=8)
        plt.ylabel("Cumulative % Variance", fontsize=10)
        plt.xlabel("PC", fontsize=10);
        plt.plot(np.cumsum(explained_var_log))
   
        
    def filter_cells_indices(self, cells_indices):
        '''
        cells: ids of cells to keep
        '''
        self.cell_ids = self.cell_ids[cells_indices]
        self.sample_ids = self.sample_ids[cells_indices]
        self.lib_size = self.lib_size[cells_indices]
        self.gene_detection = self.gene_detection[cells_indices]
        self.data = self.data.iloc[cells_indices]
        
        try:
            self.data_normalized = self.data_normalized.iloc[cells_indices]
        except:
            print("Warning: no normalized data!")
        
        self.pc_log = None
        self.pc_log_explained_var = None
        self.npca_log = None
        self.pc = None
        self.pc_explained_var = None
        self.npca = None
        

    def filter_genes_indices(self, genes_indices):
        '''
        genes: ids of genes to keep
        '''
        
        self.genes = np.array(self.genes)[genes_indices]

        self.data = self.data[self.genes]
        self.gene_expression = self.gene_expression[genes_indices]

        try:
            self.data_normalized = self.data_normalized[self.genes]
        except:
            print("Warning: no normalized data!")
        
        self.pc_log = None
        self.pc_log_explained_var = None
        self.npca_log = None
        self.pc = None
        self.pc_explained_var = None
        self.npca = None
    
    def run_magic(self,t=3,knn=10,data=None):
        print("Running MAGIC with t=%d ..." % t)
        
        T = self.dm_res_log['T']
        T_steps = T ** t
        
        if data is None:
            data = deepcopy(np.log2(self.data_normalized+.1))
        
        self.data_imputed = pd.DataFrame(np.dot(T_steps.todense(), data.values), 
                index=data.index, columns=data.columns)
        
            
    def tSNE(self, seed=12345, perplexity=30, theta=.5):
                      
            print("Performing tSNE on PCs...")
            if self.pc_log is not None:
                self.tsne_log = bhtsne.tsne(self.pc_log[:,:self.npca_log], rand_seed=seed, perplexity=perplexity, theta=theta)
            else:
                raise ValueError("No principal components found - Run pca first!")
                
            
    def diffusionmaps_palantir(self,ndc=20,no_eigs=None,no_eigs_log=None,knn=30): 
        
        pca_projections = pd.DataFrame(self.pc_log,index=self.data.index)
        
        res = palantir.utils.run_diffusion_maps(pca_projections, n_components=ndc, knn=knn, n_jobs=8)
                                       
        DMEigs = pd.DataFrame(res['EigenVectors'])
        DMEigVals = pd.Series(res['EigenValues'])

        eig_vals = np.ravel(DMEigVals)

        if not no_eigs:
            self.no_eigs_log = np.argsort(eig_vals[:(len(eig_vals)-1)] - eig_vals[1:])[-1] + 1
            print("# DCs based on Eigen gap:", self.no_eigs_log)
        else:
            self.no_eigs = no_eigs
            print("# DCs, User defined:", self.no_eigs)                       
        
        use_eigs = list(range(1, self.no_eigs_log))                            
        self.eig_vals_log = np.ravel(DMEigVals.values[use_eigs])
        self.dm_res_log = res
        self.eig_vecs_log = DMEigs
        self.eig_vecs_scaled_log = DMEigs.values[:, use_eigs] * (self.eig_vals_log / (1-self.eig_vals_log))
                                       
                                       
    def force_directed_layout(self,k=30):
                      
        print("Building force directed on logged data...")
        nbrs = NearestNeighbors(n_neighbors=int(k), metric='euclidean',
                                n_jobs=5).fit(self.pc_log[:,:self.npca_log])
        kNN = nbrs.kneighbors_graph(self.pc_log[:,:self.npca_log], mode='distance')

        # Adaptive k
        adaptive_k = int(np.floor(k / 3))
        nbrs = NearestNeighbors(n_neighbors=int(adaptive_k),
                                metric='euclidean', n_jobs=5).fit(self.pc_log[:,:self.npca_log])
        adaptive_std = nbrs.kneighbors_graph(self.pc_log[:,:self.npca_log], mode='distance').max(axis=1)
        adaptive_std = np.ravel(adaptive_std.todense())

        # Kernel
        x, y, dists = find(kNN)

        # X, y specific stds
        dists = dists / adaptive_std[x]
        N = self.data.shape[0]
        W = csr_matrix((np.exp(-dists), (x, y)), shape=[N, N])

        # Diffusion components
        kernel = W + W.T
                           
        self.layout_log = harmony.plot.force_directed_layout(kernel)
        
                      
def color_tSNE(tsne,colors,s=7,title='',cmap='viridis'):
    tsne = np.array(tsne)
    order = np.random.choice(range(len(colors)),size=len(colors),replace=False)
    vmin = np.percentile(colors, 0)
    vmax = np.percentile(colors, 99)


    fig = plt.scatter(np.array(tsne[:,0])[order], np.array(tsne[:,1])[order],s=s,c=np.array(colors)[order],vmin=vmin,vmax=vmax,cmap=cmap)

    plt.title(title)
    plt.colorbar()
    plt.axis('off')

def gene_plot(tsne,data,genes,s=7,title='',cmap='Spectral_r'):
    tsne = np.array(tsne)
                      
    n = len(set(genes)); max_cols = 5
    nrows = int(np.ceil(n / max_cols)); ncols = int(min((max_cols, n)))
    fig = plt.figure(figsize=[4 * ncols, 4*nrows])     
                      
    for i,gene in enumerate(np.intersect1d(genes,list(data))):
        colors = np.log(data[gene]+.1)
        vmin = np.percentile(colors, 0)
        vmax = np.percentile(colors, 99)
        order = np.random.choice(range(len(colors)),size=len(colors),replace=False)
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.scatter(np.array(tsne[:,0])[order], np.array(tsne[:,1])[order],s=s,c=np.array(colors)[order],vmin=vmin,vmax=vmax,cmap=cmap)
        ax.set_title(gene)
        ax.set_axis_off() 


def categorical_plot(tsne,cat,s=5):
    rcParams['figure.figsize'] = (15,15)
    palette =  {0:"#000000",1:"#010067",2:"#D5FF00",3:"#FF0056",4:"#9E008E", 5:"#0E4CA1", 6:"#FFE502", 7:"#005F39", 8:"#00FF00", 9:"#95003A",10:"#FF937E",
          11:"#A42400", 12:"#001544", 13:"#91D0CB",14:"#620E00",15:"#6B6882",16:"#0000FF",17:"#007DB5",18:"#6A826C",19:"#00AE7E",20:"#C28C9F",
          21:"#BE9970",22:"#008F9C",23:"#5FAD4E",24:"#FF0000", 25:"#FF00F6",26:"#FF029D",27:"#683D3B",28:"#FF74A3",29:"#968AE8",30:"#98FF52",
          31:"#A75740",32:"#01FFFE",33:"#FFEEE8",34:"#FE8900",35:"#BDC6FF",36:"#01D0FF",37:"#BB8800",38:"#7544B1",39:"#A5FFD2",40:"#FFA6FE",
          41:"#774D00", 42:"#7A4782", 43:"#263400", 44:"#004754", 45:"#43002C", 46:"#B500FF", 47:"#FFB167", 48:"#FFDB66", 49:"#90FB92",
          50:"#7E2DD2", 51:"#BDD393", 52:"#E56FFE", 53:"#DEFF74", 54:"#00FF78", 55:"#009BFF", 56:"#006401", 57:"#0076FF", 58:"#85A900", 59:"#00B917",
          60:"#788231", 61:"#00FFC6", 62:"#FF6E41", 63:"#E85EBE"}
    palette[70] = "#FFFFFF"

    
    tsne = np.array(tsne)
    colors = [palette[color] for color in cat]
    order = np.random.choice(range(len(colors)),size=len(colors),replace=False)                
    plt.scatter(np.array(tsne[:,0])[order], np.array(tsne[:,1])[order],s=s,c=np.array(colors)[order])
    plt.axis("off")
    
    patches = []
    for color in range(np.max(cat)+1):
        patches.append(mpatches.Patch(color=palette[color],label=color))
    plt.legend(handles=patches,loc=1,borderaxespad=-5)
    
       

def binary_plot(tsne,cat,s=5,colored=False):
    tsne = np.array(tsne)
                      
    n = len(set(cat)); max_cols = 5
    nrows = int(np.ceil(n / max_cols)); ncols = int(min((max_cols, n)))
    fig = plt.figure(figsize=[4 * ncols, 4*nrows])
    
    if colored:
        for i,item in enumerate(list(set(cat))):
            colors = (cat == item) * 1
            order = np.argsort(np.array(colors))
            tmp = np.array(['lavender',palette[i]])
            ax = fig.add_subplot(nrows, ncols, i+1)
            ax.scatter(np.array(tsne[:,0])[order], np.array(tsne[:,1])[order],s=s,c=tmp[np.array(colors * 1)][order])
            ax.set_title(item)
            ax.set_axis_off()   
                      
    else:
        for i,item in enumerate(list(set(cat))):
            colors = (cat == item) * 1
            order = np.argsort(np.array(colors))
            tmp = np.array(['lavender','black'])
            ax = fig.add_subplot(nrows, ncols, i+1)
            ax.scatter(np.array(tsne[:,0])[order], np.array(tsne[:,1])[order],s=s,c=tmp[np.array(colors * 1)][order])
            ax.set_title(item)
            ax.set_axis_off()   

                
                      
def k_adaptive_kernel(X, knn=30, adapative_k=10, n_jobs=1):

        # Nearest neighbor graph
        nbrs = NearestNeighbors(n_neighbors=int(knn), metric='euclidean', n_jobs=n_jobs).fit(X.T)
        kNN = nbrs.kneighbors_graph(X.T, mode='distance' ) 

        # Adaptive k
        nbrs = NearestNeighbors(n_neighbors=int(adapative_k), metric='euclidean', n_jobs=n_jobs).fit(X.T)
        adaptive_std = nbrs.kneighbors_graph(X.T, mode='distance' ).max(axis=1) 
        adaptive_std = np.ravel(adaptive_std.todense())

        # Kernel
        N = X.shape[1]
        x, y, dists = find(kNN)

        # X, y specific stds
        sigmas = (adaptive_std[x] ** 2 + adaptive_std[y] ** 2) / 2

        # dists = (dists ** 2)/(adaptive_std[x] ** 2)
        dists = dists/adaptive_std[x]
        # dists = (dists ** 2) / sigmas * 2
        W = csr_matrix( (np.exp(-dists), (x, y)), shape=[N, N] )

        return W


def save_sc(sc,file_path):
    sc.data_normalized = csr_matrix(sc.data_normalized.values)
    sc.data = csr_matrix(sc.data.values)
    sc.data_imputed=None
    
    max_bytes = 2**31 - 1

    ## write
    bytes_out = pickle.dumps(sc)
    with open(file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])
            


def load_sc(file_path):
    
    max_bytes = 2**31 - 1

    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    sc = pickle.loads(bytes_in)
    
    sc.data = pd.DataFrame(sc.data.toarray(),columns=sc.genes,index=sc.cell_ids)
    sc.data_normalized = pd.DataFrame(sc.data_normalized.toarray(),columns=sc.genes,index=sc.cell_ids)
    
    return sc                     
                                
                      
