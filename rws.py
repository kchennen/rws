"""Usage: rws.py [-h] (--in=<input>) (--out=<output>) [--rank=<ranks>]

This program is a Python implementation of the Random Walk with Resistance (RWS)
method for improving PPI networks described in: 

Lei, Chengwei, and Jianhua Ruan. "A novel link prediction algorithm for reconstructing 
protein-protein interaction networks by topological similarity." Bioinformatics (2012).

The original matlab code is available from the author's website 
(http://my.cs.utsa.edu/~clei/projects/RWS/RWS.html)

Options:
  --in=<input>    The input graph
  --out=<output>  The output graph
  --rank=<ranks>  The ranks of the output edges
"""
from docopt import docopt
import numpy as np
import sys


def rws(a, epsilon=None, beta=None, stop_value=None):
    """
    This is currently just a straightforward port of the
    Matlab code to Pyton.  In the future, I may make it more
    "Pythonic", as the current imlementations are a little 
    terse / dense.
    """

    nodes = a.shape[0]

    if not stop_value:
        stop_value = 1.0 / np.mean(np.sum(a, axis=1))
    if not beta:
        beta = 1.0 / np.mean(np.sum(a, axis=1)) / nodes
    if not epsilon:
        epsilon = 1.0 / np.sum(np.sum(a, axis=1)) / np.mean(np.sum(a, axis=1))

    m,n = a.shape
    aa = np.maximum(a, np.eye(m))

    bb = np.sum(aa, axis=1)
    bb = 1.0 / bb

    cc = np.outer(np.ones((m,1)),bb) * aa

    cc = cc.T


    cc[ np.isnan(cc) ] = 0

    current = np.eye(m)
    newcurrent = current.copy()

    active = np.ones((m,1))

    step = 0
    improve_count = 0
    active_nodes = 0

    improve = np.ones((n,1))
    while np.sum(active) > 0.0:
        print(step, np.sum(active), improve_count)
    	active_nodes = np.sum(active)
    	current = newcurrent

    	curr_act = np.dot(active, np.ones((1,m)))*current
    	curr_stop=(np.dot(active,np.ones((1,m)))-1)*current

    	sel = (((np.dot(current,cc)>0).astype(np.int)-(current>0).astype(np.int))==1).astype(np.int)
    	
    	
    	
    	temp = np.dot(curr_act,cc)-(sel*( np.dot((curr_act>0).astype(np.int),aa) )*beta)
    	temp = temp*(temp>0).astype(np.int)
    	newcurrent = temp


        newcurrent=temp;
        
        newcurrent=newcurrent*(newcurrent>(beta*cc)).astype(np.int)
        
        
        newcurrent=newcurrent-np.dot((current>0).astype(int),aa)*epsilon
        newcurrent=newcurrent*(newcurrent>0).astype(np.int)
     

        newcurrent=newcurrent-curr_stop;
        newcurrent=np.outer((1/np.sum(newcurrent.T, axis=0)).T,np.ones((1,nodes)))*newcurrent #normalize
        
        improve=np.sum(np.abs(current-newcurrent), axis=1)
        
        active=np.reshape( (improve>stop_value).astype(np.int), active.shape )
        
        if active_nodes == np.sum(active):
            improve_count=improve_count+1
        else:
        	improve_count=0
       
        step=step+1

        if improve_count>1:
            break

    
    return current

def corrbyrws(a):
	rws1 = rws(a)
	rwscorr = np.corrcoef(rws1.T)
	rwscorr[np.isnan(rwscorr)] = 0.0
	return rwscorr

def renet(a, edges_u_want=None):
    net = ((a+a.T)>0).astype(np.int)
    edges = np.sum(np.sum(net * np.array((np.ones(net.shape)-np.eye(net.shape[0]))), axis=0))+net.shape[1]
    print("edges",edges)
    if edges_u_want:
    	edges = edges_u_want * 2 + net.shape[1]

    fullcorr=corrbyrws(a)
    
    allvalue = np.sort(fullcorr.flatten())
    print("Cutoff = {0}".format(allvalue[-edges-1]))
    newNet = (fullcorr > (allvalue[-edges-1])).astype(np.int)
    return newNet


def test():
    m = np.array([[1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
       [1, 1, 0, 1, 0, 0, 1, 1, 1, 1],
       [1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
       [1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
       [1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
       [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
       [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
       [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]])

    renet(m)


def main(args):
    import networkx as nx

    iname = args['--in']
    oname = args['--out']
    G = nx.read_adjlist(iname, comments='#')
    print("Original network size = {0}".format(G.size()))

    A = np.array(nx.adjacency_matrix(G))
    B = renet(A)
    G2 = nx.from_numpy_matrix(B)
    
    print("New network size = {0}".format(G2.size()))

    rndict = dict(zip(G2.nodes(),G.nodes()))
    nx.relabel_nodes(G2, rndict, copy=False)

    nx.write_adjlist(G2, oname)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)

