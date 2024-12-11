# -----------------------------------------------------------------
# Librairies
# -----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import sys
import os
import argparse

from script_simulation_COFIPL_vs_SCOFIPL import parallel_Monte_Carlo

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.generation import (phasegeneration,
                            simulateCov)
from src.utility import (ToeplitzMatrix, 
                     calculateMSE)
    
if __name__ == "__main__":

    # Create an argument parser
    parser = argparse.ArgumentParser(description="MSE of the phase difference")

    # Define the arguments 
    parser.add_argument("--l", 
                        type=int, 
                        default=40, 
                        help="Number of time stamps in the time series")
    parser.add_argument("--p", 
                        type=int, 
                        default=35, 
                        help="The size of the past time series")
    parser.add_argument("--k", 
                        type=int, 
                        default=5, 
                        help="The size of the new time series")
    parser.add_argument("--rho", 
                        type=float, 
                        default=0.98, 
                        help="Correlation coefficient for Toeplitz coherence matrix")
    parser.add_argument("--n_list", 
                        type=str, 
                        default=", ".join([str(x) for x in range(35, 80, 10)]), 
                        help="List of the size of patch to use")
    parser.add_argument("--n_trials", 
                        type=int, 
                        default=1000, 
                        help="Number of Monte-Carlo Trials")
    parser.add_argument("--sampledist", 
                        type=str, 
                        default="Gaussian", 
                        help="The sample distribution")
    parser.add_argument("--iter_max_MM", 
                        type=int, 
                        default=100, 
                        help="The number of iterations of the MM algorithm")
    parser.add_argument("--Multi", 
                        type=bool, 
                        default=True, 
                        help="Parallel computing choice")
    parser.add_argument("--n_threads", 
                        type=int, 
                        default=-1, 
                        help="The number of threads")
    parser.add_argument("--maxphase", 
                        type=int, 
                        default=2, 
                        help="The maximum value of the phase")
    parser.add_argument("--phasechoice", 
                        type=str, 
                        default="linear", 
                        help="Choice of phase: 'linear', or 'random'")
    parser.add_argument("--distance",
                        type=str,
                        default="KL",
                        help="Choice of the distance: 'KL' or 'LS'")
    parser.add_argument("--plugin",
                        type=str,
                        default="SCM",
                        help="Choice of the plug-in: 'SCM', 'PO', 'SKSCM', 'SKPO', 'BWCM', 'BWPO' ")
    parser.add_argument("--beta",
                        type=float,
                        default=0.9,
                        help='The parameter of the shrinkage to identity regularization')
    parser.add_argument("--b",
                        type=float,
                        default=9,
                        help='The parameter of the tapering regularization')

    # Safely parse known arguments and ignore unknown arguments passed by Jupyter
    args, unknown = parser.parse_known_args()

    # Convert `n_list` to a list of integers
    args.n_list = [int(x) for x in args.n_list.split(",")]

    # add `parameters`
    args.parameters = [args.k, args.p, args.b, args.beta, args.iter_max_MM, args.sampledist]

    # Optional: Print parsed arguments to verify
    print("Parsed arguments:", args)

    print("MSE over size of patch simulation with parameters:")
    for key, val in vars(args).items():
        print(f"  * {key}: {val}")


    path = "/home/elhajjad/Documents/Scripts/S_COFI_PL/MSE/results/"


    true_delta = phasegeneration(args.phasechoice,args.l) # generate phase with either random or linear. for linear, define last phase is needed
    true_delta_new = true_delta[-args.k:]
    SigmaTrue = ToeplitzMatrix(args.rho, args.l)
    trueCov = simulateCov(SigmaTrue, true_delta)



    folder = str(args.sampledist)\
        +'_rho_'+str(args.rho)\
            +'_p+k=_'+str(args.l)\
                +'_n>='+str(args.n_list[0])\
                    +'_n<'+str(args.n_list[-1])\
                        +'_iterMM=_'+str(args.iter_max_MM)\
                            +str(args.distance)\
                                +'_'\
                                +str(args.plugin)
    pathout = path+folder+ '/'
    os.makedirs(pathout)

    delta_phases_MM_seq_ones = [[] for i in range(len(args.n_list))] 
    delta_phase_MM_offline = [[] for i in range(len(args.n_list))]

    for key, value in enumerate(args.n_list):
        delta_phases_MM_seq_ones[key], delta_phase_MM_offline[key]= parallel_Monte_Carlo(value, trueCov, args.parameters, args.n_trials, args.n_threads, args.Multi)
        
    MSE_delta_phases_MM_seq_ones = calculateMSE(delta_phases_MM_seq_ones, true_delta_new, args.n_trials, args.n_list)
    MSE_delta_phase_MM_offline = calculateMSE(delta_phase_MM_offline, true_delta, args.n_trials, args.n_list)

    for m in range(0, args.k):
        plt.figure()
        plt.xlabel('n')
        plt.ylabel('MSE')
        plt.plot(args.n_list, \
                 np.array(MSE_delta_phases_MM_seq_ones)[:,m],\
                    'v-',\
                        color ='purple',\
                            label = 'S-COFI-PL')
        plt.plot(args.n_list,\
                 np.array(MSE_delta_phase_MM_offline)[:,args.p+m],\
                    'v-',\
                        color ='blue', \
                            label = 'COFI-PL')
        plt.legend()
        plt.grid("True")
        # plt.title('At date '+str(p+m+1)+', Gaussian model, p+k='+str(p+k)+', rho='+str(rho))
        plt.savefig(pathout+folder+'_MSE_rho_'+str(args.rho)+'_date_'+str(args.p+m+1)+'.pdf', dpi = 400)
        plt.show()

    for m in range(0, args.k):
        plt.hist(np.array(delta_phases_MM_seq_ones)[0,:,m],\
                 bins=np.linspace(-np.pi,np.pi,70),\
                    histtype='step',\
                        edgecolor='purple',\
                            lw=1,\
                                fill= True,\
                                    alpha = 0.3,\
                                        color ='purple',\
                                            label = 'S-COFI-PL')
        plt.hist(np.array(delta_phase_MM_offline)[0,:, args.p+m],\
                 bins=np.linspace(-np.pi,np.pi,70),\
                    histtype='step',\
                        edgecolor='blue',\
                        lw=1,fill= True,\
                            alpha = 0.2,\
                                color ='blue',\
                                    label = 'COFI-PL')
        plt.axvline(x=true_delta[m+args.p],color='black', linestyle='-',label='simulated phase')
        plt.title('Histogram of phase difference at image '+str(args.p+m+1)+' with n='+str(args.n_list[0]))
        plt.legend()
        plt.xlabel('phase (in radian)')
        plt.savefig(pathout+folder+'_hist_rho_'+str(args.rho)+'_date_'+str(args.p+m+1)+'_n='+str(args.n_list[0])+'_.pdf', dpi = 400)
        plt.show()


    for m in range(0, args.k):
        plt.hist(np.array(delta_phases_MM_seq_ones)[-1,:,m],\
                 bins=np.linspace(-np.pi,np.pi,70),\
                    histtype='step',\
                        edgecolor='purple',\
                            lw=1,\
                                fill= True,\
                                    alpha = 0.3,\
                                        color ='purple',\
                                            label = 'S-COFI-PL')
        plt.hist(np.array(delta_phase_MM_offline)[-1,:, args.p+m],\
                 bins=np.linspace(-np.pi,np.pi,70),\
                    histtype='step',\
                        edgecolor='blue',\
                            lw=1,\
                                fill= True,\
                                    alpha = 0.2,\
                                        color ='blue',\
                                            label = 'COFI-PL')
        plt.axvline(x=true_delta[m+args.p],color='black', linestyle='-',label='simulated phase')
        plt.title('Histogram of phase difference at image '+str(args.p+m+1)+' with n='+str(args.n_list[-1]))
        plt.legend()
        plt.xlabel('phase (in radian)')
        plt.savefig(pathout+folder+'_hist_rho_'+str(args.rho)+'_date_'+str(args.p+m+1)+'_n='+str(args.n_list[-1])+'_.pdf', dpi = 400)
        plt.show()
