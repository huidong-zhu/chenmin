import os
import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats.distributions import chi2
from sklearn.linear_model import LinearRegression
from multipy.fdr import lsu
from collections import OrderedDict
import re
pd.set_option('display.width',None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def r_rdataframe(reaction_ID,step,Reactions):
    '''

    :param reaction_ID: Reaction_ID of GEMs for S.cerevisiae (GEMS 7.6)
    :param step:  the position of the reaction equation is being dealt with
    :param Reactions: the entire list of reactions entered into the model
    :return:
    A metabolite dataframe is printed as follows:
    Reaction_ID   Reactants_ID    Measured_Value    Km                   Km_value
    r_0466        s_0568           True             0.127921035814782     True
    r_0466        s_1207           True             0.0238522852342474    True
    r_0466        s_0803           False            NA                    False
    r_0466        s_0335           False            NA                    False

    True represents the values of reactants or Km can participate calculation following
    '''

    reactants_series = Reactions.iloc[step, 1]
    Km_series=str(Reactions.iloc[step,4])
    space_number = len(list(a for a in reactants_series if a == " "))
    reactants_dataframe = pd.DataFrame(np.random.rand((space_number + 1) * 5).reshape(space_number + 1, 5),
                                    columns=['Reaction_ID', 'Reactants_ID', 'Measured_Value','Km','Km_value'])
    for b in range(0, space_number + 1):
        reactants_involved = reactants_series.split(' ', space_number)[b]
        reactants_dataframe.iloc[b, 1] = reactants_involved
        reactants_screened_name = Metabolites.loc[(Metabolites['Model_Metabolite_ID'] == reactants_involved)]
        Km_involved=Km_series.split(' ',space_number)[b]
        reactants_dataframe.iloc[b,3]=Km_involved
        if pd.isnull(reactants_screened_name['MS1'].tolist()) or np.any(reactants_screened_name == 0):
            reactants_measured_value = "False"
            reactants_dataframe.iloc[b, 2] = reactants_measured_value
        else:
            reactants_measured_value = "True"
            reactants_dataframe.iloc[b, 2] = reactants_measured_value
        if Km_involved=='nan' or Km_involved=='0' or Km_involved=='NA':
            Km_value="False"
            reactants_dataframe.iloc[b, 4] = Km_value
        else:
            Km_value="True"
            reactants_dataframe.iloc[b, 4] = Km_value
    reactants_dataframe['Reaction_ID'] = reaction_ID
    return reactants_dataframe

def r_edataframe(step,Reactions,reactants_dataframe,reaction_ID):
    '''

    :param step:      the position of the reaction equation is being dealt with
    :param Reactions:  the entire list of reactions entered into the model
    :param reactants_dataframe: metabolite dataframe including reaction_ID, reactants, and Km
    :param reaction_ID:    Reaction_ID of GEMs for S.cerevisiae (GEMS 7.6)
    :return:
    A enzyme dataframe is printed as follows:
    Reaction_ID   Enzyme_ID    Measured_Value
    r_0466        YNL241C            True
    r_0466        YGR248W            False
    r_0466        YHR163W            True
    True represents the values of enzymes can participate calculation following
    '''
    if ("True" in list(reactants_dataframe['Measured_Value']) and "True" in list(reactants_dataframe['Km_value'])):
        # Proceed to the next step while ensuring that all metabolites can be detected
        enzyme_ID = Reactions.iloc[step, 2]
        space_number_2 = len(list(a for a in enzyme_ID if a == " "))
        enzyme_dataframe = pd.DataFrame(np.random.rand((space_number_2 + 1) * 3).reshape(space_number_2 + 1, 3),
                                    columns=['Reaction_ID', 'Enzyme_ID', 'Measured_Value'])
        enzyme_primary_matrix = np.array([0])
        enzyme_primary_measured_matrix = np.array([0])
        for c in range(0, space_number_2 + 1):
            enzyme_involved = enzyme_ID.split(' ', space_number_2)[c]
            enzyme_second_matrix = np.array([enzyme_involved])
            enzyme_primary_matrix = np.append(enzyme_primary_matrix, enzyme_second_matrix, axis=0)
            enzyme_second_measured_matrix = Enzymes.loc[(Enzymes['Gene'] == enzyme_involved)]

            if pd.isnull(enzyme_second_measured_matrix['MS1'].tolist()) or np.any(
                    enzyme_second_measured_matrix == 0) or enzyme_second_measured_matrix.size == 0:
                # When the protein data including 0, NA under any conditions,false is given.
                enzyme_measured_value = "False"
                enzyme_primary_measured_matrix = np.append(enzyme_primary_measured_matrix,
                                                           np.array([enzyme_measured_value]), axis=0)
            else:
                enzyme_measured_value = "True"
                enzyme_primary_measured_matrix = np.append(enzyme_primary_measured_matrix,
                                                           np.array([enzyme_measured_value]), axis=0)
        enzyme_third_matrix = np.delete(enzyme_primary_matrix, 0, 0)
        enzyme_third_measured_matrix = np.delete(enzyme_primary_measured_matrix, 0, 0)
        enzyme_dataframe['Enzyme_ID'] = enzyme_third_matrix
        reaction_second_matrix = np.full(shape=(space_number_2 + 1), fill_value=reaction_ID)
        enzyme_dataframe['Reaction_ID'] = reaction_second_matrix
        enzyme_dataframe['Measured_Value'] = enzyme_third_measured_matrix

    else:
        # If the metabolites value involved in the reaction is false,the cycle ends and next reaction will start
        print("metabolites or Km haven't been detected")
        enzyme_dataframe = pd.DataFrame(columns=['Reaction_ID', 'Enzyme_ID', 'Isoenzyme', 'Measured_Value'])
    return enzyme_dataframe



def a_dataframe(reaction_ID,proposed_enzyme_ID,Enzyme_Transform):
    """

    :param reaction_ID: Reaction_ID of GEMs for S.cerevisiae (GEMS 7.6)
    :param proposed_enzyme_ID: a list including all the enzymes whose value are true in enzyme dataframe for specific reaction_ID
    :param Enzyme_Transform: A dataframe including all the enzymes and its inhibitors and activators
    :return:
    inhibitors_dataframe as follows:
        Reaction_ID   Inhibitors_ID    Inhibitors_Value
    r_0466             s_1360            True
    r_0466             s_0423            True
    r_0466             s_1212            True

    activators_dataframe as follows:
    Reaction_ID   Activators_ID    Activators_Value
    r_0466        False            False

    True represents the values of inhibitors or activators can participate calculation following
    """
    inhibitors_dataframe = pd.DataFrame(
        columns=['Reaction_ID', 'Inhibitors_ID', 'Inhibitors_Value'])
    activitors_dataframe = pd.DataFrame(
        columns=['Reaction_ID',  'Activators_ID',  'Activators_Value'])
    inh_enzyme_first_list=list()
    act_enzyme_first_list=list()
    for d in proposed_enzyme_ID:
        a_enzyme_involved=Enzyme_Transform.loc[(Enzyme_Transform['Enzyme_ID']==d)]
        inh_involved=list(a_enzyme_involved.iloc[:,1])
        act_involved=list(a_enzyme_involved.iloc[:,2])
        if pd.isnull(inh_involved):
            inh_enzyme_first_list = inh_enzyme_first_list+list()
        else:
            inh_enzyme_first_list = inh_enzyme_first_list+inh_involved
        if pd.isnull(act_involved):
            act_enzyme_first_list = act_enzyme_first_list+list()
        else:
            act_enzyme_first_list = act_enzyme_first_list+act_involved
    if len(inh_enzyme_first_list)==0:
        inhibitors_dataframe['Inhibitors_ID']= np.array(['False'])
        inhibitors_dataframe['Inhibitors_Value'] = np.array(['False'])
        inhibitors_dataframe['Reaction_ID'] = reaction_ID
    else:
        big_inh_enzyme_first_matrix = np.array([0])
        big_inh_enzyme_value_first_matrix = np.array([0])
        for inhibitor_reg in inh_enzyme_first_list:
            space_number_3 = len(list(a for a in inhibitor_reg if a == " "))
            inh_enzyme_first_matrix = np.array([0])
            inh_enzyme_value_first_matrix = np.array([0])
            for e in range(0, space_number_3 + 1):
                a_inh_involved = inhibitor_reg.split(' ', space_number_3)[e]
                inh_enzyme_first_matrix = np.append(inh_enzyme_first_matrix, np.array([a_inh_involved]), axis=0)

                inh_screened_name = Metabolites.loc[(Metabolites['Model_Metabolite_ID'] == a_inh_involved)]
                if pd.isnull(inh_screened_name['MS1'].tolist()) or np.any(
                        inh_screened_name == 0):
                    inh_measured_value = "False"
                    inh_enzyme_value_first_matrix = np.append(inh_enzyme_value_first_matrix, np.array([inh_measured_value]),
                                                              axis=0)

                else:
                    inh_measured_value = "True"
                    inh_enzyme_value_first_matrix = np.append(inh_enzyme_value_first_matrix, np.array([inh_measured_value]),
                                                              axis=0)
            inh_enzyme_second_matrix = np.delete(inh_enzyme_first_matrix, 0, 0)
            inh_enzyme_value_second_matrix = np.delete(inh_enzyme_value_first_matrix, 0, 0)
            big_inh_enzyme_first_matrix = np.append(big_inh_enzyme_first_matrix, inh_enzyme_second_matrix, axis=0)
            big_inh_enzyme_value_first_matrix = np.append(big_inh_enzyme_value_first_matrix,
                                                              inh_enzyme_value_second_matrix, axis=0)
        big_inh_enzyme_second_matrix = np.delete(big_inh_enzyme_first_matrix, 0, 0)
        big_inh_enzyme_value_second_matrix = np.delete(big_inh_enzyme_value_first_matrix, 0, 0)
        inhibitors_dataframe['Inhibitors_ID'] = big_inh_enzyme_second_matrix
        inhibitors_dataframe['Inhibitors_Value'] =big_inh_enzyme_value_second_matrix
        inhibitors_dataframe['Reaction_ID']=reaction_ID

    if len(act_enzyme_first_list)==0:
        activitors_dataframe['Activators_ID']= np.array(['False'])
        activitors_dataframe['Activators_Value'] = np.array(['False'])
        activitors_dataframe['Reaction_ID'] = reaction_ID
    else:
        big_act_enzyme_first_matrix = np.array([0])
        big_act_enzyme_value_first_matrix = np.array([0])
        for activators_reg in act_enzyme_first_list:
            space_number_4 = len(list(a for a in activators_reg if a == " "))
            act_enzyme_first_matrix = np.array([0])
            act_enzyme_value_first_matrix = np.array([0])
            for f in range(0, space_number_4 + 1):
                a_act_involved = activators_reg.split(' ', space_number_4)[f]
                act_enzyme_first_matrix = np.append(act_enzyme_first_matrix, np.array([a_act_involved]), axis=0)

                act_screened_name = Metabolites.loc[(Metabolites['Model_Metabolite_ID'] == a_act_involved)]
                if pd.isnull(act_screened_name['MS1'].tolist()) or np.any(
                        act_screened_name == 0):
                    act_measured_value = "False"
                    act_enzyme_value_first_matrix = np.append(act_enzyme_value_first_matrix, np.array([act_measured_value]),
                                                              axis=0)

                else:
                    act_measured_value = "True"
                    act_enzyme_value_first_matrix = np.append(act_enzyme_value_first_matrix, np.array([act_measured_value]),
                                                              axis=0)
            act_enzyme_second_matrix = np.delete(act_enzyme_first_matrix, 0, 0)
            act_enzyme_value_second_matrix = np.delete(act_enzyme_value_first_matrix, 0, 0)
            big_act_enzyme_first_matrix = np.append(big_act_enzyme_first_matrix, act_enzyme_second_matrix, axis=0)
            big_act_enzyme_value_first_matrix = np.append(big_act_enzyme_value_first_matrix,
                                                              act_enzyme_value_second_matrix, axis=0)
        big_act_enzyme_second_matrix = np.delete(big_act_enzyme_first_matrix, 0, 0)
        big_act_enzyme_value_second_matrix = np.delete(big_act_enzyme_value_first_matrix, 0, 0)
        activitors_dataframe['Activators_ID'] = big_act_enzyme_second_matrix
        activitors_dataframe['Activators_Value'] = big_act_enzyme_value_second_matrix
        activitors_dataframe['Reaction_ID'] = reaction_ID
    return inhibitors_dataframe, activitors_dataframe



def r_evalue(proposed_enzyme_ID,Enzymes):
    '''

    :param proposed_enzyme_ID:a list including all the enzymes whose value are true in enzyme dataframe for specific reaction_ID
    :param Enzymes: A dataframe including all the enzymes and its value under different conditions
    :return: A enzyme abundance matrix and a enzyme abundance diagonal matrix

    '''
    enzyme_first_value_matrix = np.mat(np.zeros(10))
    for g in proposed_enzyme_ID:
        enzyme_second_value_matrix = Enzymes.loc[(Enzymes['Gene'] == g)]
        enzyme_fifth_value_matrix = np.mat(enzyme_second_value_matrix.iloc[:, 0:10])
        enzyme_first_value_matrix = np.append(enzyme_first_value_matrix,
                                                    enzyme_fifth_value_matrix, axis=0)
    enzyme_third_value_matrix = np.delete(enzyme_first_value_matrix, 0, 0)
    enzyme_forth_value_matrix = np.delete(enzyme_third_value_matrix, 0, 1)
    enzyme_calculation = (np.array(np.transpose(enzyme_forth_value_matrix.sum(axis=0)))).astype(float)
    enzyme_calculation2=np.ravel(enzyme_forth_value_matrix.sum(axis=0))
    enzyme_calculation3=(np.array(np.diag(enzyme_calculation2))).astype(float)
    return enzyme_calculation,enzyme_calculation3

def met_judge(reactants_dataframe,proposed_inhibitors_ID,proposed_activators_ID):
    '''
   determining which of the reactants belong to the pseudo allosteric modulators, and extracting it to another list,
   at the same time, those of the pseudo allosteric modulators list belong to the reactant are removed

    :param reactants_dataframe: metabolite dataframe including reaction_ID, reactants, and Km

    :param proposed_inhibitors_ID:  a list including all the inhibitors whose value are true in inhibitors_dataframe for specific reaction_ID

    :param proposed_activators_ID: a list including all the activators whose value are true in inhibitors_dataframe for specific reaction_ID

    :return: 4 lists(the first is the allosteric inhibitor of the reactants, the second is the allosteric activator
    of the reactants, the third is the allosteric inhibitor of the reactants removed from the reactants, and the fourth
    is the allosteric inhibitor removed from the reactants)

    '''
    proposed_reactants_ID1 = list(reactants_dataframe['Reactants_ID'][reactants_dataframe['Measured_Value'] == 'True'])
    proposed_reactants_ID2 = list(reactants_dataframe['Reactants_ID'][reactants_dataframe['Km_value'] == 'True'])
    reactants_measured_list=list()
    for j in  proposed_reactants_ID1:
        if j in proposed_reactants_ID2:
            reactants_measured_list=reactants_measured_list+j.split()
    reactants_inhibitors_list=list()
    reactants_noinhibitors_list=list()
    for k in reactants_measured_list:
        if k in proposed_inhibitors_ID:
            reactants_inhibitors_list=reactants_inhibitors_list+k.split()
            proposed_inhibitors_ID.remove(k)
        else:
            reactants_noinhibitors_list=reactants_noinhibitors_list+k.split()
    for l in reactants_measured_list:
        if l in proposed_activators_ID:
            proposed_activators_ID.remove(l)
    return (reactants_inhibitors_list,reactants_noinhibitors_list, proposed_inhibitors_ID,proposed_activators_ID)






def r_mvalue(reactants_list,Metabolites):
    '''

    :param reactants_list: a list including all the metabolites whose value is true.
    :param Metabolites:    a dataframe including all the metabolite concentrations.
    :return:  a matrix of metabolite concentrations.
    '''
    if len(reactants_list)==0:
        reactants_eighth_value_matrix=(np.array(np.zeros(9))).astype(float)
    else:
        reactants_first_value_matrix = np.mat(np.ones(9))
        for h in reactants_list:
            reactants_second_value_matrix = Metabolites.loc[(Metabolites['Model_Metabolite_ID'] == h)]
            reactants_third_value_matrix = np.delete(np.mat(reactants_second_value_matrix.iloc[:, 0:10]), 0,
                                                     1)
            Km = float((np.mat(reactants_dataframe['Km'][reactants_dataframe['Reactants_ID'] == h]))[0, 0])
            reactants_forth_value_matrix = reactants_third_value_matrix / Km
            reactants_first_value_matrix = np.append(reactants_first_value_matrix, reactants_forth_value_matrix,
                                                     axis=0)
        reactants_fifth_value_matrix = np.delete(reactants_first_value_matrix, 0, 0)
        reactants_sixth_value_matrix = np.log(np.array(reactants_fifth_value_matrix, dtype='float'))
        reactants_eighth_value_matrix = (np.array(np.transpose(reactants_sixth_value_matrix))).astype(float)
    return(reactants_eighth_value_matrix)



def r_fvalue(reaction_ID,Fluxes,FVA_Fluxes):
    '''

    :param reaction_ID: Reaction_ID of GEMs for S.cerevisiae (GEMS 7.6)
    :param Fluxes: a dataframe including all the flux value;
    :param FVA_Fluxes: a dataframe including all the FVA_vale;
    :return: a matrix of metabolite concentrations; an array of  lower bound of FVA_values;
             an array of  upper bound of FVA_values
    '''
    Flux_first_value_matrix = Fluxes.loc[(Fluxes['Model_Reaction_ID'] == reaction_ID)]
    Flux_second_value_matrix = np.delete(np.mat(Flux_first_value_matrix), 0, 1)
    Flux_forth_value_matrix = (np.array(np.transpose(Flux_second_value_matrix))).astype(float)
    FVA_Fluxes_first_value_matrix = FVA_Fluxes.loc[FVA_Fluxes['Model_Reaction_ID'] == reaction_ID]
    FVA_Fluxes_second_value_matrix = np.delete(np.mat(FVA_Fluxes_first_value_matrix), 0, 1)
    j_obs_low1 = (FVA_Fluxes_second_value_matrix[:, 0:9]).astype(float)
    j_obs_up1 = (FVA_Fluxes_second_value_matrix[:, 9:18]).astype(float)
    return Flux_forth_value_matrix,  j_obs_low1, j_obs_up1


def thermodynamic(reaction_ID,Thermodynamic_data):
    '''
    :param reaction_ID: Reaction_ID of GEMs for S.cerevisiae (GEMS 7.6);
    :param Thermodynamic_data: a dataframe including all the thermodynamics data;
    :return: a matrix of thermodynamics data;
    '''
    thermodynamic_first_value_matrix=Thermodynamic_data.loc[(Thermodynamic_data['Model_Reaction_ID'] == reaction_ID)]
    thermodynamic_second_value_matrix=(np.array(np.delete(np.mat(thermodynamic_first_value_matrix),0,1))).astype(float)
    if np.isnan(thermodynamic_second_value_matrix).any() or np.any(thermodynamic_second_value_matrix == 0):
        deltG1=np.mat(np.ones(9))
    else:
        deltG1=np.abs(1-np.exp(thermodynamic_second_value_matrix/302.15/0.008314))
    deltG2=(np.transpose(deltG1)).astype(float)
    return deltG2

# Define the log likelihood function for being used in the bayes inference
def logp(ux,lx,mu,sigma):
    cdf_up = pm.math.exp(pm.Normal.dist(mu,sigma).logcdf(ux))
    cdf_low = pm.math.exp(pm.Normal.dist(mu,sigma).logcdf(lx))
    return pm.math.log(cdf_up-cdf_low)-pm.math.log(ux-lx)


def r_avalue(allosteric_value_list,Metabolites):
    '''

    :param allosteric_value_list: a list including all the allosteric modulators whose value is true
    :param Metabolites: a dataframe including all the metabolite concentrations.
    :return: a matrix of allosteric modulators concentrations
    '''
    allosteric_first_value_matrix = np.mat(np.ones(9))
    for i in allosteric_value_list:
        allosteric_second_value_matrix = Metabolites.loc[(Metabolites['Model_Metabolite_ID'] == i)]
        allosteric_third_value_matrix = np.delete(np.mat(allosteric_second_value_matrix.iloc[:, 0:10]), 0,
                                                  1)
        allosteric_first_value_matrix = np.append(allosteric_first_value_matrix, allosteric_third_value_matrix,
                                                  axis=0)

    allosteric_fifth_value_matrix = np.delete(allosteric_first_value_matrix, 0, 0)
    allosteric_sixth_value_matrix = np.log(np.array(allosteric_fifth_value_matrix, dtype='float'))
    allosteric_eighth_value_matrix = (np.array(np.transpose(allosteric_sixth_value_matrix))).astype(float)
    return allosteric_eighth_value_matrix



def r_savalue(allosteric_regulators,Metabolites):
    '''

    :param allosteric_regulators:  a allosteric modulator;
    :param Metabolites:  a dataframe including all the metabolite concentrations;
    :return: a matrix of a allosteric modulator concentrations;
    '''
    allosteric_regulators_matrix1=Metabolites.loc[(Metabolites['Model_Metabolite_ID'] == allosteric_regulators)]
    allosteric_regulators_matrix2 = np.delete(np.mat(allosteric_regulators_matrix1.iloc[:, 0:10]), 0,
                                              1)
    allosteric_regulators_matrix3 = np.log(np.array(allosteric_regulators_matrix2, dtype='float'))
    allosteric_regulators_matrix6=(np.array(np.transpose(allosteric_regulators_matrix3))).astype(float)
    return allosteric_regulators_matrix6




def MCMC_NNLS(deltG,flux_value,enzyme_value,enzyme_diagmatrix,reactants_noinhibitors_value,reactants_inhibitors_value,j_obs_low,j_obs_up,pesudo_proposed_allosteric_ID,proposed_inhibitors_ID_1, proposed_activators_ID_1,excel,pesudo_allosteric_list, pesudo_inhibitors_list, pesudo_activators_list,reactants_inhibitors_list,reactants_noinhibitors_list, best_allosteric = None, lowermodel = None):

    if lowermodel is None:  #it means this is the first round
        models, traces, loos = OrderedDict(), OrderedDict(), OrderedDict()
        compareDict, nameConvDict = dict(), dict()
        try:
            with pm.Model() as models['reactants']:
                flux_observed = np.array(np.log(flux_value) - np.log(enzyme_value) - np.log(deltG)).astype(float)
                ln_j_obs_low = np.transpose(np.log(np.transpose(j_obs_low)) - np.log(enzyme_value) - np.log(deltG))
                ln_j_obs_up = np.transpose(np.log(np.transpose(j_obs_up)) - np.log(enzyme_value) - np.log(deltG))
                dataframe1 = pd.DataFrame(flux_observed)
                reactants_list_lenth=len(reactants_inhibitors_list)+len(reactants_noinhibitors_list)
                if np.all(reactants_noinhibitors_value == 0):
                    reactants_noinhibitors_multi_kinetic_order=(np.array(np.zeros(9))).astype(float)
                else:
                    reactants_noinhibitors_kinetic_order = pm.Uniform('noinhibitors_alpha', lower=0, upper=5,
                                                                      shape=reactants_noinhibitors_value.shape[1])
                    reactants_noinhibitors_multi_kinetic_order = pm.math.dot(reactants_noinhibitors_value,reactants_noinhibitors_kinetic_order)
                if np.all(reactants_inhibitors_value==0):
                    reactants_inhibitors_multi_kinetic_order = (np.array(np.zeros(9))).astype(float)
                else:
                    reactants_inhibitors_kinetic_order = pm.Uniform('inhibitors_alpha', lower=-5, upper=0,
                                                                      shape=reactants_inhibitors_value.shape[1])
                    reactants_inhibitors_multi_kinetic_order = pm.math.dot(reactants_inhibitors_value,
                                                                             reactants_inhibitors_kinetic_order)
                pesudo_Kcatmin = (np.array(np.max(np.log(np.transpose(j_obs_up)) - np.log(enzyme_value)))).astype(float)
                print(pesudo_Kcatmin)
                ln_kcat = pm.Uniform('ln_kcat', lower=pesudo_Kcatmin, upper=2.3026 + pesudo_Kcatmin)
                flux_predicted = pm.Deterministic('flux_P', ln_kcat +reactants_noinhibitors_multi_kinetic_order+ reactants_inhibitors_multi_kinetic_order)
                flux_mean_squared_error = pm.math.sqrt(((flux_observed - flux_predicted) ** 2).sum(
                    axis=0) / pm.math.abs_(reactants_noinhibitors_value.shape[0]))
                flux_likelihood = pm.Normal.dist(flux_predicted, flux_mean_squared_error)
                j_obs = pm.DensityDist('j_obs', logp,
                                       observed={'ux': ln_j_obs_up, 'lx': ln_j_obs_low, 'mu': flux_predicted,
                                                 'sigma': flux_mean_squared_error},
                                       random=flux_likelihood.random)
                traces['reactants'] = pm.sample(10000, tune=50000, cores=1,
                                                start=pm.find_MAP(fmin=optimize.fmin_powell),
                                                progressbar=False)
                traceplot = pm.summary(traces['reactants'])
                print(traceplot)
                loos['reactants'] = pm.loo(traces['reactants'], models['reactants'])
                kvals = loos['reactants'].pareto_k.values
                dataframe2 = pd.DataFrame(traceplot)
                pm.traceplot(traces['reactants'])
                plt.savefig(
                    'debug/Bayesinference/Reaction/parameters/' + reaction_ID + '/reactants_coefficient.eps',
                    dpi=600, format='eps')
                plt.close('all')
                flux_posterior_distribution_matrix = np.mat(
                    traceplot.iloc[
                    reactants_list_lenth + 1:reactants_list_lenth + 1 + reactants_noinhibitors_value.shape[0],
                    0])
                print(flux_posterior_distribution_matrix)
                dataframe4 = pd.DataFrame(flux_posterior_distribution_matrix)
                flux_observed_matrix1 = np.squeeze(flux_observed)
                fit_model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
                fit_model.fit(np.transpose(flux_posterior_distribution_matrix), flux_observed_matrix1)
                fit_model_coefficient = np.mat(
                    fit_model.score(np.transpose(flux_posterior_distribution_matrix), flux_observed_matrix1))
                dataframe5 = pd.DataFrame(fit_model_coefficient)
                dataframe6 = pd.DataFrame(fit_model.coef_)
                fit_model_predicted = fit_model.predict(np.transpose(flux_posterior_distribution_matrix))
                font = {'family': 'Arial', 'weight': 'normal', 'size': 15, }
                plt.scatter(np.transpose(flux_posterior_distribution_matrix).tolist(),
                            flux_observed_matrix1.tolist(), c='g', marker='o', s=40)

                plt.plot(np.transpose(flux_posterior_distribution_matrix.tolist()), fit_model_predicted, c='r')
                plt.yticks(fontproperties='Arial', size=15)
                plt.xticks(fontproperties='Arial', size=15)
                plt.tick_params(width=2, direction='in')
                plt.xlabel("flux_posterior_distribution_value", font)
                plt.ylabel("Flux_observed", font)
                plt.savefig(
                    'debug/Bayesinference/Reaction/parameters/' + reaction_ID + '/reactants-fitting_coefficient.eps',
                    dpi=600, format='eps')
                plt.close('all')
                dataframe7 = pd.DataFrame(kvals)

                writer = pd.ExcelWriter(
                    'debug/Bayesinference/Reaction/parameters/' + reaction_ID + '/reactants_coefficient.xlsx')
                dataframe1.to_excel(writer, 'Flux_observed')
                dataframe2.to_excel(writer, 'reactants_coefficient_summary')
                dataframe4.to_excel(writer, 'flux_posterior_distribution_data')
                dataframe5.to_excel(writer, 'fitting_coefficient')
                dataframe6.to_excel(writer, 'fitting_slope')
                dataframe7.to_excel(writer, 'kvals')
                writer.save()

        except RuntimeError:
            print('something error has happened, the program will start from next reference')
            return None, None

        compareDict[models['reactants']] = traces['reactants']
        nameConvDict[models['reactants']] = 'reactants'
        compRst = pm.compare(compareDict)
        best_md_loc = compRst.index[compRst['rank'] == 0][0]
        best_allosteric.append(nameConvDict[best_md_loc])
        best_tc_loc = traces[nameConvDict[best_md_loc]]
        best_md = (best_md_loc, best_tc_loc)
        return MCMC_NNLS(deltG, flux_value, enzyme_value, enzyme_diagmatrix, reactants_noinhibitors_value,reactants_inhibitors_value, j_obs_low, j_obs_up,
                         pesudo_proposed_allosteric_ID, proposed_inhibitors_ID_1, proposed_activators_ID_1, excel,
                         pesudo_allosteric_list, pesudo_inhibitors_list, pesudo_activators_list,reactants_inhibitors_list,reactants_noinhibitors_list, best_allosteric,
                         best_md)

    else:
        assert best_allosteric
        models, traces, loos = OrderedDict(), OrderedDict(), OrderedDict()
        compareDict, nameConvDict = dict(), dict()
        for pesudo_allosteric_regulators in pesudo_proposed_allosteric_ID:
            print(pesudo_allosteric_regulators)
            try:
                with pm.Model() as models[pesudo_allosteric_regulators]:
                    flux_observed = np.array(np.log(flux_value) - np.log(enzyme_value) - np.log(deltG)).astype(float)
                    ln_j_obs_low = np.transpose(np.log(np.transpose(j_obs_low)) - np.log(enzyme_value) - np.log(deltG))
                    ln_j_obs_up = np.transpose(np.log(np.transpose(j_obs_up)) - np.log(enzyme_value) - np.log(deltG))
                    dataframe1 = pd.DataFrame(flux_observed)
                    reactants_list_lenth = len(reactants_inhibitors_list) + len(reactants_noinhibitors_list)
                    if np.all(reactants_noinhibitors_value == 0):
                        reactants_noinhibitors_multi_kinetic_order = (np.array(np.zeros(9))).astype(float)
                    else:
                        reactants_noinhibitors_kinetic_order = pm.Uniform('noinhibitors_alpha', lower=0, upper=5,
                                                                          shape=reactants_noinhibitors_value.shape[1])
                        reactants_noinhibitors_multi_kinetic_order = pm.math.dot(reactants_noinhibitors_value,
                                                                                 reactants_noinhibitors_kinetic_order)
                    if np.all(reactants_inhibitors_value == 0):
                        reactants_inhibitors_multi_kinetic_order = (np.array(np.zeros(9))).astype(float)
                    else:
                        reactants_inhibitors_kinetic_order = pm.Uniform('inhibitors_alpha', lower=-5, upper=0,
                                                                        shape=reactants_inhibitors_value.shape[1])
                        reactants_inhibitors_multi_kinetic_order = pm.math.dot(reactants_inhibitors_value,
                                                                               reactants_inhibitors_kinetic_order)
                    rea_km_ko_value = reactants_noinhibitors_multi_kinetic_order+ reactants_inhibitors_multi_kinetic_order
                    current_pesudo_allosteric_value_matrix = r_savalue(pesudo_allosteric_regulators, Metabolites)
                    current_pesudo_allosteric_median = np.median(current_pesudo_allosteric_value_matrix, axis=0)
                    current_log_Km_value = pm.Uniform(pesudo_allosteric_regulators + '_c_log_Km',
                                                      lower=-15 + current_pesudo_allosteric_median,
                                                      upper=15 + current_pesudo_allosteric_median,
                                                      shape=current_pesudo_allosteric_value_matrix.shape[1])
                    if pesudo_allosteric_regulators in proposed_inhibitors_ID and pesudo_allosteric_regulators in proposed_activators_ID:
                        current_kinetic_order = pm.Uniform(pesudo_allosteric_regulators + '_c_alpha', lower=-5, upper=5,
                                                           shape=current_pesudo_allosteric_value_matrix.shape[1])
                    elif pesudo_allosteric_regulators in proposed_inhibitors_ID and pesudo_allosteric_regulators not in proposed_activators_ID:
                        current_kinetic_order = pm.Uniform(pesudo_allosteric_regulators + '_c_alpha', lower=-5, upper=0,
                                                           shape=current_pesudo_allosteric_value_matrix.shape[1])

                    elif pesudo_allosteric_regulators not in proposed_inhibitors_ID and pesudo_allosteric_regulators in proposed_activators_ID:
                        current_kinetic_order = pm.Uniform(pesudo_allosteric_regulators + '_c_alpha', lower=0, upper=5,
                                                           shape=current_pesudo_allosteric_value_matrix.shape[1])

                    current_rea_km_ko_value = pm.math.dot(current_pesudo_allosteric_value_matrix - current_log_Km_value,
                                                          current_kinetic_order)

                    if len(pesudo_allosteric_list) == 0:
                        allosteric_rea_km_ko_value = (np.array(np.zeros(9))).astype(float)
                        allosteric_value_shape0 =  reactants_list_lenth + 1
                    else:
                        pesudo_allosteric_matrix = r_avalue(pesudo_allosteric_list, Metabolites)
                        pesudo_allosteric_matrix_median = np.median(pesudo_allosteric_matrix, axis=0)
                        allsoteric_log_Km_value = pm.Uniform(pesudo_allosteric_regulators + '_al_log_Km',
                                                             lower=-15 + pesudo_allosteric_matrix_median,
                                                             upper=15 + pesudo_allosteric_matrix_median,
                                                             shape=len(pesudo_allosteric_list))
                        allosteric_kinetic_order = pm.Uniform(pesudo_allosteric_regulators + '_al_alpha',
                                                              lower=-5, upper=5, shape=len(pesudo_allosteric_list))
                        allosteric_rea_km_ko_value = pm.math.dot(pesudo_allosteric_matrix - allsoteric_log_Km_value,
                                                                 allosteric_kinetic_order)
                        allosteric_value_shape0 =  reactants_list_lenth + pesudo_allosteric_matrix.shape[1]

                    if len(pesudo_inhibitors_list) == 0:
                        inhibitors_rea_km_ko_value = (np.array(np.zeros(9))).astype(float)
                        allosteric_value_shape1 = allosteric_value_shape0
                    else:
                        pesudo_inhibitors_matrix = r_avalue(pesudo_inhibitors_list, Metabolites)

                        pesudo_inhibitors_matrix_median = np.median(pesudo_inhibitors_matrix, axis=0)
                        inhibitors_log_Km_value = pm.Uniform(pesudo_allosteric_regulators + '_i_log_Km',
                                                             lower=-15 + pesudo_inhibitors_matrix_median,
                                                             upper=15 + pesudo_inhibitors_matrix_median,
                                                             shape=len(pesudo_inhibitors_list))
                        inhibitors_kinetic_order = pm.Uniform(pesudo_allosteric_regulators + '_i_alpha',
                                                              lower=-5, upper=0, shape=len(pesudo_inhibitors_list))
                        inhibitors_rea_km_ko_value = pm.math.dot(pesudo_inhibitors_matrix - inhibitors_log_Km_value,
                                                                 inhibitors_kinetic_order)
                        allosteric_value_shape1 = allosteric_value_shape0 + pesudo_inhibitors_matrix.shape[1]

                    if len(pesudo_activators_list) == 0:
                        activators_rea_km_ko_value = (np.array(np.zeros(9))).astype(float)
                        allosteric_value_shape2 = allosteric_value_shape1
                    else:
                        pesudo_activators_matrix = r_avalue(pesudo_activators_list, Metabolites)

                        pesudo_activators_matrix_median = np.median(pesudo_activators_matrix, axis=0)
                        activators_log_Km_value = pm.Uniform(pesudo_allosteric_regulators + '_ac_log_Km',
                                                             lower=-15 + pesudo_activators_matrix_median,
                                                             upper=15 + pesudo_activators_matrix_median,
                                                             shape=len(pesudo_activators_list))
                        activators_kinetic_order = pm.Uniform(pesudo_allosteric_regulators + '_ac_alpha',
                                                              lower=0, upper=5, shape=len(pesudo_activators_list))
                        activators_rea_km_ko_value = pm.math.dot(pesudo_activators_matrix - activators_log_Km_value,
                                                                 activators_kinetic_order)
                        allosteric_value_shape2 = allosteric_value_shape1 + pesudo_activators_matrix.shape[1]

                    pesudo_Kcatmin = (np.array(np.max(np.log(np.transpose(j_obs_up)) - np.log(enzyme_value)))).astype(
                        float)
                    ln_kcat = pm.Uniform('ln_kcat', lower=pesudo_Kcatmin, upper=2.3026 + pesudo_Kcatmin)

                    flux_predicted = pm.Deterministic(pesudo_allosteric_regulators + 'flux',
                                                      ln_kcat + rea_km_ko_value + current_rea_km_ko_value + allosteric_rea_km_ko_value + activators_rea_km_ko_value + inhibitors_rea_km_ko_value)

                    flux_mean_squared_error = pm.math.sqrt(((flux_observed - flux_predicted) ** 2).sum(
                        axis=0) / pm.math.abs_(reactants_noinhibitors_value.shape[0]))
                    flux_likelihood = pm.Normal.dist(flux_predicted, flux_mean_squared_error)
                    j_obs = pm.DensityDist(pesudo_allosteric_regulators + 'j_obs', logp,
                                           observed={'ux': ln_j_obs_up, 'lx': ln_j_obs_low, 'mu': flux_predicted,
                                                     'sigma': flux_mean_squared_error},
                                           random=flux_likelihood.random)
                    traces[pesudo_allosteric_regulators] = pm.sample(10000, tune=90000, cores=1,
                                                                     start=pm.find_MAP(fmin=optimize.fmin_powell),
                                                                     progressbar=False)
                    traceplot = pm.summary(traces[pesudo_allosteric_regulators])
                    print(traceplot)
                    loos[pesudo_allosteric_regulators] = pm.loo(traces[pesudo_allosteric_regulators],
                                                                models[pesudo_allosteric_regulators])
                    kvals = loos[pesudo_allosteric_regulators].pareto_k.values
                    dataframe2 = pd.DataFrame(traceplot)
                    pm.traceplot(traces[pesudo_allosteric_regulators])
                    plt.savefig(
                        'debug/Bayesinference/Reaction/parameters/' + reaction_ID + '/' + "-".join(
                            best_allosteric) + pesudo_allosteric_regulators + '_coefficient.eps',
                        dpi=600, format='eps')
                    plt.close('all')

                    flux_posterior_distribution_matrix = np.mat(
                        traceplot.iloc[
                        allosteric_value_shape2 * 2 + 1 - reactants_list_lenth:allosteric_value_shape2 * 2 + 1 +
                                                                                   reactants_noinhibitors_value.shape[
                                                                                       0] -  reactants_list_lenth,
                        0])
                    print(flux_posterior_distribution_matrix)

                    dataframe4 = pd.DataFrame(flux_posterior_distribution_matrix)
                    flux_observed_matrix1 = np.squeeze(flux_observed)
                    fit_model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
                    fit_model.fit(np.transpose(flux_posterior_distribution_matrix), flux_observed_matrix1)
                    fit_model_coefficient = np.mat(
                        fit_model.score(np.transpose(flux_posterior_distribution_matrix), flux_observed_matrix1))
                    dataframe5 = pd.DataFrame(fit_model_coefficient)
                    dataframe6 = pd.DataFrame(fit_model.coef_)
                    fit_model_predicted = fit_model.predict(np.transpose(flux_posterior_distribution_matrix))
                    font = {'family': 'Arial', 'weight': 'normal', 'size': 15, }
                    plt.scatter(np.transpose(flux_posterior_distribution_matrix).tolist(),
                                flux_observed_matrix1.tolist(), c='g', marker='o', s=40)

                    plt.plot(np.transpose(flux_posterior_distribution_matrix.tolist()), fit_model_predicted, c='r')
                    plt.yticks(fontproperties='Arial', size=15)
                    plt.xticks(fontproperties='Arial', size=15)
                    plt.tick_params(width=2, direction='in')
                    plt.xlabel("flux_posterior_distribution_value", font)
                    plt.ylabel("Flux_observed", font)
                    plt.savefig(
                        'debug/Bayesinference/Reaction/parameters/' + reaction_ID + '/' + "-".join(
                            best_allosteric) + pesudo_allosteric_regulators + '_fitting_curve.eps',
                        dpi=600, format='eps')
                    plt.close('all')
                    dataframe7 = pd.DataFrame(kvals)

                    writer = pd.ExcelWriter(
                        'debug/Bayesinference/Reaction/parameters/' + reaction_ID + '/' + "-".join(
                            best_allosteric) + pesudo_allosteric_regulators + '_coefficient.xlsx')
                    dataframe1.to_excel(writer, 'Flux_observed')
                    dataframe2.to_excel(writer, 'reactants_coefficient_summary')
                    dataframe4.to_excel(writer, 'flux_posterior_distribution_data')
                    dataframe5.to_excel(writer, 'fitting_coefficient')
                    dataframe6.to_excel(writer, 'fitting_slope')
                    dataframe7.to_excel(writer, 'kvals')
                    writer.save()

            except RuntimeError:
                print('something error has happened, the program will start from next reference')
                continue

            compareDict[models[pesudo_allosteric_regulators]] = traces[pesudo_allosteric_regulators]
            nameConvDict[models[pesudo_allosteric_regulators]] = pesudo_allosteric_regulators

        compareDict[lowermodel[0]] = lowermodel[1]
        assert compareDict
        compRst = pm.compare(compareDict)
        print(compRst)
        compRst.to_excel(excel, "-".join(best_allosteric) + 'model_compare')
        excel.save()
        best_md_loc = compRst.index[compRst['rank'] == 0][0]
        if best_md_loc == lowermodel[0]:
            print('Finally, found the best model is\033[1;31;43m', best_allosteric, '\033[0m')
            return compRst, best_allosteric
        else:
            best_tc_loc = traces[nameConvDict[best_md_loc]]
            best_md = (best_md_loc, best_tc_loc)
            best_allosteric.append(nameConvDict[best_md_loc])
            pesudo_proposed_allosteric_ID.remove(nameConvDict[best_md_loc])
            if nameConvDict[best_md_loc] in proposed_inhibitors_ID and nameConvDict[
                best_md_loc] in proposed_activators_ID:
                pesudo_allosteric_list = pesudo_allosteric_list + nameConvDict[best_md_loc].split(' ')

            elif nameConvDict[best_md_loc] in proposed_inhibitors_ID and nameConvDict[
                best_md_loc] not in proposed_activators_ID:
                pesudo_inhibitors_list = pesudo_inhibitors_list + nameConvDict[best_md_loc].split(' ')

            elif nameConvDict[best_md_loc] in proposed_activators_ID and nameConvDict[
                best_md_loc] not in proposed_inhibitors_ID:
                pesudo_activators_list = pesudo_activators_list + nameConvDict[best_md_loc].split(' ')

            return MCMC_NNLS(deltG, flux_value, enzyme_value, enzyme_diagmatrix, reactants_noinhibitors_value,reactants_inhibitors_value, j_obs_low, j_obs_up,
                             pesudo_proposed_allosteric_ID, proposed_inhibitors_ID_1, proposed_activators_ID_1, excel,
                             pesudo_allosteric_list, pesudo_inhibitors_list, pesudo_activators_list,reactants_inhibitors_list,reactants_noinhibitors_list, best_allosteric,
                             best_md)

Reactions=pd.read_excel('Hierachical regulation analysis-CCMS(openmebius).xlsx',sheet_name='Reaction')
Metabolites=pd.read_excel('Hierachical regulation analysis-CCMS(openmebius).xlsx',sheet_name='metabolites_mM')
Enzymes=pd.read_excel('Hierachical regulation analysis-CCMS(openmebius).xlsx',sheet_name='enzyme_umolgDCW_mean')
Fluxes=pd.read_excel('Hierachical regulation analysis-CCMS(openmebius).xlsx',sheet_name='Flux_mean')
FVA_Fluxes=pd.read_excel('Hierachical regulation analysis-CCMS(openmebius).xlsx',sheet_name='FVA_Fluxes_mean')
Enzyme_Transform=pd.read_excel('Hierachical regulation analysis-CCMS(openmebius).xlsx',sheet_name='enzyme transform')
Thermodynamic_data=pd.read_excel('Hierachical regulation analysis-CCMS(openmebius).xlsx',sheet_name='Thermodynamic Data')

for step in range(9,len(Reactions)):
    reaction_ID=Reactions.iloc[step,0]
    print(reaction_ID)
    reactants_dataframe=r_rdataframe(reaction_ID,step,Reactions)
    enzyme_dataframe =r_edataframe(step,Reactions,reactants_dataframe,reaction_ID)
    proposed_enzyme_ID= list(enzyme_dataframe['Enzyme_ID'][enzyme_dataframe['Measured_Value'] == 'True'])
    enzyme_value,enzyme_diagmatrix=r_evalue(proposed_enzyme_ID,Enzymes)

    flux_value, j_obs_low, j_obs_up = r_fvalue(reaction_ID, Fluxes, FVA_Fluxes)
    deltG = thermodynamic(reaction_ID,Thermodynamic_data)
    inhibitors_dataframe, activitors_dataframe = a_dataframe(reaction_ID, proposed_enzyme_ID, Enzyme_Transform)
    proposed_inhibitors_ID = list(
        inhibitors_dataframe['Inhibitors_ID'][inhibitors_dataframe['Inhibitors_Value'] == 'True'])
    proposed_activators_ID = list(
        activitors_dataframe['Activators_ID'][activitors_dataframe['Activators_Value'] == 'True'])
    reactants_inhibitors_list,reactants_noinhibitors_list, proposed_inhibitors_ID_1,proposed_activators_ID_1=met_judge(reactants_dataframe, proposed_inhibitors_ID,proposed_activators_ID)
    reactants_noinhibitors_value = r_mvalue(reactants_noinhibitors_list, Metabolites)
    reactants_inhibitors_value=r_mvalue(reactants_inhibitors_list, Metabolites)
    proposed_allosteric_ID_1 = proposed_inhibitors_ID_1 + proposed_activators_ID_1
    proposed_allosteric_ID_2 = list(set(proposed_allosteric_ID_1))
    print(flux_value)
    print(j_obs_up)
    print(j_obs_low)
    if np.all(flux_value == 0) or np.isnan(flux_value).any() or np.isnan(j_obs_up).any() or np.isnan(
            j_obs_low).any():
        print(reaction_ID, "has no detective flux, next reaction will start soon")
        continue
    else:
        if np.isnan(enzyme_value).any() or np.all(enzyme_value == 0):
            pass
        else:
            os.makedirs('debug/Bayesinference/Reaction/parameters/' + reaction_ID)
            os.makedirs('debug/Bayesinference/Reaction/statistic/' + reaction_ID)
            excel = pd.ExcelWriter(
                'debug/Bayesinference/Reaction/statistic/' + reaction_ID + '/-results_statistics .xlsx')
            best_allosteric = []
            pesudo_allosteric_list = list()
            pesudo_inhibitors_list = list()
            pesudo_activators_list = list()
            posterior_distribution = MCMC_NNLS(deltG, flux_value, enzyme_value, enzyme_diagmatrix, reactants_noinhibitors_value,reactants_inhibitors_value,
                                               j_obs_low, j_obs_up,
                                               proposed_allosteric_ID_2, proposed_inhibitors_ID_1,
                                               proposed_activators_ID_1, excel, pesudo_allosteric_list,
                                               pesudo_inhibitors_list, pesudo_activators_list, reactants_inhibitors_list,reactants_noinhibitors_list,best_allosteric,
                                               lowermodel=None)
            best_allosteric_dataframe = pd.DataFrame(best_allosteric)
            best_allosteric_dataframe.to_excel(excel, 'best_allosteric')
            excel.save()

















