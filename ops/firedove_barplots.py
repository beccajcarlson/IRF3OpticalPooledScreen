import seaborn as sns
import matplotlib.pyplot as plt
import os
from ops.imports import *
from scipy.stats import chi2
import statsmodels.stats.multitest as mt
import math


pal = [(39/255,95/255,153/255),(80/255,176/255,220/255),
       (237/255,244/255,248/255)
       (146/255,144/255,142/255),
       (78/255,78/255,80/255),
(235/255,154/255,88/255),(87/255,126/255,130/255)]




def produce_barplot_general(df, replist, nbins, genes, feature, alpha, df_name, pal = pal, pgene = 'nontargeting', pshow = True, range_vals = 'auto', gate = None, plot_nt = True, pval_sort = False):
    if pal == 'green':
        pal = [ (0.6313725490196078, 0.7019607843137254, 0.39215686274509803), (108/255, 126/255, 51/255)]
    if pal == 'red':
        pal = [(0.6549019607843137, 0.23529411764705882, 0.27450980392156865),(109/255,0,27/255)]
    if pal == 'yellow':
        pal = [(1, 223/255, 0/255), (212/255, 175/255, 55/255)]
    else:
        pal = pal
    sns.set(font_scale = 1.5, style = 'white')
    #sns.set(font_scale = .5, style = 'white')
    df = df[df.gene.isin(genes)]

    if gate != None:
        og_shape = df.shape[0]
        print(og_shape)
        df = df.query(gate)
        print('gate applied, retained ', int(df.shape[0]/og_shape*100), '% of data')
    ## get AUCs
    def gene_dists(df,nbins, range_vals, feature):
        aucs = []
        ctrlsample = pd.concat([df.query('sgRNA == "nontargeting_0"').sample(n=3000,random_state=0),
                               df.query('sgRNA == "nontargeting_1"').sample(n=3000,random_state=0)],axis=0)
        print('number nt', ctrlsample.shape[0])
        i = 0
        print(feature)
        i += 1

        ctrl = np.cumsum(np.histogram(ctrlsample[feature],bins=nbins,range=range_vals)[0]
                       /ctrlsample[feature].shape[0])

        aucs.append(df.groupby(['gene','sgRNA', 'rep'],sort=False)[feature].apply( ### sort = True
               lambda x: np.trapz((ctrl - np.cumsum(np.histogram(x, bins=nbins,
                range=range_vals)[0]/len(x))), dx = 1) / nbins))
        aucs = pd.concat(aucs,axis=1)
        return aucs


    if range_vals == 'auto':
        range_vals = (np.percentile((df[feature][~np.isnan(df[feature])]),5e-1),
                                              np.percentile(df[feature][~np.isnan(df[feature])],100-5e-1))
        print('range for AUC: ', range_vals)
    else:
        range_vals = range_vals
    aucs = gene_dists(df, nbins = 50, range_vals = range_vals, 
                      feature = feature).reset_index()
    aucs['sgRNA_num'] = ['sg'+s.split('_')[1] for s in aucs.sgRNA]
    print('auc done')
    ###

    sglist = pd.unique(aucs.sgRNA)
    pvals = []

    if pgene == 'nontargeting':
        for sgRNA in sglist:
            pval = scipy.stats.ttest_ind(aucs.query('sgRNA == @sgRNA')[feature],
                                       aucs.query('gene == "nontargeting"')[feature])[1]

            pvals.append((sgRNA.split('_')[0],
                         pval,
                        len(replist)))
    else:
        print('comparing to ', pgene)
        for sgRNA in sglist:
            pval = scipy.stats.ttest_ind(aucs.query('sgRNA == @sgRNA')[feature],
                                       aucs.query('gene == @pgene')[feature])[1]

            pvals.append((sgRNA.split('_')[0],
                         pval,
                        len(replist)))
    pvals = pd.DataFrame(pvals,columns=['gene','pval','n_sg'])
  
    grped1 = pvals.groupby(['gene'])['pval'].agg(n_sg = lambda x: len(x))
    grped2 = pvals.groupby(['gene'])['pval'].agg(fisher_combined_chi = lambda x: -2*sum(np.log(x)))
    grped = pd.concat([grped1, grped2], axis = 1)


    unsort = pvals.groupby(['gene'], sort = False).head(1)
    grped = pd.merge(unsort[['gene']], grped, on = 'gene')

    grped['pval'] = chi2.sf(grped.fisher_combined_chi, grped.n_sg*2)
    sigflag = mt.multipletests(grped['pval'], method = 'fdr_bh', alpha = alpha)[0]
    adj_p = mt.multipletests(grped['pval'], method = 'fdr_bh', alpha = alpha)[1]
    grped['fdr_bh_adj_p'] = adj_p
    grped['significant'] = sigflag
    

    grped['pval_text'] = ['*' if p < alpha else '' for p in grped.fdr_bh_adj_p]
    grped.loc[grped.pval < 1e-2,'pval_text'] = '**'
    grped.loc[grped.pval < 1e-3,'pval_text'] = '***'
    grped.loc[grped.pval < 1e-4,'pval_text'] = '****'   
    
    print(grped)
    print('grped done')

    plt.figure(figsize = (12,8))
    sns.set(font_scale = 2, style = 'white')
    print(aucs)

    g=sns.barplot(data = aucs, x='gene', y=feature, palette = pal,  
                  hue = 'sgRNA_num', # width = .35,
               ci = 'sd')
    handles, labels = g.get_legend_handles_labels()

    g=sns.swarmplot(data = aucs, x='gene', y=feature, color = 'k', 
                  hue = 'sgRNA_num', dodge = True
               )
    g.legend(handles[:2], labels[:2], frameon=False)


    plt.ylabel('Delta Cumulative AUC')
    
    plt.xticks(rotation = 25, horizontalalignment="right")
    plt.xlabel('')
    sns.despine(top = True, right = True)
    plt.legend()

    plt.savefig('figs/%s_%s_barplot.tif'%(df_name,feature),dpi=300,bbox_inches='tight',transparent=True)
    plt.show()


    return aucs, grped
