import seaborn as sns
import matplotlib.pyplot as plt
import os
from ops.imports import *
from scipy.stats import chi2
import statsmodels.stats.multitest as mt
import math


# pal = [(167/255,60/255,70/255),(69/255,104/255,144/255),(161/255,179/255,100/255),(110/255,0/255,95/255),
# (235/255,154/255,88/255),(87/255,126/255,130/255)]

pal = [(39/255,95/255,153/255),(80/255,176/255,220/255),
       #(237/255,244/255,248/255)
       (146/255,144/255,142/255),
       (78/255,78/255,80/255),
(235/255,154/255,88/255),(87/255,126/255,130/255)]


def produce_barplot(df, replist, nbins, genes, feature, alpha, df_name, range_vals = 'auto', gate = None, plot_nt = True, pval_sort = False):
    df = df[df.gene.isin(genes)]

    if gate != None:
        og_shape = df.shape[0]


        df = df.query(gate)
        print('gate applied, retained ', int(df.shape[0]/og_shape*100), '% of data')
    ## get AUCs
    def gene_dists(df,nbins, range_vals, feature):
        aucs = []
        ctrlsample = pd.concat([df.query('sgRNA == "nontargeting_0"').sample(n=3000,random_state=0),
                               df.query('sgRNA == "nontargeting_1"').sample(n=3000,random_state=0)],axis=0)
        print(ctrlsample.shape)
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

    sglist = pd.unique(aucs.sgRNA)#.query('gene != "nontargeting"').sgRNA)
    pvals = []

    for sgRNA in sglist:
        pval = scipy.stats.ttest_ind(aucs.query('sgRNA == @sgRNA')[feature],
                                   aucs.query('gene == "nontargeting"')[feature])[1]

        pvals.append((sgRNA.split('_')[0],
                     pval,
                    len(replist)))
    pvals = pd.DataFrame(pvals,columns=['gene','pval','n_sg'])

    grped = pvals.groupby(['gene']).agg({'pval': {'n_sg':lambda x: len(x), 
                                                'fisher_combined_chi': lambda x: -2*sum(np.log(x))}})
    grped.columns = grped.columns.get_level_values(1)

    grped['pval'] = chi2.sf(grped.fisher_combined_chi, grped.n_sg*2)
    sigflag = mt.multipletests(grped['pval'], method = 'fdr_bh', alpha = alpha)[0]
    adj_p = mt.multipletests(grped['pval'], method = 'fdr_bh', alpha = alpha)[1]
    grped['fdr_bh_adj_p'] = adj_p
    grped['significant'] = sigflag

    grped['pval_text'] = ['*' if p < alpha else 'ns' for p in grped.pval]
    grped.loc[grped.pval < 1e-2,'pval_text'] = '**'
    grped.loc[grped.pval < 1e-3,'pval_text'] = '***'
    grped.loc[grped.pval < 1e-4,'pval_text'] = '****'         
    print('grped done')
    print(grped)

    plt.figure(figsize = (12,8))
    sns.set(font_scale = 1.5, style = 'white')

    if pval_sort == True:
        if plot_nt == False:
            g=sns.barplot(data = aucs.query('gene != "nontargeting"'), x='gene', y=feature, palette = pal, order = grped.query('gene != "nontargeting"').sort_values('pval').index, 
                          hue = 'sgRNA_num',
                       ci = 'sd')
        else:
            g=sns.barplot(data = aucs, x='gene', y=feature, palette = pal, order = grped.sort_values('pval').index, 
                          hue = 'sgRNA_num',
                       ci = 'sd')

    else:
        if plot_nt == False:
            g=sns.barplot(data = aucs.query('gene != "nontargeting"'), x='gene', y=feature, palette = pal, 
                          hue = 'sgRNA_num',
                       ci = 'sd')
        else:
            g=sns.barplot(data = aucs, x='gene', y=feature, palette = pal,  
                          hue = 'sgRNA_num',
                       ci = 'sd')

    sort = []
    for i in g.patches:
        sort.append(i.get_x())
    g.patches = [g.patches[i] for i in np.argsort(sort)]


    i = 0
    for p1, p2 in zip(g.patches[::2],g.patches[1::2]):
        height = math.copysign(max(np.abs(p1.get_height()),np.abs(p2.get_height())),p1.get_height())

        if plot_nt == True:
            if pval_sort == True:
                g.annotate(grped.sort_values('pval').pval_text[i], 
                           (p2.get_x(), height + np.copysign(.01,height)), 
                            ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
            else:
                # print(grped.index[i], height + np.copysign(.025,height))

                # print((p2.get_height()))
                # print((p1.get_height()))
                # print(math.copysign(max(np.abs(p1.get_height()),np.abs(p2.get_height()))*1.3,p1.get_height()))
                # print('--------')
                
                
                #print(height, height + np.copysign(.02,height))
                # g.annotate(grped.pval_text[i], 
                #            (p2.get_x(), max(p1.get_height(),p2.get_height())*1.1), 
                #             ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
                g.annotate(grped.pval_text[i],
                           (p2.get_x(), height + np.copysign(.01,height)), 
                           ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
        else:
            if pval_sort == True:
                g.annotate(grped.query('gene != "nontargeting"').sort_values('pval').pval_text[i], 
                           (p2.get_x(), height + np.copysign(.01,height)), 
                            ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
            else:
                g.annotate(grped.query('gene != "nontargeting"').pval_text[i], 
                           (p2.get_x(), height + np.copysign(.02,height)), 
                            ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
        i += 1

    plt.ylabel('Delta Cumulative AUC')
    #g.set_xticklabels(rotation=15)#, horizontalalignment='right'
    plt.xticks(rotation = 25, horizontalalignment="right")
    plt.xlabel('')
    sns.despine(top = True, right = True)
    plt.legend(frameon=False)

    plt.savefig('figs/%s_%s_barplot.tif'%(df_name,feature),dpi=300,bbox_inches='tight',transparent=True)
    plt.show()


    def plot_hist_gene_ctrl(df,geneofint,ax,feature,xlim=(0,15000),replist=replist,cumflag=False,lw=2,font_scale=1):
        histflag = False
        gene = 'nontargeting'
        sgs = list(pd.unique(df.query('gene == @gene').sgRNA))
        i = 0
        for sg in sgs[:len(replist)]:
            for rep in replist: 
                
                sns.set(font_scale = font_scale, style = 'white')
                if i == 0:
                    c = pal[2]
                if i == 1:
                    c = pal[3]
                if rep == 'rep_2':
                    i += 1
                    g=sns.distplot(df.query('(gene == @gene) & (sgRNA == @sg) & (rep == @rep)')[feature],
                                 color=c, hist=histflag, label = '%s'%('NT_' + sg[-1]), kde_kws={'cumulative': cumflag,
                                                                                  "lw": lw},ax=ax)

                else:
                    g=sns.distplot(df.query('(gene == @gene) & (sgRNA == @sg) & (rep == @rep)')[feature],
                                 color=c, hist=histflag, kde_kws={'cumulative': cumflag,
                                                                                  "lw": lw},ax=ax)
                    plt.xlabel('')
                    # plt.legend(frameon=False)


        gene = geneofint
        sgs = list(pd.unique(df.query('gene == @gene').sgRNA))
        i = 0
        for sg in sgs[:len(replist)]:
            

            for rep in replist:
                if i == 0:
                    c = pal[0]
                if i == 1:
                    c = pal[1]
                if rep == 'rep_2':
                    i += 1
                    g=sns.distplot(df.query('(gene == @gene) & (sgRNA == @sg) & (rep == @rep)')[feature],
                                 color=c, hist=histflag, label = '%s'%sg[-1], kde_kws={'cumulative': cumflag,
                                                                                  "lw": lw},ax=ax)
                    
                else:
                    g=sns.distplot(df.query('(gene == @gene) & (sgRNA == @sg) & (rep == @rep)')[feature],
                             color=c, hist=histflag, kde_kws={'cumulative': cumflag,
                                                                                  "lw": lw},ax=ax)
                
                plt.xlim(xlim)
                # plt.legend(frameon=False)
                plt.xlabel('')
                plt.title('%s'%gene)

                sns.despine(top=True,left=True,right=True,bottom=True)

                g.set(yticklabels = [], yticks = [], xticklabels = [], xticks = [])



    nrows = int(len(genes)/3)
    ncols = int(len(genes)/nrows)
    plt.figure(figsize = (40*nrows,40*ncols))
    #['ATP13A1','DDX58','MAVS','CAPN15']
    print(range_vals)
    fig, axes = plt.subplots(nrows, ncols)

    for i, ax, gene in zip(range(int(nrows*ncols)), axes.flat, genes):
        plot_hist_gene_ctrl(df, gene, ax=ax, lw = .8, font_scale = .42, cumflag = False, 
                            feature = feature, 
                            xlim = range_vals)
        ax.legend(frameon=False)
        ax.set_xlim(range_vals)
        ax.set_xlabel('')
        ax.set_title('%s'%gene)
    plt.savefig('figs/%s_%s_hist.tif'%(df_name,feature),dpi=300,bbox_inches='tight',transparent=True)
    plt.show()

        

    return aucs, grped


def produce_barplot_only(df, replist, nbins, genes, feature, alpha, df_name, range_vals = 'auto', gate = None, plot_nt = True, pval_sort = False):
    df = df[df.gene.isin(genes)]

    if gate != None:
        og_shape = df.shape[0]

        df = df.query(gate)
        print('gate applied, retained ', int(df.shape[0]/og_shape*100), '% of data')
    ## get AUCs
    def gene_dists(df,nbins, range_vals, feature):
        aucs = []
        ctrlsample = pd.concat([df.query('sgRNA == "nontargeting_0"').sample(n=3000,random_state=0),
                               df.query('sgRNA == "nontargeting_1"').sample(n=3000,random_state=0)],axis=0)
        print(ctrlsample.shape)
        i = 0
        print(feature)
        i += 1

        ctrl = np.cumsum(np.histogram(ctrlsample[feature],bins=nbins,range=range_vals)[0]
                       /ctrlsample[feature].shape[0])

        aucs.append(df.groupby(['gene','sgRNA', 'rep'])[feature].apply(
               lambda x: np.trapz((np.cumsum(np.histogram(x, bins=nbins,
                range=range_vals)[0]/len(x)) - ctrl), dx = 1) / nbins))
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

    sglist = pd.unique(aucs.sgRNA)#.query('gene != "nontargeting"').sgRNA)
    pvals = []

    for sgRNA in sglist:
        pval = scipy.stats.ttest_ind(aucs.query('sgRNA == @sgRNA')[feature],
                                   aucs.query('gene == "nontargeting"')[feature])[1]

        pvals.append((sgRNA.split('_')[0],
                     pval,
                    len(replist)))
    pvals = pd.DataFrame(pvals,columns=['gene','pval','n_sg'])

    grped = pvals.groupby(['gene']).agg({'pval': {'n_sg':lambda x: len(x), 
                                                'fisher_combined_chi': lambda x: -2*sum(np.log(x))}})
    grped.columns = grped.columns.get_level_values(1)

    grped['pval'] = chi2.sf(grped.fisher_combined_chi, grped.n_sg*2)
    sigflag = mt.multipletests(grped['pval'], method = 'fdr_bh', alpha = alpha)[0]
    adj_p = mt.multipletests(grped['pval'], method = 'fdr_bh', alpha = alpha)[1]
    grped['fdr_bh_adj_p'] = adj_p
    grped['significant'] = sigflag

    grped['pval_text'] = ['*' if p < alpha else 'ns' for p in grped.pval]
    grped.loc[grped.pval < 1e-2,'pval_text'] = '**'
    grped.loc[grped.pval < 1e-3,'pval_text'] = '***'
    grped.loc[grped.pval < 1e-4,'pval_text'] = '****'         
    print('grped done')

    plt.figure(figsize = (12,8))
    sns.set(font_scale = 1.5, style = 'white')

    if pval_sort == True:
        if plot_nt == False:
            g=sns.barplot(data = aucs.query('gene != "nontargeting"'), x='gene', y=feature, palette = pal, order = grped.query('gene != "nontargeting"').sort_values('pval').index, 
                          hue = 'sgRNA_num',
                       ci = 'sd')
        else:
            g=sns.barplot(data = aucs, x='gene', y=feature, palette = pal, order = grped.sort_values('pval').index, 
                          hue = 'sgRNA_num',
                       ci = 'sd')

    else:
        if plot_nt == False:
            g=sns.barplot(data = aucs.query('gene != "nontargeting"'), x='gene', y=feature, palette = pal, 
                          hue = 'sgRNA_num',
                       ci = 'sd')
        else:
            g=sns.barplot(data = aucs, x='gene', y=feature, palette = pal,  
                          hue = 'sgRNA_num',
                       ci = 'sd')

    sort = []
    for i in g.patches:
        sort.append(i.get_x())
    g.patches = [g.patches[i] for i in np.argsort(sort)]

    i = 0
    for p1, p2 in zip(g.patches[::2],g.patches[1::2]):
        height = math.copysign(max(np.abs(p1.get_height()),np.abs(p2.get_height())),p1.get_height())

        if plot_nt == True:
            if pval_sort == True:
                g.annotate(grped.sort_values('pval').pval_text[i], 
                           (p2.get_x(), height + np.copysign(.01,height)), 
                            ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
            else:
                # print(grped.index[i], height + np.copysign(.025,height))

                # print((p2.get_height()))
                # print((p1.get_height()))
                # print(math.copysign(max(np.abs(p1.get_height()),np.abs(p2.get_height()))*1.3,p1.get_height()))
                # print('--------')
                
                
                #print(height, height + np.copysign(.02,height))
                # g.annotate(grped.pval_text[i], 
                #            (p2.get_x(), max(p1.get_height(),p2.get_height())*1.1), 
                #             ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
                g.annotate(grped.pval_text[i],
                           (p2.get_x(), height + np.copysign(.01,height)), 
                           ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
        else:
            if pval_sort == True:
                g.annotate(grped.query('gene != "nontargeting"').sort_values('pval').pval_text[i], 
                           (p2.get_x(), height + np.copysign(.01,height)), 
                            ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
            else:
                g.annotate(grped.query('gene != "nontargeting"').pval_text[i], 
                           (p2.get_x(), height + np.copysign(.01,height)), 
                            ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
        i += 1

    plt.ylabel('Delta Cumulative AUC')
    #g.set_xticklabels(rotation=15)#, horizontalalignment='right'
    plt.xticks(rotation = 25, horizontalalignment="right")
    plt.xlabel('')
    sns.despine(top = True, right = True)
    plt.legend(frameon=False)

    plt.show()

    return aucs, grped


def produce_barplot_3reps(df, replist, nbins, genes, feature, alpha, df_name, pgene = 'nontargeting', pshow = True, range_vals = 'auto', gate = None, plot_nt = True, pval_sort = False):
    sns.set(font_scale = 1.5, style = 'white')
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
        print('number nt')
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

    sglist = pd.unique(aucs.sgRNA)#.query('gene != "nontargeting"').sgRNA)
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

    grped = pvals.groupby(['gene']).agg({'pval': {'n_sg':lambda x: len(x), 
                                                'fisher_combined_chi': lambda x: -2*sum(np.log(x))}})
    unsort = pvals.groupby(['gene'], sort = False).head(1)
    grped.columns = grped.columns.get_level_values(1)
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
    print(aucs)
    print('grped done')

    plt.figure(figsize = (12,8))
    sns.set(font_scale = 2, style = 'white')


    if pval_sort == True:
        if plot_nt == False:
            g=sns.barplot(data = aucs.query('gene != "nontargeting"'), x='gene', y=feature, palette = pal, order = grped.query('gene != "nontargeting"').sort_values('pval').index, 
                          hue = 'sgRNA_num',
                       ci = 'sd')
        else:
            g=sns.barplot(data = aucs, x='gene', y=feature, palette = pal, order = grped.sort_values('pval').index, 
                          hue = 'sgRNA_num',
                       ci = 'sd')

    else:
        if plot_nt == False:
            g=sns.barplot(data = aucs.query('gene != "nontargeting"'), x='gene', y=feature, palette = pal, 
                          hue = 'sgRNA_num',
                       ci = 'sd')
        else:
            g=sns.barplot(data = aucs, x='gene', y=feature, palette = pal,  
                          hue = 'sgRNA_num',
                       ci = 'sd')

    
    if pshow == True:
        sort = []
        for i in g.patches:
            sort.append(i.get_x())
        g.patches = [g.patches[i] for i in np.argsort(sort)]

        i = 0
        for p1, p2 in zip(g.patches[::2],g.patches[1::2]):
            height = math.copysign(max(np.abs(p1.get_height()),np.abs(p2.get_height())),p1.get_height())

            if plot_nt == True:
                if pval_sort == True:
                    g.annotate(grped.sort_values('pval').pval_text[i], 
                               (p2.get_x(), height + np.copysign(.03,height)), 
                                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
                else:
                    # print(grped.index[i], height + np.copysign(.025,height))

                    # print((p2.get_height()))
                    # print((p1.get_height()))
                    # print(math.copysign(max(np.abs(p1.get_height()),np.abs(p2.get_height()))*1.3,p1.get_height()))
                    # print('--------')
                    
                    
                    #print(height, height + np.copysign(.02,height))
                    # g.annotate(grped.pval_text[i], 
                    #            (p2.get_x(), max(p1.get_height(),p2.get_height())*1.1), 
                    #             ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
                    g.annotate(grped.pval_text[i],
                               (p2.get_x(), height + np.copysign(.02,height)), 
                               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
            else:
                if pval_sort == True:
                    g.annotate(grped.query('gene != "nontargeting"').sort_values('pval').pval_text[i], 
                               (p2.get_x(), height + np.copysign(.03,height)), 
                                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
                else:
                    g.annotate(grped.query('gene != "nontargeting"').pval_text[i], 
                               (p2.get_x(), height + np.copysign(.014,height)), 
                                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
            i += 1
    else:
        print(grped['pval'])

    #plt.ylabel('Delta Cumulative AUC')
    plt.ylabel('IRF3 Translocation Score')
    #g.set_xticklabels(rotation=15)#, horizontalalignment='right'
    plt.xticks(rotation = 25, horizontalalignment="right")
    plt.xlabel('')
    sns.despine(top = True, right = True)
    plt.legend(frameon=False)

    plt.savefig('figs/%s_%s_barplot.tif'%(df_name,feature),dpi=300,bbox_inches='tight',transparent=True)
    plt.show()


    def plot_hist_gene_ctrl_3reps(df,geneofint,ax,feature,xlim=(0,15000),replist=replist,cumflag=False,lw=2,font_scale=1):
        histflag = False
        gene = 'nontargeting'
        sgs = list(pd.unique(df.query('gene == @gene').sgRNA))
        i = 0
        for sg in sgs:

            for rep in replist: 
                #print(rep,sg)
                sns.set(font_scale = font_scale, style = 'white')
                if i == 0:
                    c = pal[2]
                if i == 1:
                    c = pal[3]
                if rep == 4:
                    i += 1
                    g=sns.distplot(df.query('(gene == @gene) & (sgRNA == @sg) & (rep == @rep)')[feature],
                                 color=c, hist=histflag, label = '%s'%('NT_' + sg[-1]), kde_kws={'cumulative': cumflag,
                                                                                  "lw": lw},ax=ax)

                else:
                    g=sns.distplot(df.query('(gene == @gene) & (sgRNA == @sg) & (rep == @rep)')[feature],
                                 color=c, hist=histflag, kde_kws={'cumulative': cumflag,
                                                                                  "lw": lw},ax=ax)
                    plt.xlabel('')
                    # plt.legend(frameon=False)


        gene = geneofint
        sgs = list(pd.unique(df.query('gene == @gene').sgRNA))
        i = 0
        for sg in sgs[:len(sgs)]:
            

            for rep in replist:
                if i == 0:
                    c = pal[0]
                if i == 1:
                    c = pal[1]
                if rep == 4:
                    i += 1
                    g=sns.distplot(df.query('(gene == @gene) & (sgRNA == @sg) & (rep == @rep)')[feature],
                                 color=c, hist=histflag, label = '%s'%sg[-1], kde_kws={'cumulative': cumflag,
                                                                                  "lw": lw},ax=ax)
                    
                else:
                    g=sns.distplot(df.query('(gene == @gene) & (sgRNA == @sg) & (rep == @rep)')[feature],
                             color=c, hist=histflag, kde_kws={'cumulative': cumflag,
                                                                                  "lw": lw},ax=ax)
                
                plt.xlim(xlim)
                # plt.legend(frameon=False)
                plt.xlabel('')
                plt.title('%s'%gene)

                sns.despine(top=True,left=True,right=True,bottom=True)

                g.set(yticklabels = [], yticks = [], xticklabels = [], xticks = [])


    print(len(genes))
    nrows = int(len(genes)/3)
    ncols = int(len(genes)/nrows)
    plt.figure(figsize = (40*nrows,40*ncols))
    print(range_vals)
    print(nrows,ncols)
    fig, axes = plt.subplots(nrows, ncols)

    for i, ax, gene in zip(range(int(nrows*ncols)), axes.flat, genes):
        print(i, ax, gene)
        plot_hist_gene_ctrl_3reps(df, gene, ax=ax, lw = .8, font_scale = .42, cumflag = False, 
                            feature = feature, 
                            xlim = range_vals)
        ax.legend(frameon=False)
        print(range_vals)
        ax.set_xlim(range_vals)
        ax.set_xlabel('')
        ax.set_title('%s'%gene)
    plt.savefig('figs/%s_%s_hist.tif'%(df_name,feature),dpi=300,bbox_inches='tight',transparent=True)
    plt.show()

        

    return aucs, grped


def produce_barplot_nosg(df, replist, nbins, genes, feature, alpha, df_name, pgene = 'nontargeting', pshow = True, range_vals = 'auto', gate = None, plot_nt = True, pval_sort = False):
    df = df[df.gene.isin(genes)]

    if gate != None:
        og_shape = df.shape[0]
        print(og_shape)
        df = df.query(gate)
        print('gate applied, retained ', int(df.shape[0]/og_shape*100), '% of data')
    ## get AUCs
    def gene_dists(df,nbins, range_vals, feature):
        aucs = []
        ctrlsample = df.query('gene == "nontargeting"').sample(n=3000,random_state=0)
        print('number nt')
        i = 0
        print(feature)
        i += 1

        ctrl = np.cumsum(np.histogram(ctrlsample[feature],bins=nbins,range=range_vals)[0]
                       /ctrlsample[feature].shape[0])

        aucs.append(df.groupby(['gene','rep'],sort=False)[feature].apply( ### sort = True
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

    print('auc done')
    print(aucs)
    ###

    pvals = []

    if pgene == 'nontargeting':
        for gene in pd.unique(aucs.gene):
            pval = scipy.stats.ttest_ind(aucs.query('gene == @gene')[feature],
                                       aucs.query('gene == "nontargeting"')[feature])[1]

            pvals.append((gene,
                         pval,
                        len(replist)))
    print(pvals)
    pvals = pd.DataFrame(pvals).iloc[:,:2]
    pvals.columns=['gene','pval']

    # grped = pvals.groupby(['gene']).agg({'pval': {'n_sg':lambda x: len(x), 
    #                                             'fisher_combined_chi': lambda x: -2*sum(np.log(x))}})
    # unsort = pvals.groupby(['gene'], sort = False).head(1)
    # grped.columns = grped.columns.get_level_values(1)
    # grped = pd.merge(unsort[['gene']], grped, on = 'gene')

    # grped['pval'] = chi2.sf(grped.fisher_combined_chi, grped.n_sg*2)
    grped = pvals
    sigflag = mt.multipletests(grped['pval'], method = 'fdr_bh', alpha = alpha)[0]
    adj_p = mt.multipletests(grped['pval'], method = 'fdr_bh', alpha = alpha)[1]
    grped['fdr_bh_adj_p'] = adj_p
    grped['significant'] = sigflag
    

    grped['pval_text'] = ['*' if p < alpha else '' for p in grped.fdr_bh_adj_p]
    grped.loc[grped.pval < 1e-2,'pval_text'] = '**'
    grped.loc[grped.pval < 1e-3,'pval_text'] = '***'
    grped.loc[grped.pval < 1e-4,'pval_text'] = '****'   
    
    print(grped)
    print(aucs)
    print('grped done')

    plt.figure(figsize = (12,8))
    sns.set(font_scale = 1.5, style = 'white')
    print(aucs)

    if pval_sort == True:
        if plot_nt == False:
            g=sns.barplot(data = aucs.query('gene != "nontargeting"'), x='gene', y=feature, palette = pal, order = grped.query('gene != "nontargeting"').sort_values('pval').index, 
                          hue = pal,
                       ci = 'sd')
        else:
            g=sns.barplot(data = aucs, x='gene', y=feature, palette = pal, order = grped.sort_values('pval').index, 
                          hue = pal[0],
                       ci = 'sd')

    else:
        if plot_nt == False:
            g=sns.barplot(data = aucs.query('gene != "nontargeting"'), x='gene', y=feature, palette = pal, 
                          hue = 'sgRNA_num',
                       ci = 'sd')
        else:
            g=sns.barplot(data = aucs, x='gene', y=feature, palette = pal,  
                         # hue = pal[:grped.shape[0]],
                       ci = 'sd')

    
    if pshow == True:
        sort = []
        for i in g.patches:
            sort.append(i.get_x())
        g.patches = [g.patches[i] for i in np.argsort(sort)]

        i = 0
        for p1, p2 in zip(g.patches[::2],g.patches[1::2]):
            height = math.copysign(max(np.abs(p1.get_height()),np.abs(p2.get_height())),p1.get_height())

            if plot_nt == True:
                if pval_sort == True:
                    g.annotate(grped.sort_values('pval').pval_text[i], 
                               (p2.get_x(), height + np.copysign(.03,height)), 
                                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
                else:
                    # print(grped.index[i], height + np.copysign(.025,height))

                    # print((p2.get_height()))
                    # print((p1.get_height()))
                    # print(math.copysign(max(np.abs(p1.get_height()),np.abs(p2.get_height()))*1.3,p1.get_height()))
                    # print('--------')
                    
                    
                    #print(height, height + np.copysign(.02,height))
                    # g.annotate(grped.pval_text[i], 
                    #            (p2.get_x(), max(p1.get_height(),p2.get_height())*1.1), 
                    #             ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
                    g.annotate(grped.pval_text[i],
                               (p2.get_x(), height + np.copysign(.02,height)), 
                               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
            else:
                if pval_sort == True:
                    g.annotate(grped.query('gene != "nontargeting"').sort_values('pval').pval_text[i], 
                               (p2.get_x(), height + np.copysign(.03,height)), 
                                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
                else:
                    g.annotate(grped.query('gene != "nontargeting"').pval_text[i], 
                               (p2.get_x(), height + np.copysign(.03,height)), 
                                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
            i += 1
    else:
        print(grped['pval'])

    plt.ylabel('Delta Cumulative AUC')
    #g.set_xticklabels(rotation=15)#, horizontalalignment='right'
    plt.xticks(rotation = 25, horizontalalignment="right")
    plt.xlabel('')
    sns.despine(top = True, right = True)
    plt.legend(frameon=False)

    plt.savefig('figs/%s_%s_barplot.tif'%(df_name,feature),dpi=300,bbox_inches='tight',transparent=True)
    plt.show()


    def plot_hist_gene_ctrl_3reps(df,geneofint,ax,feature,xlim=(0,15000),replist=replist,cumflag=False,lw=2,font_scale=1):
        histflag = False
        gene = 'nontargeting'
        sgs = list(pd.unique(df.query('gene == @gene').sgRNA))
        i = 0
        for sg in sgs:

            for rep in replist: 
                #print(rep,sg)
                sns.set(font_scale = font_scale, style = 'white')
                if i == 0:
                    c = pal[2]
                if i == 1:
                    c = pal[3]
                if rep == 4:
                    i += 1
                    g=sns.distplot(df.query('(gene == @gene) & (sgRNA == @sg) & (rep == @rep)')[feature],
                                 color=c, hist=histflag, label = '%s'%('NT_' + sg[-1]), kde_kws={'cumulative': cumflag,
                                                                                  "lw": lw},ax=ax)

                else:
                    g=sns.distplot(df.query('(gene == @gene) & (sgRNA == @sg) & (rep == @rep)')[feature],
                                 color=c, hist=histflag, kde_kws={'cumulative': cumflag,
                                                                                  "lw": lw},ax=ax)
                    plt.xlabel('')
                    # plt.legend(frameon=False)


        gene = geneofint
        sgs = list(pd.unique(df.query('gene == @gene').sgRNA))
        i = 0
        for sg in sgs[:len(sgs)]:
            

            for rep in replist:
                if i == 0:
                    c = pal[0]
                if i == 1:
                    c = pal[1]
                if rep == 4:
                    i += 1
                    g=sns.distplot(df.query('(gene == @gene) & (sgRNA == @sg) & (rep == @rep)')[feature],
                                 color=c, hist=histflag, label = '%s'%sg[-1], kde_kws={'cumulative': cumflag,
                                                                                  "lw": lw},ax=ax)
                    
                else:
                    g=sns.distplot(df.query('(gene == @gene) & (sgRNA == @sg) & (rep == @rep)')[feature],
                             color=c, hist=histflag, kde_kws={'cumulative': cumflag,
                                                                                  "lw": lw},ax=ax)
                
                plt.xlim(xlim)
                # plt.legend(frameon=False)
                plt.xlabel('')
                plt.title('%s'%gene)

                sns.despine(top=True,left=True,right=True,bottom=True)

                g.set(yticklabels = [], yticks = [], xticklabels = [], xticks = [])



    nrows = int(len(genes)/3)
    ncols = int(len(genes)/nrows)
    plt.figure(figsize = (40*nrows,40*ncols))
    #['ATP13A1','DDX58','MAVS','CAPN15']
    print(range_vals)
    fig, axes = plt.subplots(nrows, ncols)

    for i, ax, gene in zip(range(int(nrows*ncols)), axes.flat, genes):
        plot_hist_gene_ctrl_3reps(df, gene, ax=ax, lw = .8, font_scale = .42, cumflag = False, 
                            feature = feature, 
                            xlim = range_vals)
        ax.legend(frameon=False)
        print(range_vals)
        ax.set_xlim(range_vals)
        ax.set_xlabel('')
        ax.set_title('%s'%gene)
    plt.savefig('figs/%s_%s_hist.tif'%(df_name,feature),dpi=300,bbox_inches='tight',transparent=True)
    plt.show()

        

    return aucs, grped


def produce_barplot_general(df, replist, nbins, genes, feature, alpha, df_name, pal, pgene = 'nontargeting', pshow = True, range_vals = 'auto', gate = None, plot_nt = True, pval_sort = False):
    if pal == 'green':
        pal = [ (0.6313725490196078, 0.7019607843137254, 0.39215686274509803), (108/255, 126/255, 51/255)]
    if pal == 'red':
        pal = [(0.6549019607843137, 0.23529411764705882, 0.27450980392156865),(109/255,0,27/255)]
    #pal = #(235/255,154/255,88/255),(87/255,126/255,130/255)]
    sns.set(font_scale = 1.5, style = 'white')
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

    sglist = pd.unique(aucs.sgRNA)#.query('gene != "nontargeting"').sgRNA)
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
    print(pvals)
    #grped = pvals.groupby(['gene']).agg({'pval': {'n_sg':lambda x: len(x), 
    #                                            'fisher_combined_chi': lambda x: -2*sum(np.log(x))}})

    grped1 = pvals.groupby(['gene'])['pval'].agg(n_sg = lambda x: len(x))
    grped2 = pvals.groupby(['gene'])['pval'].agg(fisher_combined_chi = lambda x: -2*sum(np.log(x)))
    print(grped1)
    grped = pd.concat([grped1, grped2], axis = 1)


    unsort = pvals.groupby(['gene'], sort = False).head(1)
    #grped.columns = grped.columns.get_level_values(1)
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
    #print(aucs)
    print('grped done')

    plt.figure(figsize = (12,8))
    sns.set(font_scale = 2, style = 'white')


    if pval_sort == True:
        if plot_nt == False:
            g=sns.barplot(data = aucs.query('gene != "nontargeting"'), x='gene', y=feature, palette = pal, order = grped.query('gene != "nontargeting"').sort_values('pval').index, 
                          hue = 'sgRNA_num',
                       ci = 'sd')
        else:
            g=sns.barplot(data = aucs, x='gene', y=feature, palette = pal, order = grped.sort_values('pval').index, 
                          hue = 'sgRNA_num',
                       ci = 'sd')

    else:
        if plot_nt == False:
            g=sns.barplot(data = aucs.query('gene != "nontargeting"'), x='gene', y=feature, palette = pal, 
                          hue = 'sgRNA_num',
                       ci = 'sd')
        else:
            g=sns.barplot(data = aucs, x='gene', y=feature, palette = pal,  
                          hue = 'sgRNA_num',
                       ci = 'sd')

    
    if pshow == True:
        sort = []
        for i in g.patches:
            sort.append(i.get_x())
        g.patches = [g.patches[i] for i in np.argsort(sort)]

        i = 0
        for p1, p2 in zip(g.patches[::2],g.patches[1::2]):
            height = math.copysign(max(np.abs(p1.get_height()),np.abs(p2.get_height())),p1.get_height())

            if plot_nt == True:
                if pval_sort == True:
                    g.annotate(grped.sort_values('pval').pval_text[i], 
                               (p2.get_x(), height + np.copysign(.03,height)), 
                                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
                else:
                    # print(grped.index[i], height + np.copysign(.025,height))

                    # print((p2.get_height()))
                    # print((p1.get_height()))
                    # print(math.copysign(max(np.abs(p1.get_height()),np.abs(p2.get_height()))*1.3,p1.get_height()))
                    # print('--------')
                    
                    
                    #print(height, height + np.copysign(.02,height))
                    # g.annotate(grped.pval_text[i], 
                    #            (p2.get_x(), max(p1.get_height(),p2.get_height())*1.1), 
                    #             ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
                    g.annotate(grped.pval_text[i],
                               (p2.get_x(), height + np.copysign(.02,height)), 
                               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
            else:
                if pval_sort == True:
                    g.annotate(grped.query('gene != "nontargeting"').sort_values('pval').pval_text[i], 
                               (p2.get_x(), height + np.copysign(.03,height)), 
                                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
                else:
                    g.annotate(grped.query('gene != "nontargeting"').pval_text[i], 
                               (p2.get_x(), height + np.copysign(.014,height)), 
                                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
            i += 1
    else:
        print(grped['pval'])

    plt.ylabel('Delta Cumulative AUC')
    #plt.ylabel('IRF3 Translocation Score')
    #g.set_xticklabels(rotation=15)#, horizontalalignment='right'
    plt.xticks(rotation = 25, horizontalalignment="right")
    plt.xlabel('')
    sns.despine(top = True, right = True)
    plt.legend(frameon=False)

    plt.savefig('figs/%s_%s_barplot.tif'%(df_name,feature),dpi=300,bbox_inches='tight',transparent=True)
    plt.show()


    # def plot_hist_gene_ctrl(df,geneofint,ax,feature,xlim=(0,15000),replist=replist,cumflag=False,lw=2,font_scale=1):
    #     histflag = False
    #     gene = 'nontargeting'
    #     sgs = list(pd.unique(df.query('gene == @gene').sgRNA))
    #     i = 0
    #     for sg in sgs:

    #         for rep in replist: 
    #             #print(rep,sg)
    #             sns.set(font_scale = font_scale, style = 'white')
    #             if i == 0:
    #                 c = pal[2]
    #             if i == 1:
    #                 c = pal[3]
    #             if rep == replist[-1]:
    #                 i += 1
    #                 g=sns.distplot(df.query('(gene == @gene) & (sgRNA == @sg) & (rep == @rep)')[feature],
    #                              color=c, hist=histflag, label = '%s'%('NT_' + sg[-1]), kde_kws={'cumulative': cumflag,
    #                                                                               "lw": lw},ax=ax)

    #             else:
    #                 g=sns.distplot(df.query('(gene == @gene) & (sgRNA == @sg) & (rep == @rep)')[feature],
    #                              color=c, hist=histflag, kde_kws={'cumulative': cumflag,
    #                                                                               "lw": lw},ax=ax)
    #                 plt.xlabel('')
    #                 # plt.legend(frameon=False)


    #     gene = geneofint
    #     sgs = list(pd.unique(df.query('gene == @gene').sgRNA))
    #     i = 0
    #     for sg in sgs[:len(sgs)]:
            

    #         for rep in replist:
    #             if i == 0:
    #                 c = pal[0]
    #             if i == 1:
    #                 c = pal[1]
    #             if rep == 4:
    #                 i += 1
    #                 g=sns.distplot(df.query('(gene == @gene) & (sgRNA == @sg) & (rep == @rep)')[feature],
    #                              color=c, hist=histflag, label = '%s'%sg[-1], kde_kws={'cumulative': cumflag,
    #                                                                               "lw": lw},ax=ax)
                    
    #             else:
    #                 g=sns.distplot(df.query('(gene == @gene) & (sgRNA == @sg) & (rep == @rep)')[feature],
    #                          color=c, hist=histflag, kde_kws={'cumulative': cumflag,
    #                                                                               "lw": lw},ax=ax)
                
    #             plt.xlim(xlim)
    #             # plt.legend(frameon=False)
    #             plt.xlabel('')
    #             plt.title('%s'%gene)

    #             sns.despine(top=True,left=True,right=True,bottom=True)

    #             g.set(yticklabels = [], yticks = [], xticklabels = [], xticks = [])


    # print(len(genes))
    # nrows = int(len(genes)/3)
    # ncols = int(len(genes)/nrows)
    # plt.figure(figsize = (40*nrows,40*ncols))
    # print(range_vals)
    # print(nrows,ncols)
    # fig, axes = plt.subplots(nrows, ncols)

    # for i, ax, gene in zip(range(int(nrows*ncols)), axes.flat, genes):
    #     print(i, ax, gene)
    #     plot_hist_gene_ctrl(df, gene, ax=ax, lw = .8, font_scale = .42, cumflag = False, 
    #                         feature = feature, 
    #                         xlim = range_vals)
    #     ax.legend(frameon=False)
    #     print(range_vals)
    #     ax.set_xlim(range_vals)
    #     ax.set_xlabel('')
    #     ax.set_title('%s'%gene)
    # plt.savefig('figs/%s_%s_hist.tif'%(df_name,feature),dpi=300,bbox_inches='tight',transparent=True)
    # plt.show()

        

    return aucs, grped
