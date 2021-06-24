import seaborn as sns
import matplotlib.pyplot as plt
import os
from ops.imports import *


pal = [(167/255,60/255,70/255),(69/255,104/255,144/255),(161/255,179/255,100/255),(110/255,0/255,95/255),
(235/255,154/255,88/255),(87/255,126/255,130/255)]

def test_func2(to_print):
    print(to_print)




def profile_gene(df_hist, df_images, features, channels, gene, xlim=None, ylim=None, vmin=None, vmax=None, seed=1):
    for i in range(len(features)):
        #print(i)
        if xlim != None:
            density_plot(df=df_hist, genes=['NT',gene], figname=gene, feature=features[i], xlim=xlim[i],
                folder = gene)

            if vmin != None:
                plt_gene_nt_cells(gene_ofint=gene,gene_df=df_images,channel=channels[i],vmin=vmin[i], vmax=vmax[i],
                folder = gene, seed = seed)

            else:
                plt_gene_nt_cells(gene_ofint=gene,gene_df=df_images,channel=channels[i],
                folder = gene, seed = seed)


        else:
            density_plot(df=df_hist, genes=['NT',gene], figname=gene, feature=features[i],
                folder = gene)

            if vmin != None:
                #print(features[i])

                plt_gene_nt_cells(gene_ofint=gene,gene_df=df_images,channel=channels[i],vmin=vmin[i], vmax=vmax[i],
                    folder = gene, seed = seed)

            else:
                plt_gene_nt_cells(gene_ofint=gene,gene_df=df_images,channel=channels[i],
                    folder = gene, seed = seed)



def density_plot(df, genes, feature, xlim=None, ylim=None,xlabel=None,figname=None, font_scale=1.5, folder=None):
    
    fdensity=plt.figure(figsize=(5,3))
    for gene in genes:
        subset = df[df.gene==gene]

        sns.set(palette=pal,style='white',font_scale=font_scale)
        # Draw the density plot
        g=sns.distplot(subset[feature], hist = False, kde = True,
                     kde_kws = {'linewidth': 3},
                     label = gene)
        
    sns.despine(top=True, right=True, left=True, bottom=False)

    # Plot formatting
    if xlim !=None:
        plt.xlim(xlim)
    if ylim !=None:
        plt.ylim(ylim)

    plt.legend(prop={'size': 16})
    if xlabel !=None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel('%s'%feature)
    g.set(yticklabels=[],yticks=[])
    if folder != None:
        fdensity.savefig('figures/%s/%s_%s.tif'%(folder,feature,figname),bbox_inches='tight',dpi=300)

    else:
        fdensity.savefig('figures/%s_%s.tif'%(feature,figname),bbox_inches='tight',dpi=300)
    #plt.show()


def density_plot_barcodes(df, barcodes, feature, samegene = False, xlim=None, ylim=None,xlabel=None,figname=None, font_scale=1.5, folder=None):
    

    fdensity=plt.figure(figsize=(5,3))
    sns.set(palette=pal,style='white',font_scale=font_scale)
    # Draw the density plot for NT
    subset = df[df.gene=='NT']

    g=sns.distplot(subset[feature], hist = False, kde = True,
                     kde_kws = {'linewidth': 3},
                     label = 'NT')

    ct = 0
    for bc in barcodes:
        ct += 1
        subset = df[df.cell_barcode_0==bc]
        gene = list(subset.gene)[0]
        # Draw the density plot
        if samegene == True:
            g=sns.distplot(subset[feature], hist = False, kde = True,
                     kde_kws = {'linewidth': 3},
                     label = gene + ' ' + str(ct))
        else: 
            g=sns.distplot(subset[feature], hist = False, kde = True,
                     kde_kws = {'linewidth': 3},
                     label = gene)
        
        
    sns.despine(top=True, right=True, left=True, bottom=False)
    # Plot formatting
    if xlim !=None:
        plt.xlim(xlim)
    if ylim !=None:
        plt.ylim(ylim)

    plt.legend(prop={'size': 16})
    if xlabel !=None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel('%s'%feature)
    g.set(yticklabels=[],yticks=[])
    if folder != None:
        fdensity.savefig('figures/%s/%s_%s.tif'%(folder,feature,figname),bbox_inches='tight',dpi=300)

    else:
        fdensity.savefig('figures/%s_%s.tif'%(feature,figname),bbox_inches='tight',dpi=300)
    #plt.show()

def plt_gene_nt_cells(gene_ofint,gene_df,channel,vmin,vmax,nfiles=10,start=0, folder=None,seed=1):
    
    ofinterest=gene_df.loc[gene_df.gene==gene_ofint,['cell','Site','well']]
    ofinterest=ofinterest.reindex(np.random.RandomState(seed=seed).permutation(ofinterest.index))

    files = []
    for i in range(nfiles*2):
        cell,Site,well=ofinterest.iloc[i,:]
        file = 'process/20X_individual_cells/20X_%s_Site-%s.%s.cell.tif' %(well, Site, cell)
        if os.path.isfile(file) == False:
            continue
        files.append(read(file))
        
    ofinterest=gene_df.loc[gene_df.gene=='NT',['cell','Site','well']]
    ofinterest=ofinterest.reindex(np.random.RandomState(seed=seed).permutation(ofinterest.index))

    nt_files = []
    for i in range(nfiles*2):
        cell,Site,well=ofinterest.iloc[i,:]
        file = 'process/20X_individual_cells/20X_%s_Site-%s.%s.cell.tif' %(well, Site, cell)
        if os.path.isfile(file) == False:
            continue
        nt_files.append(read(file))

        # left column is with guide, right is nontargeting
    f = plt.figure(figsize=(30,10))  # width, height in inches
    

    #nsubplots=5
    for i in range(nfiles*2):
        sub = f.add_subplot(nfiles*2, 2, i + 1)
        sub.axis('off')

        if i % 2 == 0:
            #print(i)
            #print(start,nfiles,i, channel)
            sub.imshow(np.array(files[start:start+nfiles])[int(i/2),channel],vmin=vmin,vmax=vmax,cmap='viridis')
            #plt.imsave('figures/individual_cells_%s.png'%gene_ofint,np.array(files[start:start+nfiles])[int(i/2),channel],dpi=300)
        else:
            sub.imshow(np.array(nt_files[start:start+nfiles])[int((i-1)/2),channel],vmin=vmin,vmax=vmax,cmap='viridis')
            
    f.subplots_adjust(hspace=0,wspace=.0002,top=2,bottom=0,left=0,right=.07)
    if folder != None:
        f.savefig('figures/%s/individual_cells_channel%s_%s.tif'%(folder,channel,gene_ofint),bbox_inches='tight',dpi=300)

    else:
        f.savefig('figures/individual_cells_channel%s_%s.tif'%(channel,gene_ofint),bbox_inches='tight',dpi=300)



