import nlu
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1,rc={"lines.linewidth": 2.5})


pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 40)
pd.options.display.float_format = "{:.2f}".format



##################
# Word embedding #
##################

model_pipe = nlu.load("bert elmo")
txt = 'Unicorns have been sighted on Mars! Trump to Visit California After Criticism Over Silence on Wildfires'
predictions = model_pipe.predict(txt, output_level='token', positions=True)
predictions.head()


# t-SNE Plot
from sklearn.manifold import TSNE

def get_tsne_df(predictions, embd_column, hue_column = None):
    """ Function for get t-SNE ready df 
    Cast column to np aray and generate TSNE embedding and store them into DF with label ready for hue plot
    Some rows contain NONE text as result of preprocessing, thus we have some NA embeddings and drop them
    
    Parameters
    ------------
    predictions -> nlu prediction output as pandas data frame
    embd_column -> column name for the embedding column as str 
    hue_column -> column name for hue. Leave this empty for hue the sentences
    """
    predictions.dropna(how='any', inplace=True)
    # We first create a column of type np array
    predictions['np_array'] = predictions[embd_column].apply(lambda x: np.array(x))
    # Make a matrix from the vectors in the np_array column via list comprehension
    mat = np.matrix([x for x in predictions.np_array])
    
    # Fit and transform T-SNE algorithm
    model = TSNE(n_components=2) #n_components means the lower dimension
    low_dim_data = model.fit_transform(mat)

    if hue_column:
        t_df = pd.DataFrame(low_dim_data, index = predictions[hue_column])
        t_df.columns = ['x','y']
    
    else:
        t_df = pd.DataFrame(low_dim_data)
        t_df.columns = ['x', 'y']

    return t_df 







# Set subplot
# n -> c r
# 1 -> 1 1
# 2 -> 2 1
# 3 -> 3 1 
# 4 -> 2 2
# 5 -> 3 2
# 6 -> 3 2
# for i in range(1,7):
#     print("# %s" % i,min(i,3), 1 if i <=3 else 2)


# Infer the embedding column names
EMBED_COL_NAMES = [c for c in predictions.columns if c.endswith("_embeddings")]
# EMBED_COL_NAMES.extend(["bert_embeddings", "bert_embeddings"])
# EMBED_COL_NAMES.extend(["bert_embeddings"])
n_plots = len(EMBED_COL_NAMES)

fig, axs = plt.subplots(ncols = 2 if n_plots == 4 else min(n_plots, 3) , nrows = 1 if n_plots <= 3 else 2 )

subplot_idx_dict = {}
subplot_idx_dict[2] = [0, 1]
subplot_idx_dict[3] = [0, 1, 2]
subplot_idx_dict[4] = [(0,0), (0,1), (1,0), (1,1)]
subplot_idx_dict[5] = [(0,0), (0,1), (0,2), (1,0), (1,1)]
subplot_idx_dict[6] = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
subplot_idx_list = subplot_idx_dict[n_plots]

for idx, emb_c in enumerate(EMBED_COL_NAMES):
    t_embedd = get_tsne_df(predictions, emb_c)
    if n_plots == 1:
        ax = axs
    elif n_plots in [2,3]:
        ax = axs[subplot_idx_list[idx]]
    else:
        subpl_r, subpl_c = subplot_idx_list[idx]
        ax = axs[subpl_r][subpl_c]
    
    ax = sns.scatterplot(data = t_embedd, x = 'x', y = 'y', ax = ax)
    ax.set_title('T-SNE {}'.format(emb_c))

plt.show()


 
