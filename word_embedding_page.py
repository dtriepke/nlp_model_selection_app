import nlu
import streamlit as st
import base64
import time
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')   
sns.set_palette('muted')


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

    if hue_column is not None:
        t_df = pd.DataFrame(low_dim_data, index = pd.factorize(hue_column)[0] )
        t_df.columns = ['x','y']
    
    else:
        t_df = pd.DataFrame(low_dim_data, index = np.ones(len(low_dim_data)))
        t_df.columns = ['x', 'y']

    return t_df

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in ->  pandas dataframe
    out -> href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="Report.csv" >Download csv file</a>' 

    return href


def show(session_state):
    """Run this function for showing the word embedding section in the app
    """
    NLU_MODEL_NAMES = ["bert", "electra", "elmo", "glove", "xlnet", "albert"]

    # SIDEBAR
    st.sidebar.write(
        """
        --------------
        # Setup
        *Start here to select your project setup.
        You can choose between a vast variety of models and select either a text input or upload a CSV file with your text.*
        """
    )

    st.sidebar.header("Step 1")
    model_names = st.sidebar.multiselect(
            "Select one or more models"
            ,NLU_MODEL_NAMES
            ,session_state.word_embed_selected_model_names.split() # Remember selection
            )
    session_state.word_embed_selected_model_names = ' '.join(model_names)
    
    st.sidebar.header("Step 2")
    INPUT_FORMATS = ["Copy Paste Text", "Upload CSV File"]
    session_state.word_embed_input_format = st.sidebar.radio(
            "Select the input format"
            ,INPUT_FORMATS
            ,index=int(np.where(np.array(INPUT_FORMATS) == session_state.word_embed_input_format)[0][0]) # Remember selection
            )
    
    if session_state.word_embed_input_format == "Upload CSV File":
        st.sidebar.header("Step 3")
        LABELED_OPTIONS = ["Labeled data", "Unlabeled data"]
        session_state.word_embed_is_labeled = st.sidebar.radio(
                "Are the data labeled?"
                ,LABELED_OPTIONS
                ,index=int(np.where(np.array(LABELED_OPTIONS) == session_state.word_embed_is_labeled)[0][0]) # Remember selection
                )
        st.sidebar.write("*Note: Select 'Labeled data' in order to hue the t-sne plot.*")



    # MAIN PAGE
    st.title("Word Embeddings with NLU")
    st.info("This is a comparison of some of the embedding developments of [**John Snow Lab**](https://nlu.johnsnowlabs.com/). \
            Here you can find **BERT**, **ALBERT**, **ELMO**, **ELECTRA**, **XLNET** and **GLOVE** embeddings in one output. "
            "You can download the output or use the result for NLP model selection."
            )

    st.write("""
    ## References
    - [BERT Paper](https://arxiv.org/pdf/1810.04805.pdf)
    - [ALBERT Paper](https://openreview.net/forum?id=H1eA7AEtvS)
    - [ELMO Paper](https://arxiv.org/abs/1802.05365)
    - [ELECTRA Paper](https://arxiv.org/abs/2003.10555)
    - [XLNET Paper](https://arxiv.org/pdf/1906.08237.pdf)
    - [GLOVE Paper](https://nlp.stanford.edu/pubs/glove.pdf)
    """)

    # Load the nlu models: show just if
    #   a) at least one model is selected OR
    #   b) at least one model has been loaded
    if session_state.word_embed_selected_model_names or session_state.word_embed_loaded_model_names:
        st.header("Load a model")
        btn_load = st.button("Download selected model(s) from AWS", key="btn_load")
        # Case: at least one model is already loaded AND download button is seleced without any model selection
        if btn_load and not session_state.word_embed_selected_model_names:
            with st.spinner("**Warning**: No model selected. Please select first at least one embedding model from the sidebar!"):
                time.sleep(3)
            btn_load = False
        # Case: selected model already loaded AND download button is selected
        if btn_load and (session_state.word_embed_selected_model_names == session_state.word_embed_loaded_model_names):
            with st.spinner("**Info**: Selected models '{}' already loaded. Stop request.".format(session_state.word_embed_selected_model_names)):
                time.sleep(3)
            btn_load = False
        # Case: load selected model
        if btn_load:
            with st.spinner("Download started this may take some minutes ... :coffee:"):
                session_state.word_embed_pipe = nlu.load(session_state.word_embed_selected_model_names)
                session_state.word_embed_loaded_model_names = ' '.join(model_names)
                # Reset results if exist: csv input
                session_state.word_embed_csv_input = pd.DataFrame()
                session_state.word_embed_csv_out = pd.DataFrame()
                session_state.word_embed_csv_is_predicted = False
                session_state.word_embed_csv_label_column_name = "-"
                session_state.word_embed_csv_word_column_name = "-"
                # Reset results if exist: txt input
                session_state.word_embed_txt_out = pd.DataFrame()
                session_state.word_embed_txt_is_predicted = False
        
        # Run data input Flow: just if at least one model is loaded; 
        if session_state.word_embed_loaded_model_names: 
            st.success("**Info**: loaded models are: {} ".format(session_state.word_embed_loaded_model_names))

            #########################
            # Flow: text input flow #
            #########################
            if session_state.word_embed_input_format == "Copy Paste Text":
                st.header("Get Embeddings from Text here!")
                # Write Here...
                session_state.word_embed_txt_input = st.text_area("Enter text for embedding here", session_state.word_embed_txt_input)
                # Get prediction
                # NOTE: btn_pred state not cached so it returns to false state and not re download every time
                btn_pred = st.button("Calculate", key = "btn_predict")
                if btn_pred:
                    with st.spinner("Calculation started ... :coffee:"):
                        session_state.word_embed_txt_out = session_state.word_embed_pipe.predict(session_state.word_embed_txt_input, positions=True, output_level ='token')
                        session_state.word_embed_txt_is_predicted = True
                
                if session_state.word_embed_txt_is_predicted:
                    st.success("Calculation done!")

                    # Results
                    st.header("Visualize Embeddings for the first 10 Words")
                    st.dataframe(session_state.word_embed_txt_out.head(10))

                    st.header("t-SNE plot for each embeddings")
                    predictions = session_state.word_embed_txt_out 
                    EMBED_COL_NAMES = [c for c in predictions.columns if c.endswith("_embeddings")] # Infer the embedding column names

                    # Draw Subplots
                    n_plots = len(EMBED_COL_NAMES) 
                    fig, axs = plt.subplots(ncols = 2 if n_plots == 4 else min(n_plots, 3) , nrows = 1 if n_plots <= 3 else 2 )
                    subplot_idx_dict = {}
                    subplot_idx_dict[2] = [0, 1]
                    subplot_idx_dict[3] = [0, 1, 2]
                    subplot_idx_dict[4] = [(0,0), (0,1), (1,0), (1,1)]
                    subplot_idx_dict[5] = [(0,0), (0,1), (0,2), (1,0), (1,1)]
                    subplot_idx_dict[6] = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
                    for idx, emb_c in enumerate(EMBED_COL_NAMES):
                        t_embedd = get_tsne_df(predictions, emb_c)
                        if n_plots == 1:
                            ax = axs
                        elif n_plots in [2,3]: # 1 row
                            ax = axs[subplot_idx_dict[n_plots][idx]]
                        else: # row and column
                            subpl_r, subpl_c = subplot_idx_dict[n_plots][idx]
                            ax = axs[subpl_r][subpl_c]  
                        ax = sns.scatterplot(data = t_embedd, x = 'x', y = 'y', ax = ax, c = t_embedd.index.tolist(), s = 100)
                        ax.set_title('T-SNE {}'.format(emb_c))
                    st.pyplot(fig)

                    st.header("Download Embedding Table")
                    link = get_table_download_link(session_state.word_embed_txt_out)
                    st.write(link, unsafe_allow_html = True)

             
            ########################
            # Flow: csv input flow #
            ########################
            else:
                st.header("Get Embeddings from CSV file here!")

                uploaded_file = st.file_uploader("Choose a CSV file to upload", type = "csv")
                # st.write("DEBUG:", uploaded_file)
                # st.write("DEBUG:", session_state.word_embed_csv_input)

                # No file selected
                if uploaded_file is None:
                    st.info("Upload a CSV.")
                    # Clear cache: User removed seletced file 
                    if len(session_state.word_embed_csv_input) > 0:
                        session_state.word_embed_csv_input = pd.DataFrame()

                # After file selection, read CSV one time
                if uploaded_file and len(session_state.word_embed_csv_input) == 0:
                    session_state.word_embed_csv_input = pd.read_csv(uploaded_file, sep=",", header=[0], encoding="utf-8", dtype = "unicode")
                    
                # After CSV has been loaded:
                if len(session_state.word_embed_csv_input) > 0:
                    st.write(session_state.word_embed_csv_input)

                    # Map Column
                    st.write('**Map Column**')
                    COLUMNS_NAMES = ["-"] + session_state.word_embed_csv_input.columns.tolist()
                    session_state.word_embed_csv_word_column_name = st.selectbox("Select word column"
                                                                                 ,COLUMNS_NAMES
                                                                                 ,index = int(np.where(np.array(COLUMNS_NAMES) == session_state.word_embed_csv_word_column_name)[0][0]) # Remember selection
                                                                                 )
                    if session_state.word_embed_is_labeled == "Labeled data":
                        session_state.word_embed_csv_label_column_name = st.selectbox("Select label column"
                                                                                     ,COLUMNS_NAMES
                                                                                     ,index = int(np.where(np.array(COLUMNS_NAMES) == session_state.word_embed_csv_label_column_name)[0][0]) # Remember selection
                                                                                    )
                        
                    # Get prediction
                    # NOTE: btn_pred state not cached for single prediction (state will return to false after one time trigger)
                    if session_state.word_embed_csv_word_column_name != "-":
                        btn_pred2 = st.button("Calculate", key = "btn_predict")
                        if btn_pred2:
                            with st.spinner("Calculation started ... :coffee:"):
                                session_state.word_embed_csv_out = session_state.word_embed_pipe.predict(session_state.word_embed_csv_input[[session_state.word_embed_csv_word_column_name]], positions=True, output_level ='token')
                                session_state.word_embed_csv_is_predicted = True
                    
                    if session_state.word_embed_csv_is_predicted:
                        st.success("Calculation done!")

                        # Results
                        st.header("Visualize Embeddings for the first 10 Words")
                        st.dataframe(session_state.word_embed_csv_out.head(10))

                        st.header("t-SNE plot for each embeddings")
                        predictions = session_state.word_embed_csv_out 
                        EMBED_COL_NAMES = [c for c in predictions.columns if c.endswith("_embeddings")] # Infer the embedding column names

                        # Draw Subplots
                        n_plots = len(EMBED_COL_NAMES) 
                        fig, axs = plt.subplots(ncols = 2 if n_plots == 4 else min(n_plots, 3) , nrows = 1 if n_plots <= 3 else 2 )
                        subplot_idx_dict = {}
                        subplot_idx_dict[2] = [0, 1]
                        subplot_idx_dict[3] = [0, 1, 2]
                        subplot_idx_dict[4] = [(0,0), (0,1), (1,0), (1,1)]
                        subplot_idx_dict[5] = [(0,0), (0,1), (0,2), (1,0), (1,1)]
                        subplot_idx_dict[6] = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
                        for idx, emb_c in enumerate(EMBED_COL_NAMES):
                            t_embedd = get_tsne_df(predictions, emb_c, hue_column = session_state.word_embed_csv_input[session_state.word_embed_csv_label_column_name] if (session_state.word_embed_is_labeled == "Labeled data") else None)
                            if n_plots == 1:
                                ax = axs
                            elif n_plots in [2,3]: # 1 row
                                ax = axs[subplot_idx_dict[n_plots][idx]]
                            else: # row and column
                                subpl_r, subpl_c = subplot_idx_dict[n_plots][idx]
                                ax = axs[subpl_r][subpl_c]  
                            ax = sns.scatterplot(data = t_embedd, x = 'x', y = 'y', ax = ax, c = t_embedd.index.tolist(), s = 100)
                            ax.set_title('T-SNE {}'.format(emb_c))
                        st.pyplot(fig)

                        st.header("Download Embedding Table")
                        link = get_table_download_link(session_state.word_embed_csv_out)
                        st.write(link, unsafe_allow_html = True)


