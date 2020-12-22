import nlu
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re


def sentence_split(txt):
    """Function to split srings into sentences
    
    Parameters 
    ---------------
    txt -> the text to be splitted into sentences
    
    Output
    --------------
    non empty list object
    """
    txt = re.sub("[.!?]", "[SEP]", str(txt))
    txt = re.sub("\n", "", txt)

    return [s for s in txt.split("[SEP]") if len(s) is not 0]

@st.cache
def get_weighted_confidence_scores(df):
    """Function for aggregate over the sentence based fakenews decisions and apply the weighted mean.
    Weights are the shares of fakes and real sentences in the data
    ---------
    in -> pandas data frame from nlu prediction 
    out -> fake_confidence, real_confidence as weighted mean of each class
    """
    df = df.reset_index()
    df["fakenews_confidence"] = df.fakenews_confidence.astype(float)

    df_agg = df.groupby('fakenews', as_index = True)["fakenews_confidence"] \
        .agg([("mean confidence score", np.mean)
            # ,("var confidence score", np.var)
            # ,("number sentences", np.size)
            ,("weights", lambda x: x.size / float(df.__len__())) 
            ]) \
        .reset_index()

    df_agg["weighted_mean"] = df_agg["mean confidence score"] * df_agg["weights"]
    fake_confidence = df_agg.loc[df_agg.fakenews == "FAKE", "weighted_mean"]
    real_confidence = df_agg.loc[df_agg.fakenews == "REAL", "weighted_mean"]

    fake_confidence = float(fake_confidence) if fake_confidence.any() else 0.0
    real_confidence = float(real_confidence) if real_confidence.any() else 0.0

    return fake_confidence, real_confidence


@st.cache
def get_describe_over_fake_real_classes(df):
    """ This function aggregates over the classes FAKE and REAL and outputs some statistics
    in ->  pandas data frame from nlu prediction
    out -> pandas data frame 
    """

    df = df.reset_index()
    df["fakenews_confidence"] = df.fakenews_confidence.astype(float)

    df_agg = df.groupby('fakenews', as_index = True)["fakenews_confidence"] \
        .agg([("mean confidence score", np.mean)
            ,("var confidence score", np.var)
            ,("number sentences", np.size)
            ,("weights", lambda x: x.size / float(df.__len__())) 
            ]) \
        .reset_index()

    return df_agg



def show(session_state):
    """Run this function for showing the fake news section in the app
    """

    NLU_MODEL_NAMES = ["en.classify.fakenews"]

    # MAIN PAGE
    st.title("Fake News Classifier :newspaper:")
    st.info("This is a pre trained language model for fake news detection."
            "The **fake news classifiers** is an version of the development of [**John Snow Lab**](https://nlu.johnsnowlabs.com/)." 
            "It uses universal sentence embeddings and was trained with the classifierdl algorithm provided by Spark NLP.")
    
    # Load a model
    st.header("1. Load a model")
    model_name = st.selectbox("Select model", NLU_MODEL_NAMES)
    
    btn_load = st.button("Download the model from AWS", key="btn_load")
    if btn_load:
        with st.spinner("Download started this may take some time ..."):
            session_state.fakenews_pipe = nlu.load(model_name)
            session_state.fakenews_is_loaded = True
    
    if session_state.fakenews_is_loaded:
        st.success("Download {} done!".format(model_name))

        # Get prediction
        st.header("2. Try the algorithm here")
        txt = st.text_area("Enter news text for classification.", ".".join(session_state.fakenews_txt))
        session_state.fakenews_txt = sentence_split(txt)

        btn_pred = st.button("Calculate", key="btn_pred")
        if btn_pred:
            with st.spinner("Calculation started ..."):
                session_state.fakenews_out = session_state.fakenews_pipe.predict(session_state.fakenews_txt)
                session_state.fakenews_is_predicted = True

        if session_state.fakenews_is_predicted:
            st.success("Calculation done!")

            # Results
            st.header("Result")
            st.write("DEBUG", session_state.fakenews_out)
            fake_confidence, real_confidence = get_weighted_confidence_scores(session_state.fakenews_out)
            if fake_confidence > real_confidence:
                fakenews = "FAKE"
                st.warning("The news are {} with a certainty of {}".format(fakenews, fake_confidence))
            else:
                fakenews = "REAL"
                st.info("The news are {} with a certainty of {}".format(fakenews, real_confidence))
            st.write("*Note: the decision is infered from the weighted mean of as FAKE or REAL detected sentences.*")

            st.header("Deep Dive")
            st.dataframe(get_describe_over_fake_real_classes(session_state.fakenews_out))
            session_state.fakenews_fig = px.histogram(session_state.fakenews_out, x = "fakenews")
            st.plotly_chart(session_state.fakenews_fig)

            
            st.write("**Sentence Embeddings**")
            st.dataframe(session_state.fakenews_out)
    else:
        st.info("No model loaded. Please load first a model!")

if __name__ == "__main__":
    pass