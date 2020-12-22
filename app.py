import nlu
import streamlit as st
import pandas as pd

import landing_page as LandingPage
import fake_news_classifier_page as FakeNewsClassifierPage
import SessionState as SessionState
import word_embedding_page as WordEmbeddingPage
import sentence_embedding_page as SentenceEmbeddingPage


# Session State
session_state = SessionState.get(
    fakenews_pipe = None
    ,fakenews_out = pd.DataFrame()
    ,fakenews_is_loaded = False
    ,fakenews_txt = ["Write Here..."]
    ,fakenews_is_predicted = False
    ,fakenews_fig = None
    # Word Embedding
    ,word_embed_input_format = "Copy Paste Text"
    ,word_embed_is_labeled = "Labeled data"
    ,word_embed_selected_model_names = ""
    ,word_embed_loaded_model_names = ""
    ,word_embed_pipe = None
    ,word_embed_txt_input = "The quick brown fox jumps over the lazy dog."
    ,word_embed_txt_out = pd.DataFrame()
    ,word_embed_txt_is_predicted = False
    ,word_embed_csv_input = pd.DataFrame()
    ,word_embed_csv_out = pd.DataFrame()
    ,word_embed_csv_is_predicted = False
    ,word_embed_csv_label_column_name = "-"
    ,word_embed_csv_word_column_name ="-"
    # Sentence Embedding
    ,sent_embed_input_format = "sentence"
    ,sent_embed_selected_model_names = ""
    ,sent_embed_is_labeled = "Labeled data"
    ,sent_embed_loaded_model_names = ""
    ,sent_embed_pipe = None
    ,sent_embed_csv_input = pd.DataFrame()
    ,sent_embed_csv_out = pd.DataFrame()
    ,sent_embed_csv_is_predicted = False
    ,sent_embed_csv_label_column_name = "-"
    ,sent_embed_csv_txt_column_name ="-")



# Consolidate pages
def main():
    """Run this to run the programme
    """
    # Page Setup
    st.set_page_config(
        page_title="NLU Showcase App",
        page_icon=":spock-hand:",
        layout="centered",
        initial_sidebar_state="expanded")

    # SIDEBAR
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("", 
                                ["Home"
                                ,"Model Selection: Word Embbeding "
                                ,"Model Selection: Sentence or Document Embedding"
                                ,"Pre Trained Model: Fake News Classifier"
                                # ,"Sarcasm Detection"
                                # ,"Sentiment Classifier"
                                # ,"Language Classifier"
                                ]
                                )

        
    if mode == "Home":
        LandingPage.show(session_state)
    elif mode == "Pre Trained Model: Fake News Classifier":
        FakeNewsClassifierPage.show(session_state)
    elif mode == "Model Selection: Word Embbeding ":
        WordEmbeddingPage.show(session_state)
    elif mode == "Model Selection: Sentence or Document Embedding":
        SentenceEmbeddingPage.show(session_state)
    else:
        None

    st.sidebar.write(
        """
        ---------
        # About 
        """)    
    st.sidebar.info(
        "This app is maintained by Dennis Triepke. "
        "You can learn more about me at [linkedin.com](https://www.linkedin.com/in/dennistriepke/)."
    )



if __name__ == "__main__":
    main()