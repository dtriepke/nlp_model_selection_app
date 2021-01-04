import streamlit as st



def show(session_state):  
    """Run this function for showing the fake news section in app
    """
    st.write(
        """
        ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

        # Welcome to NLU Powered Model Selection App :spock-hand:

        Here you can find John Snow Lab NLU showcases.
        With the freshly released NLU library which gives you 350+ NLP models and 100+ Word Embeddings, you have infinite possibilities to explore your data and gain insights. 
        The app can help you to select between the most popular pre-trained word, sentence or document embeddings based on your own data.
        """
        )



if __name__ == "__main__":
    pass