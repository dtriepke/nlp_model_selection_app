import streamlit as st



def show(session_state):  
    """Run this function for showing the fake news section in app
    """
    st.write(
        """
        ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

        # Welcome to NLU App :spock-hand:

        **Select one model form the drop down menu** in order to start.  

        Here you can find John Snow Lab NLU showcases. 
        With the freshly released NLU library which gives you 350+ NLP models and 100+ Word Embeddings, you have infinite possibilities to explore your data and gain insights. 
        """
        )



if __name__ == "__main__":
    pass