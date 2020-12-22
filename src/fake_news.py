import nlu
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 40)
pd.options.display.float_format = "{:.2f}".format



# Fake News
model_sent = nlu.load('en.classify.fakenews')
news = ['Unicorns have been sighted on Mars!'
        ,'5G and Bill Gates cause COVID'
        ,'Trump to Visit California After Criticism Over Silence on Wildfires']

# Pasted Text
news = """In one day, nine cases meant to attack President-elect Joe Biden's win in key states were denied or dropped, adding up to a brutal series of losses for the President, who's already lost and refuses to let go. 
Many of the cases are built upon a foundational idea that absentee voting and slight mismanagement of elections invite widespread fraud, which is not proven and state leaders have overwhelming said did not happen in 2020.
In court on Friday: 

The Trump campaign lost six cases in Montgomery County and Philadelphia County in Pennsylvania over whether almost 9,000 absentee ballots could be thrown out.
The Trump campaign dropped a lawsuit in Arizona seeking a review by hand of all ballots because Biden's win wouldn't change.
A Republican candidate and voters in Pennsylvania lost a case over absentee ballots that arrived after Election Day, because they didn't have the ability to sue. A case addressing similar issue is still waiting on decisions from the Supreme Court -- which has remained noticeably silent on election disputes since before Election Day.
Pollwatchers in Michigan lost their case to stop the certification of votes in Detroit, and a judge rejected their allegations of fraud.
"""

import re
def sentence_split(txt):
    txt = re.sub("[.!?]", "[SEP]", str(txt))
    txt = re.sub("\n", "", txt)

    return [s for s in txt.split("[SEP]") if len(s) is not 0]

news = sentence_split(news)


df = model_sent.predict(news, output_level="sentence")
df[['fakenews', 'fakenews_confidence','sentence']]



def get_weighted_confidence_scores(df):
    """Function for aggregate over the sentence based fakenews decision apply the weighted mean for each class.
    Weights are the shares of fakes and real sentences in the data
    ---------
    in -> pandas data frame nlu prediction
    out -> fake_confidence, real_confidence as weighted mean of each class
    """
    df = df.reset_index()
    df["fakenews_confidence"] = df.fakenews_confidence.astype(float)

    df_agg = df.groupby('fakenews', as_index = True)["fakenews_confidence"] \
        .agg([("mean", np.mean)
            ,("var", np.var)
            ,("count", np.size)
            ,("weights", lambda x: x.size / float(df.__len__())) 
            ]) \
        .reset_index()

    df_agg["weighted_mean"] = df_agg["mean"] * df_agg["weights"]
    fake_confidence = df_agg.loc[df_agg.fakenews == "FAKE", "weighted_mean"]
    real_confidence = df_agg.loc[df_agg.fakenews == "REAL", "weighted_mean"]

    return float(fake_confidence), float(real_confidence)

fake_confidence, real_confidence = get_weighted_confidence_scores(df)


# The decision is derived from the median of the sentence fakenews certainty factors
if fake_confidence > real_confidence:
    fakenws = "FAKE"
else:
    fakenws = "REAL"
