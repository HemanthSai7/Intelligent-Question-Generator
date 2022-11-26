import streamlit as st
import pandas as pd
from keybert import KeyBERT

import seaborn as sns

from src.Pipeline.TextSummarization import T5_Base
from src.Pipeline.QuestGen import sense2vec_get_words,get_question


st.title("❓ Intelligent Question Generator")
st.header("")


with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
-   The *Intelligent Question Generator* app is an easy-to-use interface built in Streamlit which uses [KeyBERT](https://github.com/MaartenGr/KeyBERT), [Sense2vec](https://github.com/explosion/sense2vec), [T5](https://huggingface.co/ramsrigouthamg/t5_paraphraser)
-   It uses a minimal keyword extraction technique that leverages multiple NLP embeddings and relies on [Transformers](https://huggingface.co/transformers/) 🤗 to create keywords/keyphrases that are most similar to a document.
- [sense2vec](https://github.com/explosion/sense2vec) (Trask et. al, 2015) is a nice twist on word2vec that lets you learn more interesting and detailed word vectors.
	    """
    )

    st.markdown("")

st.markdown("")
st.markdown("## 📌 Paste document ")

with st.form(key="my_form"):
    ce, c1, ce, c2, c3 = st.columns([0.07, 2, 0.07, 5, 1])
    with c1:
        ModelType = st.radio(
            "Choose your model",
            ["DistilBERT (Default)", "BERT", "RoBERTa", "ALBERT", "XLNet"],
            help="At present, you can choose 1 model ie DistilBERT to embed your text. More to come!",
        )

        if ModelType == "Default (DistilBERT)":
            # kw_model = KeyBERT(model=roberta)

            @st.cache(allow_output_mutation=True)
            def load_model(model):
                return KeyBERT(model=model)

            kw_model = load_model('roberta')

        else:
            @st.cache(allow_output_mutation=True)
            def load_model(model):
                return KeyBERT(model=model)

            kw_model = load_model("distilbert-base-nli-mean-tokens")

        top_N = st.slider(
            "# of results",
            min_value=1,
            max_value=30,
            value=10,
            help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
        )
        min_Ngrams = st.number_input(
            "Minimum Ngram",
            min_value=1,
            max_value=4,
            help="""The minimum value for the ngram range.
            *Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
            # help="Minimum value for the keyphrase_ngram_range. keyphrase_ngram_range sets the length of the resulting keywords/keyphrases. To extract keyphrases, simply set keyphrase_ngram_range to (1, # 2) or higher depending on the number of words you would like in the resulting keyphrases.",
        )

        max_Ngrams = st.number_input(
            "Maximum Ngram",
            value=1,
            min_value=1,
            max_value=4,
            help="""The maximum value for the keyphrase_ngram_range.
            *Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.
            To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
        )

        StopWordsCheckbox = st.checkbox(
            "Remove stop words",
            value=True,
            help="Tick this box to remove stop words from the document (currently English only)",
        )

        use_MMR = st.checkbox(
            "Use MMR",
            value=True,
            help="You can use Maximal Margin Relevance (MMR) to diversify the results. It creates keywords/keyphrases based on cosine similarity. Try high/low 'Diversity' settings below for interesting variations.",
        )

        Diversity = st.slider(
            "Keyword diversity (MMR only)",
            value=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            help="""The higher the setting, the more diverse the keywords.Note that the *Keyword diversity* slider only works if the *MMR* checkbox is ticked.""",
        )

    with c2:
        doc = st.text_area(
            "Paste your text below (max 500 words)",
            height=510,
        )

        MAX_WORDS = 500
        import re
        res = len(re.findall(r"\w+", doc))
        if res > MAX_WORDS:
            st.warning(
                "⚠️ Your text contains "
                + str(res)
                + " words."
                + " Only the first 500 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
            )

            doc = doc[:MAX_WORDS]
            # base=base=T5_Base("t5-base","cpu",2048)
            # doc=base.summarize(doc)

        submit_button = st.form_submit_button(label="✨ Get me the data!")

    if use_MMR:
        mmr = True
    else:
        mmr = False

    if StopWordsCheckbox:
        StopWords = "english"
    else:
        StopWords = None
    
if min_Ngrams > max_Ngrams:
    st.warning("min_Ngrams can't be greater than max_Ngrams")
    st.stop()

# Uses KeyBERT to extract the top keywords from a text
# Arguments: text (str)
# Returns: list of keywords (list)
keywords = kw_model.extract_keywords(
    doc,
    keyphrase_ngram_range=(min_Ngrams, max_Ngrams),
    use_mmr=mmr,
    stop_words=StopWords,
    top_n=top_N,
    diversity=Diversity,
)
# print(keywords)
    
st.markdown("## 🎈 Results ")

st.header("")


df = (
    pd.DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
    .sort_values(by="Relevancy", ascending=False)
    .reset_index(drop=True)
)

df.index += 1

# Add styling
cmGreen = sns.light_palette("green", as_cmap=True)
cmRed = sns.light_palette("red", as_cmap=True)
df = df.style.background_gradient(
    cmap=cmGreen,
    subset=[
        "Relevancy",
    ],
)

c1, c2, c3 = st.columns([1, 3, 1])

format_dictionary = {
    "Relevancy": "{:.2%}",
}

df = df.format(format_dictionary)

with c2:
    st.table(df)      

with st.expander("Note about Quantitative Relevancy"):
    st.markdown(
        """
    - The relevancy score is a quantitative measure of how relevant the keyword/keyphrase is to the document. It is calculated using cosine similarity. The higher the score, the more relevant the keyword/keyphrase is to the document.
    - So if you see a keyword/keyphrase with a high relevancy score, it means that it is a good keyword/keyphrase to use in question answering, generation ,summarization, and other NLP tasks.
    """
    )           

with st.form(key="ques_form"):
    ice, ic1, ice, ic2 ,ic3= st.columns([0.07, 2, 0.07, 5,0.07])
    with ic1:
        TopN = st.slider(
            "Top N sense2vec results",
            value=20,
            min_value=0,
            max_value=50,
            step=1,
            help="""Get the n most similar terms.""",
        )

    with ic2:
        input_keyword = st.text_input("Paste any keyword generated above")
        keywrd_button = st.form_submit_button(label="✨ Get me the questions!")

if keywrd_button:
    st.markdown("## 🎈 Questions ")    
    ext_keywrds=sense2vec_get_words(TopN,input_keyword)
    if len(ext_keywrds)<1:
        st.warning("Sorry questions couldn't be generated")
    
    for answer in ext_keywrds:
        sentence_for_T5=" ".join(doc.split())
        ques=get_question(sentence_for_T5,answer)
        ques=ques.replace("<pad>","").replace("</s>","").replace("<s>","")
        st.markdown(f'> #### {ques} ')


