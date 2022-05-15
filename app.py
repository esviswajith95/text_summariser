from soupsieve import select
import streamlit as st
import summariser as sm

st.title("Text summariser")

user_input = st.text_area("Your text", "Paste the text here", height=300)
submit = 0
submit = st.button("Summarise")

input_length = len(sm.read_article(user_input))

st.sidebar.title("Options")
summary_type =  st.sidebar.radio("Select summarisation type", ('Extractive', 'Abstractive'))
summary_length = st.sidebar.slider("Select summary length", 1, input_length, round(input_length/4.0), disabled=(summary_type=='Abstractive'))

if submit:

    if summary_type == 'Extractive':
        st.subheader("Summary:")

        with st.spinner(text="This may take a moment..."):

            summary_list = sm.extractive_summary(user_input, int(summary_length))
            summary = " ".join((str(x) for x in summary_list[:-1]))

        st.write(summary)
    
    elif summary_type == 'Abstractive':
            summary = sm.abstractive_summary(user_input) 
            st.write(summary)
    else:
        st.write("select a summarisation type")

