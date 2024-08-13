import streamlit as st
import pandas as pd


st.title("Streamlt Text Input")


name=st.text_input("Enter the name")
if name:
    st.write(f"Hello {name}")


age=st.slider("select your age",0,100,25)
st.write(f"Your age is {age}")

options=["Python","Java","C++","JS"]
choice=st.selectbox("choose your favourite Language:",options)
st.write(f"You selected : {choice}")

data={
    "Name":["John","Jane","Jake","Jill"],
    "Age":[28,24,35,40],
    "City":["New York","Los Angeles","Chicago","Houston"]
}
df=pd.DataFrame(data)
df.to_csv("sampledata.csv")
st.write(df)

Upload_file=st.file_uploader("Choose a CSV File",type="csv")
if Upload_file is not None:

    df=pd.read_csv(Upload_file)
    st.write(df)





"""
https://streamlit.io/components
"""






