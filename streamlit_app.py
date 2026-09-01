import streamlit as st

st.set_page_config(page_title="IST 688 Labs", layout="wide")

lab1 = st.Page("Lab1.py", title="Lab 1")
lab2 = st.Page("Lab2.py", title="Lab 2", default=True)

pg = st.navigation([lab2, lab1])
pg.run()