import streamlit as st

st.set_page_config(page_title="IST 688 Labs", layout="wide")

lab1 = st.Page("Lab1.py", title="Lab 1")
lab2 = st.Page("Lab2.py", title="Lab 2")
lab3 = st.Page("Lab3.py", title="Lab 3")
lab4 = st.Page("Lab4.py", title="Lab 4")
lab5 = st.Page("Lab5.py", title="Lab 5")
lab6 = st.Page("Lab6.py", title="Lab 6")
group5 = st.Page("Group 5 Lab.py", title="Group 5 Lab")
lab8 = st.Page("Lab8 Rerank.py", title="Lab8 Rerank", default=True)
pg = st.navigation([lab8, group5, lab6, lab5, lab4, lab3, lab2, lab1])
pg.run()

#.d