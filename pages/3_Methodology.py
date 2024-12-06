import streamlit as st
from util.utility import check_password


# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()
    
st.title("Methodology Flow Chart")

st.image("./image/flowchart_query.png")

st.image("./image/flowchart_categoriser.png")