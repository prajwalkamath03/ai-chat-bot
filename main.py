import streamlit as st
from streamlit_app import summerise
from streamlit_extras.add_vertical_space import add_vertical_space


st.markdown("#### Pdf Summarizer & Q&A #################################")

with st.sidebar:
    st.title('PDF Q&A App')
    st.markdown("""
    ## Hey This Tech_Titans !
    """)
    add_vertical_space(5)
    st.write('By Tech_Titans')
st.markdown("""
    

    This is a summarizer for pdf."""
)
if __name__ == "__main__":
    summerise()
