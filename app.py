import streamlit as st 
import pandas as pd
import os

import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

from pycaret.regression import setup, compare_models, pull, save_model, load_model


with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2019/12/ONE-POINT-01-1.png')
    st.title('AutoML Generator')
    choice = st.radio('Navigation',  ['Upload', 'Profiling', 'ML', 'Download'])
    st.info("This application allows you to builf automated ML pipelines")
    
if os.path.exists('sourcedata.csv'):
    df = pd.read_csv('sourcedata.csv', index_col=None)

if choice == 'Upload':
    st.title('Upload data for modeling')
    file = st.file_uploader('Upload dataset')
    if file:
        df = pd.read_csv(file, index_col=False)
        df.to_csv('sourcedata.csv', index=False)
        st.dataframe(df)

if choice == 'Profiling':
    st.title('Automated EDA')
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == 'ML':
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")


