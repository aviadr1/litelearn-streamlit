import streamlit as st
from streamlit_shap import st_shap
import numpy as np

import pandas as pd
import seaborn as sns
import litelearn

st.set_page_config(page_title="Data Selection", layout="wide", initial_sidebar_state="auto")

st.title('Litelearn demo')
st.write("[Litelearn](https://github.com/aviadr1/litelearn) a python library for building ML models without fussing over the nitty gritty details for data munging")
st.write("This demo showcases some of the library's capabilities ability to build models from dataframes, and to display model evaluations and explanations")


@st.cache_data
def load_data(dataset):
    if dataset == 'brain_networks':
        df = sns.load_dataset(dataset, header=[0, 1, 2], index_col=0)
    elif dataset == 'exercise':
        df = sns.load_dataset(dataset, index_col=0)
    else:
        df = sns.load_dataset(dataset)

    return df

@st.cache_data
def fit(df, target, drop_columns):
    if np.issubdtype(df[target].dtype, np.number):
        model = litelearn.regress_df(df, target, drop_columns=drop_columns)
    else:
        model = litelearn.classify_df(df, target, drop_columns=drop_columns)

    return model

def display_eval(model):
    evals = model.get_evaluation()

    with st.container():
        st.write('## Evaluation')
        cols = st.columns(len(evals.columns))
        for col, metric in zip(cols, evals.columns):
            test = evals.loc['test', metric]
            train = evals.loc['train', metric]
            col.metric(metric, round(test, 3), f'{round((test-train)/train*100, 1)} %')

        st.write(evals)

def display_shap(model):
    with st.container():
        st.write('## SHAP')
        st_shap(model.display_shap())

@st.cache_data
def save_model(_model):
    filename = 'model.pickle'
    data = _model.dump()
    return filename, data




# Define your pages
def upload_file():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Process the file
        st.write('File uploaded')
        # Can be used wherever a "file-like" object is accepted:
        df = pd.read_csv(uploaded_file)
        return df


def select_data():
    options = ["Option 1", "Option 2", "Option 3"]
    selected_option = st.selectbox(
        'Seaborn datasets',
       [None] + sns.get_dataset_names()
    )
    if selected_option:
        # Process the selection
        # st.write('Option selected')
        df = load_data(selected_option)
        return df

# Create the pages
pages = {
    "Select sample dataset (seaborn)": select_data,
    "Upload your own data": upload_file,
}

# Render the pages
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(pages.keys()))
page = pages[selection]
# page()


def main():
    with st.container():
        st.write('## Dataset and target selection')

        df = page()
        if df is None:
            return

        # show sample
        st.write(df.sample(5))

        target = st.selectbox(
            'select target',
            [None] + df.columns.to_list()
        )

        if target is None:
            return

        drop_columns = st.multiselect(
            'Select columns to drop (optional)',
            df.columns.to_list()
        )

    if df[target].isna().any():
        st.warning('target contains null values. dropping rows with null values in target')
        df = df.dropna(subset=[target])

    model = fit(df, target, drop_columns=drop_columns)

    # show evaluation
    # evals = model.get_evaluation())
    display_eval(model)

    # shap
    display_shap(model)

    # show feature importance
    st.write('### Permutation Importance')
    st.write(model.get_permutation_importance())

    with st.container():
        st.write('## Save model')

        filename, data = save_model(model)
        st.download_button(
            label="Download model as pickle file",
            data=data,
            file_name=filename,
            mime='application/octet-stream',
        )

        code = f'''
        # Usage:
        import pickle
        import litelearn
        import pandas as pd
        
        with open('model.pickle', 'rb') as f:
            model = pickle.load({filename})
        
        df = ...  # load some data
        pred = model.predict(df)  # predict!
        '''

        st.code(code, language='python')



    # model.display_shap()

main()