import streamlit as st
import joblib
import numpy as np
import torch
from Model_architecture import BestModel

model_state = torch.load('Model\TorchModel.pt')
model = BestModel(8)
model.load_state_dict(model_state)

st.title('Critical Heat Flux Prediction')

author_options = ['Inasaka', 'Peskov', 'Thompson', 'Weatherhead', 'Williams', 'Beus',
       'Janssen', 'Mortimore', 'Kossolapov', 'Richenderfer']
geometry_options = ['tube', 'annulus', 'plate']

with st.form('Form'):
    st.header('Set the below specifications')
    author = st.selectbox('Enter the name of author: ',options=author_options)
    geometry = st.selectbox('Enter the type of geometry:',geometry_options)
    pressure = st.slider('Enter the pressure of the reactor in MPa: ',min_value=0.10,max_value=20.68,step=0.01)
    mass_flux = st.slider('Enter the mass flux in kg/m2-s: ',min_value=0,max_value=7975,step=1)
    x_e_out = st.slider('Enter the equilibrium quality: ',min_value=-0.87,max_value=0.23,step = 0.01)
    D_e = st.slider('Enter the channel equivalent diameter in mm: ',min_value=1.0,max_value=37.5,step=0.1)
    D_h = st.slider('Enter the channel heated diameter in mm: ',min_value=1,max_value=120,step=1)
    length = st.slider('Enter the heated length in mm: ',min_value=10,max_value=3050,step=10)

    submit_values = st.form_submit_button('Predict the CHF')

if submit_values:
    author = 1 + author_options.index(author)
    geometry = 1 + geometry_options.index(geometry)

    data = torch.tensor(np.array([author,geometry,pressure,mass_flux,x_e_out,D_e,D_h,length]),dtype=torch.float,
                        requires_grad=True)

    prediction = model(data).item()

    st.header('The predicted critical heat flux is ')
    st.success(f'{prediction} MW/m2',icon='🔥')


