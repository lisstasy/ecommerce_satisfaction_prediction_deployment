
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from sklearn.preprocessing import  StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Load Classification and Clustering Pipeline models
model_classification = joblib.load('Brazilian Ecommerce Classification.bkl')
model_clustering = joblib.load('Brazilian Ecommerce Clustering.bkl')

# Create Sidebar to navigate between EDA, Classification and Clustering
sidebar = st.sidebar
mode = sidebar.radio('Mode', ['EDA', 'Classification', 'Clustering'])
st.markdown("<h1 style='text-align: center; color: #ff0000;'></h1>", unsafe_allow_html=True)

if mode == "EDA":

    def main():

        # Header of Customer Satisfaction Prediction
        html_temp="""
                    <div style="background-color:#F5F5F5">
                    <h1 style="color:#31333F;text-align:center;"> Customer Satisfaction Prediction </h1>
                    </div>
                """
        # Create sidebar to upload CSV files
        with st.sidebar.header('Upload your CSV data'):
            uploaded_file = st.sidebar.file_uploader('Upload your input csv file')

        if uploaded_file is not None:
            # Read file and Put headers
            EDA_sample = pd.read_csv(uploaded_file, index_col= 0)
            pr = ProfileReport(EDA_sample, explorative=True)
            st.header('**Input DataFrame**')
            st.write(EDA_sample)
            st.write('---')
            st.header('**Pandas Profiling Report**')
            st_profile_report(pr)
        
        else:
            st.info('Awaiting for CSV file to be uploaded.')

    if __name__ == '__main__':
        main()

if mode == "Classification":

    # Define function to predict classification based on assigned features
    def predict_satisfaction(freight_value, product_description_lenght, product_photos_qty, payment_type, payment_installments, payment_value, 
    estimated_days, arrival_days, arrival_status, seller_to_carrier_status, estimated_delivery_rate, arrival_delivery_rate, shipping_delivery_rate):

        prediction_classification = model_classification.predict(pd.DataFrame({'freight_value' :[freight_value], 'product_description_lenght' :[product_description_lenght], 'product_photos_qty' :[product_photos_qty], 'payment_type' :[payment_type], 'payment_installments' :[payment_installments], 'payment_value' :[payment_value], 'estimated_days' :[estimated_days], 'arrival_days' :[arrival_days], 'arrival_status' :[arrival_status], 'seller_to_carrier_status' :[seller_to_carrier_status], 'estimated_delivery_rate' :[estimated_delivery_rate], 'arrival_delivery_rate' :[arrival_delivery_rate], 'shipping_delivery_rate' :[shipping_delivery_rate]}))
        return prediction_classification

    def main():

        # Header of Customer Satisfaction Prediction
        html_temp="""
                    <div style="background-color:#F5F5F5">
                    <h1 style="color:#31333F;text-align:center;"> Customer Satisfaction Prediction </h1>
                    </div>
                """
        st.markdown(html_temp,unsafe_allow_html=True)
        
        # Assign all features with desired data input method
        sidebar.title('Numerical Features')
        product_description_lenght = sidebar.slider('product_description_lenght', 4,3990,100)
        product_photos_qty = sidebar.slider('product_photos_qty', 1,20,1)
        payment_installments = sidebar.slider('payment_installments', 1,24,1)
        estimated_days = sidebar.slider('estimated_days', 3,60,1)
        arrival_days = sidebar.slider('arrival_days', 0,60,1)
        payment_type = st.selectbox('payment_type', ['credit_card', 'boleto', 'voucher', 'debit_card'])
        arrival_status = st.selectbox('arrival_status', ['OnTime/Early', 'Late'])
        seller_to_carrier_status = st.selectbox('seller_to_carrier_status', ['OnTime/Early', 'Late'])
        estimated_delivery_rate = st.selectbox('estimated_delivery_rate', ['Very Slow', 'Slow', 'Neutral', 'Fast', 'Very Fast'])
        arrival_delivery_rate = st.selectbox('arrival_delivery_rate', ['Very Slow', 'Slow', 'Neutral', 'Fast', 'Very Fast'])
        shipping_delivery_rate = st.selectbox('shipping_delivery_rate Date', ['Very Slow', 'Slow', 'Neutral', 'Fast', 'Very Fast'])
        payment_value = st.text_input('payment_value', '')
        freight_value = st.text_input('freight_value', '')
        result = ''

        # Predict Customer Satsifaction
        if st.button('Predict_Satisfaction'):
            result = predict_satisfaction(freight_value, product_description_lenght, product_photos_qty, payment_type, payment_installments, payment_value, 
                                        estimated_days, arrival_days, arrival_status, seller_to_carrier_status, estimated_delivery_rate, arrival_delivery_rate, shipping_delivery_rate)
                                        
        if result == 0:
            result = 'Not Satisfied'
            st.success(f'The Customer is {result}')
        else:
            result = 'Satisfied'
            st.success(f'The Customer is {result}')

    if __name__ == '__main__':
        main()

if mode == "Clustering":

    def predict_clustering(freight_value, price, payment_value, payment_installments, payment_sequential):

        prediction_clustering = model_clustering.predict(pd.DataFrame({'freight_value' :[freight_value], 'price' :[price], 'payment_installments' :[payment_installments], 'payment_value' :[payment_value], 'payment_sequential' :[payment_sequential]}))
        return prediction_clustering

    def main():

        # Header of Customer Segmentation
        html_temp="""
                <div style="background-color:#F5F5F5">
                <h1 style="color:#31333F;text-align:center;"> Customer Segmentation </h1>
                </div>
            """
        st.markdown(html_temp,unsafe_allow_html=True)

        # Assign all features with desired data input method
        payment_installments = st.slider('payment_installments', 1,24,1)
        payment_sequential = st.slider('payment_sequential', 1,24,1)
        freight_value = st.text_input('freight_value', '')
        price = st.text_input('price', '')
        payment_value = st.text_input('payment_value', '')
        result_cluster = ''

        # Predict Cluster of the customer
        if st.button('Predict_Cluster'):
            result_cluster = predict_clustering(freight_value, price, payment_value, payment_installments, payment_sequential)
                                        
        st.success(f'Customer Cluster is {result_cluster}')
        
        # Upload CSV file
        with st.sidebar.header('Upload your CSV data'):
            uploaded_file = st.sidebar.file_uploader('Upload your input csv file')

        if uploaded_file is not None:

            # Read dataset
            sample = pd.read_csv(uploaded_file, index_col= 0)
            
            # Define sidebar for clustering algorithm
            selected_algorithm = sidebar.selectbox('Select Clustering Algorithm', ['K-Means', 'Agglomerative'])

            # Define sidebar for number of clusters
            selected_clusters = sidebar.slider('Select number of clusters', 2, 10, 1)

            # Define sidebar for PCA
            use_pca = sidebar.radio('Use PCA', ['No', 'Yes'])

            # Drop freight values with zeros
            sample.drop(sample[sample.freight_value == 0].index, inplace= True)
            # Reset Index 
            sample.reset_index(inplace= True, drop= True)
            # Handle Skeweness in sample data
            for i in ['freight_value', 'price', 'payment_value', 'payment_installments', 'payment_sequential']:
                sample[i] = np.log10(sample[i])

            # Apply standard scaler
            sc = StandardScaler(with_mean= False)
            data_scaled = sc.fit_transform(sample)

            # Select number of clusters
            if selected_algorithm == 'Agglomerative':
                hc = AgglomerativeClustering(n_clusters= selected_clusters)
                y_pred_hc = hc.fit_predict(data_scaled)

            else:
                kmean = KMeans(n_clusters= selected_clusters)
                y_pred_kmean = kmean.fit_predict(data_scaled)

            # Apply PCA
            pca = PCA(n_components= 2)
            data_pca = pca.fit_transform(data_scaled)

            # Select number of clusters for PCA
            kmean_pca = KMeans(n_clusters= selected_clusters)
            y_pred_pca = kmean_pca.fit_predict(data_pca)

            def plot_cluster(data, y_pred, num_clusters):

                # Plot Clusters
                fig, ax = plt.subplots()
                Colors= ['red', 'green', 'blue', 'purple', 'orange', 'royalblue', 'brown', 'grey', 'chocolate', 'fuchsia']
                for i in range(num_clusters):
                    ax.scatter(data[y_pred==i,0], data[y_pred==i,1], c= Colors[i], label= 'Cluster ' + str(i+1))

                ax.set_title('Customers Clusters')
                ax.legend(loc='upper left', prop={'size':5})
                ax.axis('off')
                st.pyplot(fig)

            # Option to select and plot PCA for clustering
            if use_pca == 'No' and selected_algorithm == 'K-Means':
                plot_cluster(data_scaled, y_pred_kmean, selected_clusters)

            elif use_pca == 'No' and selected_algorithm == 'Agglomerative':
                plot_cluster(data_scaled, y_pred_hc, selected_clusters)           

            else:
                plot_cluster(data_pca, y_pred_pca, selected_clusters)    
        
        else:
            st.info('Awaiting for CSV file to be uploaded.')

    if __name__ == '__main__':
        main()
