import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the page configuration
st.set_page_config(page_title="Bike Sharing Data Analysis", layout="wide")

# Title of the dashboard
st.title("Bike Sharing Data Analysis Dashboard")

# Load the dataset
@st.cache_data  # Updated caching function
def load_data():
    df_day = pd.read_csv('day.csv')
    return df_day

df_day = load_data()

# Feature Engineering for Weekend column
df_day['weekend'] = df_day['weekday'].apply(lambda x: 1 if x in [0, 6] else 0)

# Sidebar for User Input
st.sidebar.header("User Input Features")
view_option = st.sidebar.selectbox("Select analysis to view", 
                                   ("Holidays and Weekends Impact", 
                                    "Weather Factors Impact"))
st.sidebar.caption("Nama : Gabriello Dwi Januar Susanto")
st.sidebar.caption("ID Dicoding : gabriellodwi")
st.sidebar.caption("Email : gabriellodwi@gmail.com")

# Data Preview
st.header("Data Preview")
st.dataframe(df_day.head())

# Main Dashboard Content
if view_option == "Holidays and Weekends Impact":
    st.subheader("Impact of Holidays and Weekends on Bike Rentals")
    
    # Average rentals by working day and holiday
    df_grouped = df_day.groupby(['workingday', 'holiday', 'weekend'])['cnt'].mean().reset_index()
    
    st.write("### Average Bike Rentals: Working Days vs Holidays")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='workingday', y='cnt', hue='holiday', data=df_day, ax=ax)
    ax.set_title('Average Bike Rentals: Working Days vs Holidays')
    ax.set_xlabel('Working Day (1 = Working Day, 0 = Non-Working Day)')
    ax.set_ylabel('Average Total Bike Rentals')
    st.pyplot(fig)

    # Rentals on Weekends vs Weekdays
    weekend_vs_weekday = df_day.groupby('weekend')['cnt'].mean().reset_index()
    
    st.write("### Average Bike Rentals on Weekends vs Weekdays")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.barplot(x='weekend', y='cnt', data=weekend_vs_weekday, ax=ax2)
    ax2.set_title('Average Bike Rentals on Weekends vs Weekdays')
    ax2.set_xlabel('Weekend (1 = Weekend, 0 = Weekday)')
    ax2.set_ylabel('Average Total Bike Rentals')
    st.pyplot(fig2)

elif view_option == "Weather Factors Impact":
    st.subheader("Impact of Weather Factors on Bike Rentals")
    
    st.write("### Bike Rentals vs Temperature")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='temp', y='cnt', data=df_day, ax=ax3)
    ax3.set_title('Bike Rentals vs Temperature')
    ax3.set_xlabel('Normalized Temperature')
    ax3.set_ylabel('Total Bike Rentals')
    st.pyplot(fig3)
    
    st.write("### Bike Rentals vs Humidity")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='hum', y='cnt', data=df_day, ax=ax4)
    ax4.set_title('Bike Rentals vs Humidity')
    ax4.set_xlabel('Normalized Humidity')
    ax4.set_ylabel('Total Bike Rentals')
    st.pyplot(fig4)
    
    st.write("### Bike Rentals vs Windspeed")
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='windspeed', y='cnt', data=df_day, ax=ax5)
    ax5.set_title('Bike Rentals vs Windspeed')
    ax5.set_xlabel('Normalized Windspeed')
    ax5.set_ylabel('Total Bike Rentals')
    st.pyplot(fig5)
    
    st.write("### Correlation Heatmap for Weather Factors")
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    corr_matrix = df_day[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax6)
    ax6.set_title('Correlation Matrix for Bike Rentals and Weather Conditions')
    st.pyplot(fig6)

# Footer
st.markdown("""
    **Data Source**: [Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg]
""")
