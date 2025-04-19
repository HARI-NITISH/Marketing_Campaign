import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import random
import gym
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from pandas.plotting import lag_plot
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO

# Title and Sidebar Navigation
st.title("Online Marketing Campaign Optimization")
st.sidebar.title("Navigation")

# Navigation options
analysis_options = [
    "EDA",
    "ML Models",
    "Reinforcement Learning"
]
selected_option = st.sidebar.radio("Select Analysis", analysis_options)
# Load the dataset
try:
    df = pd.read_excel("data/Marketing Campaign Dataset.xlsx")
    df1= df.copy()
    data=df.copy()
    original_data = df.copy()
    st.sidebar.success("Dataset loaded successfully!")
except FileNotFoundError:
    st.error("Dataset file not found. Please ensure 'Marketing_Campaign_Dataset.xlsx' is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the dataset: {e}")
    st.stop()

if df is not None:

    if selected_option == "EDA":
        
        st.header("Channel Analysis")
        channel_analysis = df.groupby('Channel', as_index=False).agg({
            'Conversions': 'sum',
            'CTR, %': 'mean',
            'Spend, GBP': 'sum',
            'Clicks': 'sum'
        })
        channel_analysis['Cost per Conversion'] = channel_analysis['Spend, GBP'] / channel_analysis['Conversions']
        fig = px.bar(
            channel_analysis,
            x='Channel',
            y='Conversions',
            title='Total Conversions by Channel',
            hover_data=['Cost per Conversion'],
            color='Cost per Conversion',
            labels={'Conversions': 'Total Conversions', 'Channel': 'Channel'}
        )
        st.plotly_chart(fig)
        st.write('Instagram drives the most conversions (around 15,000) at the lowest cost per conversion (around 3), while Facebook and Pinterest have fewer conversions (around 12,000 each) at higher costs (around 4–5).')

        st.header("Channel Engagement")
        channel_engagement = df.groupby('Channel')[['Likes (Reactions)', 'Shares', 'Comments']].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        channel_engagement.plot(
            x='Channel',
            kind='bar',
            stacked=True,
            ax=ax,
            color=['skyblue', 'salmon', 'lime']
        )
        plt.title('Engagement Metrics by Channel', fontsize=16)
        plt.xlabel('Channel', fontsize=12)
        plt.ylabel('Average Engagement', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Engagement Type')
        st.pyplot(fig)
        st.write('Facebook and Instagram have the highest average engagement (around 80), primarily driven by likes, while Pinterest has lower engagement (around 60), with a similar distribution of likes, shares, and comments.')

        st.header("Impressions Distribution: Mobile vs. Desktop Outliers")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x="Device", y="Impressions", data=df, ax=ax)
        ax.set_title("Box Plot of Impressions across Devices")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
        st.write('Mobile devices have a higher median impression count (around 2,000) and more outliers (up to 4,000) compared to desktop devices (median around 1,500 with fewer outliers).')

        st.header("Lag Plot Of Impressions")
        fig, ax = plt.subplots(figsize=(5, 5))
        lag_plot(df['Impressions'], ax=ax)
        ax.set_title("Lag Plot of Impressions")
        st.pyplot(fig)
        
        st.write('The lag plot shows a strong positive correlation between consecutive days impressions, indicating that high (or low) impressions on one day are likely to persist into the next.')
        st.header("Monthly Trends")
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Date1']= df['Date'].dt.day
        monthly_trends = df.groupby('Month', as_index=False).agg({
            'Conversions': 'sum',
            'CTR, %': 'mean',
            'Spend, GBP': 'sum',
            'Clicks': 'sum'
        })
        fig1 = px.line(
            monthly_trends,
            x='Month',
            y='Conversions',
            title='Monthly Conversions Trend',
            labels={'Conversions': 'Number of Conversions', 'Month': 'Month'}
        )
        fig2 = px.line(
            monthly_trends,
            x='Month',
            y='CTR, %',
            title='Monthly Click-Through Rate (CTR) Trend',
            labels={'CTR, %': 'CTR (%)', 'Month': 'Month'}
        )
        st.plotly_chart(fig1)
        st.write('Monthly Conversions Trend: Conversions increased steadily from month 3 to month 11, peaking at around 5,200, after fluctuating between 4,200 and 4,800 earlier.')
        st.plotly_chart(fig2)
        st.write('Monthly Click-Through Rate (CTR) Trend: CTR rose sharply from month 7 to 9, peaking at around 0.0135, but declined slightly by month 11 after earlier fluctuations between 0.005 and 0.007.')

        st.header("Ad Analysis")
        df['Total Engagement'] = df['Likes (Reactions)'] + df['Shares'] + df['Comments']
        ad_analysis = df.groupby('Ad', as_index=False).agg({
            'Conversions': 'sum',
            'CTR, %': 'mean',
            'Total Engagement': 'sum',
            'Spend, GBP': 'sum'
        })
        ad_analysis['Cost per Conversion'] = ad_analysis['Spend, GBP'] / ad_analysis['Conversions']
        fig = px.bar(
            ad_analysis,
            x='Ad',
            y='Conversions',
            hover_data=['Cost per Conversion'],
            title='Ads by Total Conversions',
            color='Cost per Conversion',
            labels={'Conversions': 'Total Conversions', 'Ad': 'Ad'}
        )
        st.plotly_chart(fig)
        st.write('Ads by Total Conversions: The "Collection" ad generates the most conversions (around 20,000) at a cost per conversion of about 4.2, while the "Discount" ad has fewer conversions (around 15,000) at a lower cost per conversion (around 3.8).')

        
        st.header("ROI by Channel")
        df1 = df[df['Spend, GBP'] != 0]
        df1['ROI'] = df1['Total conversion value, GBP'] / df1['Spend, GBP']
        roi_channel = df1.groupby('Channel', as_index=False).agg({
            'Spend, GBP': 'sum',
            'Total conversion value, GBP': 'sum',
            'Conversions': 'sum',
            'ROI': 'mean'
        })
        fig = px.bar(
            roi_channel,
            x='Channel',
            y='ROI',
            title='Average ROI by Channel',
            color='Channel',
            labels={'ROI': 'Return on Investment', 'Channel': 'Channel'}
        )
        st.plotly_chart(fig)
        st.write('ROI by Channel: Pinterest has the highest average ROI (around 120), followed by Instagram (around 60), while Facebook has the lowest ROI (around 20).')

        st.header("ROI by City")
        df1 = df[df['Spend, GBP'] != 0]
        df1['ROI'] = df1['Total conversion value, GBP'] / df1['Spend, GBP']
        roi_city = df1.groupby('City/Location', as_index=False).agg({
            'Spend, GBP': 'sum',
            'Total conversion value, GBP': 'sum',
            'Conversions': 'sum',
            'ROI': 'mean'
        })
        fig = px.bar(
            roi_city,
            x='City/Location',
            y='ROI',
            title='Cities by ROI',
            color='ROI',
            labels={'ROI': 'Return on Investment', 'City/Location': 'City'}
        )
        st.plotly_chart(fig)
        st.write('ROI by City: Birmingham has the highest average ROI (around 100), while London and Manchester have lower but similar ROIs (around 40 each).')

        st.header("Device Analysis")
        device_analysis = df.groupby('Device', as_index=False).agg({
            'Conversions': 'sum',
            'CTR, %': 'mean',
            'Spend, GBP': 'sum',
            'Clicks': 'sum'
        })
        device_analysis['Cost per Conversion'] = device_analysis['Spend, GBP'] / device_analysis['Conversions']
        fig = px.bar(
            device_analysis,
            x='Device',
            y='Conversions',
            hover_data=['Cost per Conversion'],
            title='Total Conversions by Device',
            color='Cost per Conversion',
            labels={'Conversions': 'Total Conversions', 'Device': 'Device'}
        )
        st.plotly_chart(fig)
        st.write('Total Conversions by Device: Desktops generate the most conversions (around 20,000) at a cost per conversion of about 4.65, while mobile devices have fewer conversions (around 15,000) at a lower cost per conversion (around 4.55).')


        st.header("Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Spend, GBP'], bins=30, kde=True, color='blue', ax=ax)
        ax.set_title('Distribution of Spend', fontsize=16)
        ax.set_xlabel('Spend', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        st.pyplot(fig) 
        st.write('The histogram shows the distribution of the ‘Spend, GBP’ column: Most spending is concentrated between 0 and 20 GBP, with a peak frequency of around 1,600, and it tapers off sharply as spending increases beyond 40 GBP.')

        st.header("Engagement Metrics")
        ad_engagement = df.groupby('Ad')[['Likes (Reactions)', 'Shares', 'Comments']].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        ad_engagement.plot(x='Ad', kind='bar', stacked=True, ax=ax, color=['skyblue', 'salmon', 'lime'])
        ax.set_title('Engagement Metrics by Ad Type', fontsize=16)
        ax.set_xlabel('Ad Type', fontsize=12)
        ax.set_ylabel('Average Engagement', fontsize=12)
        ax.set_xticklabels(ad_engagement['Ad'], rotation=45)
        ax.legend(title='Engagement Type')
        st.pyplot(fig)
        st.write('Engagement Metrics by Ad Type: The "Discount" ad has higher total engagement (around 80) driven mostly by likes, while the "Collection" ad has lower engagement (around 60), with a balanced distribution of likes, shares, and comments.')


    if selected_option == "ML Models":
        st.header("Feature Engineering")
        
        df['High_Profitability'] = (df['Conversions'] / df['Spend, GBP'] > 1).astype(int)

        model_option = st.selectbox("Choose a Machine Learning Model", 
                            ["Logistic Regression", "Random Forest Classifier", "Gradient Boosting Regressor","Random Forest Regressor"])
        st.subheader(model_option)

        if model_option == "Logistic Regression":
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import classification_report
            features = ['Spend, GBP', 'CTR, %', 'Impressions', 'Clicks']
            X = df[features]
            y = df['High_Profitability']

            # Handle missing values
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1)

            model_code = """
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import classification_report

            features = ['Spend, GBP', 'CTR, %', 'Impressions', 'Clicks']
            X = df[features]
            y = df['High_Profitability']
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1)
            model = LogisticRegression(random_state=1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            print(classification_report(y_test, y_pred))
            """
        
            st.code(model_code, language="python")

            # Train Model
            model = LogisticRegression(random_state=1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Display Evaluation Metrics
            st.subheader("Output:")
            report = classification_report(y_test, y_pred, output_dict=True)
        
            # Convert classification report to DataFrame
            report_df = pd.DataFrame(report).transpose()
            st.table(report_df)
        if model_option == "Random Forest Classifier":
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from sklearn.preprocessing import StandardScaler, LabelBinarizer
            from sklearn.model_selection import train_test_split
            X = df[['Impressions', 'Clicks', 'Spend, GBP','Shares','Comments']]
            # X=df.drop(['Date','Device'],axis=1)
            y = df['Device']

            # Splitting the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model_code = """
            X = df[['Impressions', 'Clicks', 'Spend, GBP','Shares','Comments']]
            y = df['Device']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
            random_forest.fit(X_train, y_train)
            y_pred = random_forest.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.2f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            """
            st.code(model_code, language="python")
            # Creating a Random Forest model
            random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
            # Training the model
            random_forest.fit(X_train, y_train)
            # Making predictions
            y_pred = random_forest.predict(X_test)
            # Evaluating the model
            st.subheader("Output:")
            report = classification_report(y_test, y_pred, output_dict=True)
        
            # Convert classification report to DataFrame
            report_df = pd.DataFrame(report).transpose()
            st.table(report_df)

        if model_option == "Gradient Boosting Regressor":
            model_code = """
                        import xgboost as xgb
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            import random
            import plotly.express as px
            import plotly.graph_objects as go
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            X = df[['Campaign', 'Channel', 'Device',
                'Spend, GBP','Daily Average CPC']]
            y = df['Impressions']
            for col in ['Campaign', 'Channel', 'Device']:
                X[col] = X[col].astype('category')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
            dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
            params = {
                "objective": "reg:squarederror",  
                "learning_rate": 0.1,           
                "max_depth": 6,                 
                "subsample": 0.8,               
                "colsample_bytree": 0.8,        
                "seed": 42                      
            }
            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=100,             
                nfold=5,                         
                metrics={"rmse"},                
                early_stopping_rounds=10,        
                seed=42
            )
            best_num_boost_round = cv_results['test-rmse-mean'].idxmin()
            print(f"Best number of boosting rounds: {best_num_boost_round}")
            final_model = xgb.train(params, dtrain, num_boost_round=best_num_boost_round)

            predictions = final_model.predict(dtest)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            print(f"RMSE on test set: {rmse}")
            predictions = final_model.predict(dtest)
            within_range = np.abs(y_test - predictions) <= (0.2 * np.abs(y_test))
            accuracy = np.mean(within_range) * 100 
            print(f"Accuracy within 10% tolerance: {accuracy}%")
            results_df = pd.DataFrame({"Metric": ["RMSE", "Accuracy (%)"], "Value": [rmse, accuracy]})
            st.write("Performance metrics for the trained XGBoost model:")
            st.dataframe(results_df) 
            """
            st.code(model_code, language="python")
            import xgboost as xgb
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            import random
            import plotly.express as px
            import plotly.graph_objects as go
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            X = df[['Campaign', 'Channel', 'Device',
                'Spend, GBP','Daily Average CPC']]
            y = df['Impressions']
            for col in ['Campaign', 'Channel', 'Device']:
                X[col] = X[col].astype('category')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True) 
            dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
            params = {
                "objective": "reg:squarederror", 
                "learning_rate": 0.1,            
                "max_depth": 6,                  
                "subsample": 0.8,                
                "colsample_bytree": 0.8,         
                "seed": 42                       
            }
            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=100,               
                nfold=5,                           
                metrics={"rmse"},                  
                early_stopping_rounds=10,          
                seed=42
            )
            best_num_boost_round = cv_results['test-rmse-mean'].idxmin()
            print(f"Best number of boosting rounds: {best_num_boost_round}")
            final_model = xgb.train(params, dtrain, num_boost_round=best_num_boost_round)

            predictions = final_model.predict(dtest)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            print(f"RMSE on test set: {rmse}")
            predictions = final_model.predict(dtest)
            within_range = np.abs(y_test - predictions) <= (0.2 * np.abs(y_test))
            accuracy = np.mean(within_range) * 100  
            print(f"Accuracy within 10% tolerance: {accuracy}%")
            results_df = pd.DataFrame({"Metric": ["RMSE", "Accuracy (%)"], "Value": [rmse, accuracy]})
            st.write("Performance metrics for the trained XGBoost model:")
            st.dataframe(results_df) 
        if model_option == "Random Forest Regressor":
            model_code = """
            import pandas as pd
            import numpy as np
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, r2_score
            df1["Engagement Score"] = (df1["Likes (Reactions)"] + df1["Shares"] + df1["Comments"]) / np.log1p(df1["Impressions"] + 0.01)

            df1 = df1.drop(columns=["Date", "Campaign", "City/Location"], errors='ignore')

            categorical_cols = ["Channel", "Device", "Ad"]
            df1 = pd.get_dummies(df1, columns=categorical_cols, drop_first=True)

            X = df1.drop(columns=["Engagement Score"])
            y = df1["Engagement Score"]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            print(f" RMSE: {rmse:.4f}")  
            print(f" R² Score: {r2:.4f}")
            accuracy = r2 * 100
            print(f"Model Accuracy (based on R²): {accuracy:.2f}%")
            """

            st.code(model_code, language="python")
            import pandas as pd
            import numpy as np
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, r2_score
            df1["Engagement Score"] = (df1["Likes (Reactions)"] + df1["Shares"] + df1["Comments"]) / np.log1p(df1["Impressions"] + 0.01)
            df1 = df1.drop(columns=["Date", "Campaign", "City/Location"], errors='ignore')
            categorical_cols = ["Channel", "Device", "Ad"]
            df1 = pd.get_dummies(df1, columns=categorical_cols, drop_first=True)
            X = df1.drop(columns=["Engagement Score"])
            y = df1["Engagement Score"]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split into Train & Test sets
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)

            # Predictions on test set
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            st.subheader("Output:")
            st.write(f" RMSE: {rmse:.4f}") 
            st.write(f" R² Score: {r2:.4f}")

            
            accuracy = r2 * 100
            st.write(f" Model Accuracy (based on R²): {accuracy:.2f}%")



    if selected_option == "Reinforcement Learning":
            import numpy as np
            model_option = st.selectbox("Choose a Reinforcement Learning Model", 
                        ["REINFORCE", "PPO"])
            st.subheader(model_option)

            if model_option == "REINFORCE":
     
       
                states = df[['Campaign', 'Channel', 'Device']].drop_duplicates().reset_index(drop=True)
                state_space = {tuple(row): index for index, row in states.iterrows()}
                action_space = {
                    0: 'Change Channel to Instagram',
                    1: 'Change Channel to Facebook',
                    2: 'Change Channel to Pinterest',
                    3: 'Change Device to Mobile',
                    4: 'Change Device to Desktop'
                }

                # Initialize the policy table with uniform probabilities
                policy_table = np.ones((len(state_space), len(action_space))) / len(action_space)

                def choose_action(state_index):
                    return np.random.choice(len(action_space), p=policy_table[state_index])

                def simulate_episode(policy_table, data, state_space, batch_size=100):
                    sampled_data = data.sample(batch_size)
                    episode = []
                    for index, row in sampled_data.iterrows():
                        state = (row['Campaign'], row['Channel'], row['Device'])
                        state_index = state_space[state]
                        action = choose_action(state_index)
                        reward = row['Conversions']  
                        episode.append((state_index, action, reward))
                    return episode

                # Update the policy using REINFORCE
                def update_policy(policy_table, episode, gamma=0.99, alpha=0.01):
                    G = 0
                    for state_index, action, reward in reversed(episode):
                        G = reward + gamma * G
                        policy_table[state_index, action] += alpha * G * (1 - policy_table[state_index, action])
                        policy_table[state_index, :] = policy_table[state_index, :] / policy_table[state_index, :].sum()
                    return policy_table

                num_episodes = 1000
                batch_size = 1000
                for episode in range(num_episodes):
                    simulated_episode = simulate_episode(policy_table, df, state_space, batch_size)
                    policy_table = update_policy(policy_table, simulated_episode)

                # Evaluating the policy
                final_policy = pd.DataFrame(policy_table, columns=action_space.values(), index=states.apply(tuple, axis=1))
                final_policy

     
            if model_option == "PPO":
                
                st.subheader("PPO Model Inference")
                data = data.drop(['Latitude', 'Longitude'], axis=1)
                obj_col = data.select_dtypes(include=['object']).columns.tolist()
                le = LabelEncoder()
                for i in obj_col:
                    data[i] = le.fit_transform(data[i])
                
                train_data = data.copy()
                train_original_data = train_data.copy()
                train_data['Date'] = pd.to_datetime(train_data['Date'])
                train_data['DayOfWeek'] = train_data['Date'].dt.dayofweek
                train_data['Month'] = train_data['Date'].dt.month
                train_data = train_data.drop('Date', axis=1)

                numerical_cols = ['Impressions', 'CTR, %', 'Clicks', 'Daily Average CPC', 'Spend, GBP', 
                                'Conversions', 'Total conversion value, GBP', 'Likes (Reactions)', 
                                'Shares', 'Comments']

                # Normalize the numerical columns
                scaler = MinMaxScaler()
                train_data[numerical_cols] = scaler.fit_transform(train_data[numerical_cols])

                from xgboost import XGBRegressor
                reg_model = XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )

                X = train_data[['Campaign', 'Channel', 'Device', 'Spend, GBP', 'Daily Average CPC']]
                y = train_data['Impressions']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                reg_model.fit(X_train, y_train)

                y_pred = reg_model.predict(X_test)
                within_range = np.abs(y_test - y_pred) <= (0.2 * np.abs(y_test))
                accuracy = np.mean(within_range) * 100
                st.write(f"Accuracy within 10% tolerance: {accuracy}%")

                import gym
                from gym import spaces

                class MarketingEnv(gym.Env):
                    def __init__(self, data, original_data, reg_model):
                        super(MarketingEnv, self).__init__()
                        self.data = data.reset_index(drop=True)
                        self.original_data = original_data.reset_index(drop=True)
                        self.current_step = 0
                        self.reg_model = reg_model
                        self.action_space = spaces.Discrete(5)
                        self.observation_space = spaces.Box(low=0, high=1, shape=(len(data.columns),), dtype=np.float32)

                    def reset(self):
                        self.current_step = 0
                        return self.data.loc[self.current_step].values

                    def cal_impressions(self):
                        columns = ['Campaign', 'Channel', 'Device', 'Spend, GBP', 'Daily Average CPC']
                        selected_data = self.original_data.loc[self.current_step, columns]
                        self.original_data.at[self.current_step, 'Impressions'] = self.reg_model.predict([selected_data])[0]

                    def step(self, action):
                        self._take_action(action)
                        self.current_step += 1
                        done = self.current_step >= len(self.data)
                        reward = self._calculate_reward(action)
                        if not done:
                            state = self.data.loc[self.current_step].values
                        else:
                            state = np.zeros(self.observation_space.shape)
                        return state, reward, done, {}

                    def _take_action(self, action):
                        if action == 0:
                            self.original_data.at[self.current_step, 'Channel'] = 0  # Encoded label for Facebook
                        elif action == 1:
                            self.original_data.at[self.current_step, 'Channel'] = 1  # Encoded label for Instagram
                        elif action == 2:
                            self.original_data.at[self.current_step, 'Channel'] = 2  # Encoded label for Pinterest
                        elif action == 3:  # Increase budget allocation
                            self.original_data.at[self.current_step, 'Spend, GBP'] *= 1.1
                        elif action == 4:  # Decrease budget allocation
                            self.original_data.at[self.current_step, 'Spend, GBP'] *= 0.9
                        self.cal_impressions()

                    def _calculate_reward(self, action):
                        if self.current_step >= len(self.original_data):
                            return 0
                        return self.original_data.iloc[self.current_step]['Impressions']

                from stable_baselines3.common.vec_env import DummyVecEnv
                from stable_baselines3 import PPO

                # Create the environment
                env = DummyVecEnv([lambda: MarketingEnv(train_data, train_original_data, reg_model)])

                # Train the RL model
                model = PPO('MlpPolicy', env, verbose=1)
                model.learn(total_timesteps=1000)

                # Save the model
                model.save("ppo_marketing")
                st.write("Model saved as 'ppo_marketing'.")

                # Load and test the model
                model = PPO.load("ppo_marketing")
                obs = env.reset()
                for i in range(len(train_data) - 1):
                    action, _states = model.predict(obs)
                    obs, rewards, dones, info = env.step(action)
                    st.write(f"Step {i + 1}")
                    st.write(f"Action: {action}")
                    st.write(f"Reward: {rewards}")
                    st.write(f"State: {obs}")
                    if dones:
                        break
