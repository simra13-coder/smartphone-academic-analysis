# Disruptor or Facilitator? A Data-Driven Analysis of Smartphone Usage Patterns and Academic Outcomes Among Teenagers

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#Load dataset
df= pd.read_csv("Research Form.csv")

#Removing Timestamp
df=df.drop(columns=["Timestamp"])

#Renaming features for processing consistency
df.rename(columns={"Your Age (14-18)": "Age", 'Your Gender':"Gender", 'Grade Level':"Grade",
                   'Average Daily Screen Time ': "ScreenTime", 'Main Phone Usage  ': "Phone Usage",
                   'How often do you use your phone for academic purposes?':"Academic use",
                   'Average Study Time Per Day':"StudyTime", 'Your Academic Performance':"Marks",
                   'On a scale of 1–5, how distracted do you feel due to phone use while studying?  ':"Distraction",
                   'How often do you check your phone while studying?': "Phone check",
                   'Do you feel smartphones help your learning?' : "Phone learning"} , inplace=True)

# Ordinal Scaling Adjustments
screen_map= {"Less than 2 hours" : 1 , "2-4 hours" : 3 , "4-6 hours" : 5 , "6-8 hours" : 7 , "More than 8 hours" : 9}
df["ScreenTime"]=df["ScreenTime"].map(screen_map)

study_map = {"Less than 1 hour" : 0.5 , "1-2 hours" : 1 , "2-4 hours" : 3 , "4-6 hours" : 5 , "More than 6 hours" : 7}
df["StudyTime"]=df["StudyTime"].map(study_map)

marks_map = {"Below 50%" : 40 , "50-60%" : 55 , "60-70%" : 65 , "70-80%" : 75 , "80-90%" : 85,  "90%+": 95}
df["Marks"]=df["Marks"].map(marks_map)
df = df.dropna()   #Rows containing missing or incomplete values were removed using the dropna() function to ensure accurate analysis and model training.

# --- Correlation Analysis ---
print("\n--- Correlation Analysis ---")
print("Relation between screen time & marks" , df["ScreenTime"].corr(df["Marks"]))

print("-"*80)
plt.figure()
sns.regplot(x= "ScreenTime" , y="Marks" , data=df , order=2, color="green")  # FIGURE 1
plt.xlabel("Screen Time")
plt.ylabel("Marks")
plt.title("Screen Time vs Marks")
plt.tight_layout()
plt.show()

print(" Relation between studytime & Marks " , df["StudyTime"].corr(df["Marks"]))  

print("-"*80)
plt.figure()
sns.regplot(x= "StudyTime" , y="Marks" , data=df , order = 2 , color="red" )   # FIGURE 2
plt.xlabel("Study Time")
plt.ylabel("Marks")
plt.title("Study Time vs Marks")
plt.tight_layout()
plt.show()

print( " Relation between screen time & study time ", df["ScreenTime"].corr(df["StudyTime"]))  

print("-"*80)
plt.figure()
sns.regplot(x ="ScreenTime", y="StudyTime" , data=df , order =2 ,color="purple")  #FIGURE 3
plt.xlabel("Screen Time")
plt.ylabel("Study Time")
plt.title("Screen Time vs Study Time")
plt.tight_layout()
plt.show()


# --- Group Analysis ---

print("\n--- Group Analysis---")
print(df.groupby("Academic use")["Marks"].mean())

plt.figure()
df.groupby("Academic use")["Marks"].mean().plot(kind="bar" )                   # FIGURE 4
plt.xlabel("Academic Use")
plt.ylabel("Marks")
plt.title("Academic use VS Marks Trend")
plt.tight_layout()
plt.show()

print("-"*80)
print(df.groupby("Academic use")["Distraction"].mean())

plt.figure()
df.groupby("Academic use")["Distraction"].mean().plot(kind="bar" )               # FIGURE 5
plt.xlabel("Academic use")
plt.ylabel("Distraction")
plt.title("Academic use VS Distraction Trend")
plt.tight_layout()
plt.show()


print("-"*80)
print(df.groupby("Phone learning")["Marks"].mean())

plt.figure()
df.groupby("Phone learning")["Marks"].mean().plot(kind="bar" )                     # FIGURE 6
plt.xlabel("Phone learning")
plt.ylabel("Marks")
plt.title("Phone learning VS Marks")
plt.tight_layout()
plt.show()


# Correlation Heatmap
print("-"*80)
plt.figure()
corr= df[["ScreenTime","StudyTime","Marks" , "Distraction" ]].corr()             # FIGURE 7   
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

print("-"*80)


# Train / test
features = ["ScreenTime","StudyTime","Distraction"]
X= df[features]                    
y= df["Marks"]

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#---------- Machine Learning System -------------

print("\n------------ Machine Learning Module --------------")

#==================================================
#Linear Regression
#==================================================
print("=======   LINEAR REGRESSION   =======")
model= LinearRegression()
model.fit(X_train,y_train)
print("Coefficient: ",model.coef_)
print("Intercept: ",model.intercept_)

#Prediction

y_pred=model.predict(X_test)
print("Model accuracy (R²)" , r2_score(y_test,y_pred))
print()

#==================================================
# Random Forest
#==================================================
print("======   RANDOM FOREST REGRESSOR   ======")
new = RandomForestRegressor(n_estimators=100 , random_state= 42)  
new.fit(X_train,y_train)

#Prediction
new_pred=new.predict(X_test)
print("Model accuracy (R²)" , r2_score(y_test,new_pred))
print()

# Comparison
print( "===== Comparison between Linear Regression R² & RandomForest Regressor R² =====")
r2_linear = r2_score(y_test, y_pred)
r2_rf = r2_score(y_test, new_pred)

print()

print("Linear Regression R² ", r2_linear )
print("RandomForest Regressor R² ",r2_rf)

if r2_linear > r2_rf : 
    print("Linear Regression explains the variation in academic performance more effectively.")
else: 
     print("Random Forest Regressor explains the variation in academic performance more effectively.")
print("===================================================================================")
print()

# ------ Feature importance (random forest ) ------
print("====== Feature importance (Random Forest)======")
importance = new.feature_importances_

imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

imp_df.reset_index(drop=True, inplace=True)

print("\n[Feature Importance Ranking]")
print(imp_df)

plt.figure()
sns.barplot(x="Importance", y="Feature", data=imp_df, color="steelblue")                # FIGURE 8
plt.title("Feature Importance Ranking")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

print()    

# Verification Sample check
sample = pd.DataFrame([[5,4,3]], columns=X.columns)       
predicted = new.predict(sample)     #The model performs best within observed data ranges and may not accurately predict extreme scenarios.
print("Predicted Marks for Sample Profile [5,4,3]: ", predicted)

# Actual marks vs predicted marks
print("===== Comparison between Actual Marks and predicted Marks =====")
print()
comparison = pd.DataFrame({"Actual" : y_test.values , "Predicted" : new_pred})
print(comparison.tail(5))
print('Model Absolute error (MAE) : ', mean_absolute_error(y_test, new_pred))
print()

print("===== Summary Performance Evaluation =====")
print("R² score of Linear Regression: ", r2_linear)
print("R² score of Random Forest Regressor: ", r2_rf)
print("Model Absolute error (MAE) ", mean_absolute_error(y_test, new_pred) )
print()
# Adjusted final output string to accurately reference the top-performing Random Forest model parameters
mae = mean_absolute_error(y_test, new_pred)
print (f" The model explains {r2_rf*100: .1f} % of variation & predicts marks with an \n average prediction error of approximately {mae: .1f}   marks ")


# Scatter plot
plt.figure()
plt.scatter(y_test, new_pred)                                     # FIGURE 9
plt.xlabel( "Actual Marks")
plt.ylabel(" Predicted Marks")
plt.title( "Actual vs Predicted")
plt.tight_layout()
plt.show()


# Theoretical Limitation
print()
print("\n[Limitation]")
print("Model struggles to predict extreme scores, indicating missing behavioral factors.")


#Summary
print()
print()
print("--- Data Summary ---")
print(df.describe(include="all"))

#Additional things
print("\n --- Primary Smartphone Usage Profiles ---")
print(df.groupby("Phone Usage")["Marks"].mean())

plt.figure()
df.groupby("Phone Usage")["Marks"].mean().plot(kind="bar")            # FIGURE 10
plt.xlabel("Phone Usage")
plt.ylabel("Marks")
plt.title("Phone Usage VS Marks")
plt.tight_layout()
plt.show()

print()

print("\n --- High Performing Academic Outliers Profile ---")
top = df[df["Marks"] > 85]
print(top.describe()) 
print("=========================================================")
