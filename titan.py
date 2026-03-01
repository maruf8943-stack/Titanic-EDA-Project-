
# TITANIC EDA PROJECT



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("titanic_sample.csv")

print("First 5 Rows:")
print(df.head())

print("\nShape of Dataset:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


# Missing Value 


print("\nMissing Values:")
print(df.isnull().sum())


df["Age"].fillna(df["Age"].median(), inplace=True)


df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())




print("\nSurvival Count:")
print(df["Survived"].value_counts())

df["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("Survived (0=No, 1=Yes)")
plt.ylabel("Count")
plt.show()

# Gender Distribution
print("\nGender Distribution:")
print(df["Sex"].value_counts())

df["Sex"].value_counts().plot(kind="bar")
plt.title("Gender Distribution")
plt.show()

# Age Distribution
df["Age"].plot(kind="hist", bins=10)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.show()


# Survival by Gender
print("\nSurvival by Gender:")
print(pd.crosstab(df["Sex"], df["Survived"]))

pd.crosstab(df["Sex"], df["Survived"]).plot(kind="bar")
plt.title("Survival by Gender")
plt.show()

# Survival by Passenger Class
print("\nSurvival by Class:")
print(pd.crosstab(df["Pclass"], df["Survived"]))

pd.crosstab(df["Pclass"], df["Survived"]).plot(kind="bar")
plt.title("Survival by Passenger Class")
plt.show()



print("\nFare Statistics:")
print(df["Fare"].describe())

df["Fare"].plot(kind="box")
plt.title("Fare Distribution (Boxplot)")
plt.show()


print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()



# Create Family Size Feature
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

print("\nFamily Size Added:")
print(df[["SibSp","Parch","FamilySize"]].head())

print("\nSurvival by Family Size:")
print(pd.crosstab(df["FamilySize"], df["Survived"]))



df.drop(["Cabin","Ticket","Name"], axis=1, inplace=True)

print("\nFinal Dataset:")
print(df.head())


print("\nEDA Project Completed Successfully ")
