import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/Human Resources.csv")

# -----------------------------
# 1. Basic Data Understanding
# -----------------------------

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nDataset Summary:")
print(df.describe())

# -----------------------------
# 2. Attrition Distribution
# -----------------------------

attrition_count = df["Attrition"].value_counts()

print("\nAttrition Distribution:")
print(attrition_count)

attrition_rate = (df["Attrition"] == "Yes").mean() * 100
print(f"\nOverall Attrition Rate: {attrition_rate:.2f}%")

# -----------------------------
# 3. Attrition by Department
# -----------------------------

department_attrition = pd.crosstab(
    df["Department"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nAttrition by Department:")
print(department_attrition)

# -----------------------------
# 4. Attrition by Job Role
# -----------------------------

role_attrition = pd.crosstab(
    df["JobRole"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nAttrition by Job Role:")
print(role_attrition)

# -----------------------------
# 5. Attrition by Overtime
# -----------------------------

overtime_attrition = pd.crosstab(
    df["OverTime"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nAttrition by Overtime:")
print(overtime_attrition)

# -----------------------------
# 6. Create Age Groups
# -----------------------------

bins = [17, 25, 35, 45, 55, 65]
labels = ["18-25", "26-35", "36-45", "46-55", "56-65"]

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=bins,
    labels=labels
)

age_attrition = pd.crosstab(
    df["AgeGroup"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nAttrition by Age Group:")
print(age_attrition)

# -----------------------------
# 7. Visualizations
# -----------------------------

sns.set_theme(style="whitegrid")

# Attrition distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Attrition")
plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.savefig("results/attrition_distribution.png", dpi=300)
plt.show()

# Attrition by department
plt.figure(figsize=(9, 5))
sns.barplot(
    data=df,
    x="Department",
    y=(df["Attrition"] == "Yes").astype(int)
)
plt.title("Attrition by Department")
plt.xlabel("Department")
plt.ylabel("Attrition Rate")
plt.tight_layout()
plt.savefig("results/attrition_by_department.png", dpi=300)
plt.show()

# Attrition by job role
role_rate = df.groupby("JobRole")["Attrition"].apply(
    lambda x: (x == "Yes").mean() * 100
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
role_rate.plot(kind="bar")
plt.title("Attrition Rate by Job Role")
plt.xlabel("Job Role")
plt.ylabel("Attrition Rate (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("results/attrition_by_job_role.png", dpi=300)
plt.show()

# Attrition by overtime
overtime_rate = df.groupby("OverTime")["Attrition"].apply(
    lambda x: (x == "Yes").mean() * 100
)

plt.figure(figsize=(8, 5))
overtime_rate.plot(kind="bar")
plt.title("Attrition Rate by Overtime")
plt.xlabel("Overtime")
plt.ylabel("Attrition Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("results/attrition_by_overtime.png", dpi=300)
plt.show()

print("\nHR Analytics completed successfully.")
