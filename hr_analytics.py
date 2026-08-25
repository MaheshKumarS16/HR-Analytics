import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------
# 1. Load Dataset
# ---------------------------------------

df = pd.read_csv("data/Human Resources.csv")

print("Dataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ---------------------------------------
# 2. Data Cleaning
# ---------------------------------------

# Convert date columns
df["birthdate"] = pd.to_datetime(df["birthdate"], errors="coerce")
df["hire_date"] = pd.to_datetime(df["hire_date"], errors="coerce")
df["termdate"] = pd.to_datetime(df["termdate"], errors="coerce")

# Remove unnecessary whitespace from text columns
text_columns = [
    "gender",
    "department",
    "jobtitle",
    "location",
    "location_city",
    "location_state"
]

for column in text_columns:
    df[column] = df[column].astype(str).str.strip()

# ---------------------------------------
# 3. Employee Status
# ---------------------------------------

df["status"] = df["termdate"].isna().map({
    True: "Active",
    False: "Terminated"
})

print("\nEmployee Status:")
print(df["status"].value_counts())

# ---------------------------------------
# 4. Calculate Age
# ---------------------------------------

reference_date = pd.Timestamp.today()

df["age"] = (
    (reference_date - df["birthdate"]).dt.days / 365.25
).round(1)

# ---------------------------------------
# 5. Calculate Tenure
# ---------------------------------------

end_date = df["termdate"].fillna(reference_date)

df["tenure_years"] = (
    (end_date - df["hire_date"]).dt.days / 365.25
).round(1)

# ---------------------------------------
# 6. Overall Workforce Metrics
# ---------------------------------------

total_employees = len(df)

active_employees = (df["status"] == "Active").sum()

terminated_employees = (df["status"] == "Terminated").sum()

termination_rate = (
    terminated_employees / total_employees * 100
)

print("\nHR Workforce Metrics")
print("---------------------")
print("Total Employees:", total_employees)
print("Active Employees:", active_employees)
print("Terminated Employees:", terminated_employees)
print(f"Termination Rate: {termination_rate:.2f}%")

# ---------------------------------------
# 7. Employees by Department
# ---------------------------------------

department_count = df["department"].value_counts()

print("\nEmployees by Department:")
print(department_count)

plt.figure(figsize=(10, 6))

department_count.plot(kind="bar")

plt.title("Employees by Department")
plt.xlabel("Department")
plt.ylabel("Number of Employees")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    "results/employees_by_department.png",
    dpi=300
)

plt.show()

# ---------------------------------------
# 8. Employee Status Distribution
# ---------------------------------------

status_count = df["status"].value_counts()

plt.figure(figsize=(8, 5))

status_count.plot(kind="bar")

plt.title("Employee Status Distribution")
plt.xlabel("Employee Status")
plt.ylabel("Number of Employees")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "results/employee_status_distribution.png",
    dpi=300
)

plt.show()

# ---------------------------------------
# 9. Termination Rate by Department
# ---------------------------------------

department_termination = (
    df.groupby("department")["status"]
    .apply(lambda x: (x == "Terminated").mean() * 100)
    .sort_values(ascending=False)
)

print("\nTermination Rate by Department:")
print(department_termination)

plt.figure(figsize=(10, 6))

department_termination.plot(kind="bar")

plt.title("Termination Rate by Department")
plt.xlabel("Department")
plt.ylabel("Termination Rate (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    "results/termination_rate_by_department.png",
    dpi=300
)

plt.show()

# ---------------------------------------
# 10. Employees by Gender
# ---------------------------------------

gender_count = df["gender"].value_counts()

print("\nEmployees by Gender:")
print(gender_count)

plt.figure(figsize=(8, 5))

gender_count.plot(kind="bar")

plt.title("Employee Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Employees")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "results/employees_by_gender.png",
    dpi=300
)

plt.show()

# ---------------------------------------
# 11. Employee Age Distribution
# ---------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="age",
    bins=15
)

plt.title("Employee Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Employees")
plt.tight_layout()

plt.savefig(
    "results/employee_age_distribution.png",
    dpi=300
)

plt.show()

# ---------------------------------------
# 12. Employees by Job Title
# ---------------------------------------

job_title_count = df["jobtitle"].value_counts().head(15)

print("\nTop 15 Job Titles:")
print(job_title_count)

plt.figure(figsize=(10, 7))

job_title_count.sort_values().plot(kind="barh")

plt.title("Top 15 Job Titles by Employee Count")
plt.xlabel("Number of Employees")
plt.ylabel("Job Title")
plt.tight_layout()

plt.savefig(
    "results/top_job_titles.png",
    dpi=300
)

plt.show()

# ---------------------------------------
# 13. Hiring Trend by Year
# ---------------------------------------

hire_year = df["hire_date"].dt.year.value_counts().sort_index()

print("\nEmployees Hired by Year:")
print(hire_year)

plt.figure(figsize=(10, 5))

hire_year.plot(kind="line", marker="o")

plt.title("Employee Hiring Trend")
plt.xlabel("Hire Year")
plt.ylabel("Number of Employees")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/hiring_trend.png",
    dpi=300
)

plt.show()

# ---------------------------------------
# 14. Termination Trend by Year
# ---------------------------------------

termination_year = (
    df.loc[df["status"] == "Terminated", "termdate"]
    .dt.year
    .value_counts()
    .sort_index()
)

print("\nTerminations by Year:")
print(termination_year)

plt.figure(figsize=(10, 5))

termination_year.plot(kind="line", marker="o")

plt.title("Employee Termination Trend")
plt.xlabel("Termination Year")
plt.ylabel("Number of Terminations")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/termination_trend.png",
    dpi=300
)

plt.show()

# ---------------------------------------
# 15. Average Tenure
# ---------------------------------------

average_tenure = df["tenure_years"].mean()

print(
    f"\nAverage Employee Tenure: {average_tenure:.2f} years"
)

# ---------------------------------------
# 16. Save Cleaned Dataset
# ---------------------------------------

df.to_csv(
    "results/cleaned_hr_data.csv",
    index=False
)

print("\nHR Analytics completed successfully.")
