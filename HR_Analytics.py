import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "Human Resources.csv"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("Dataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. DATA CLEANING
# ============================================================

# Convert date columns safely.
# utc=True handles mixed date/time formats consistently.

df["birthdate"] = pd.to_datetime(
    df["birthdate"],
    errors="coerce",
    format="mixed",
    utc=True
)

df["hire_date"] = pd.to_datetime(
    df["hire_date"],
    errors="coerce",
    format="mixed",
    utc=True
)

df["termdate"] = pd.to_datetime(
    df["termdate"],
    errors="coerce",
    format="mixed",
    utc=True
)


# Clean text columns

text_columns = [
    "gender",
    "race",
    "department",
    "jobtitle",
    "location",
    "location_city",
    "location_state"
]

for column in text_columns:
    df[column] = (
        df[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )


# Remove duplicate records

df = df.drop_duplicates()


print("\nDate Columns:")
print(df[
    ["birthdate", "hire_date", "termdate"]
].dtypes)


# ============================================================
# 4. EMPLOYEE STATUS
# ============================================================

df["status"] = df["termdate"].isna().map({
    True: "Active",
    False: "Terminated"
})

print("\nEmployee Status:")
print(df["status"].value_counts())


# ============================================================
# 5. AGE
# ============================================================

today = pd.Timestamp.now(tz="UTC")

df["age"] = (
    (today - df["birthdate"]).dt.days / 365.25
).round(1)


# ============================================================
# 6. EMPLOYEE TENURE
# ============================================================

# Terminated employees:
#     hire date → termination date
#
# Active employees:
#     hire date → today

end_date = df["termdate"].fillna(today)

df["tenure_years"] = (
    (end_date - df["hire_date"]).dt.days / 365.25
).round(1)


# ============================================================
# 7. KEY WORKFORCE METRICS
# ============================================================

total_employees = len(df)

active_employees = (
    df["status"] == "Active"
).sum()

terminated_employees = (
    df["status"] == "Terminated"
).sum()

termination_rate = (
    terminated_employees /
    total_employees
) * 100


average_age = df["age"].mean()

average_tenure = df["tenure_years"].mean()


print("\n")
print("=" * 50)
print("HR WORKFORCE METRICS")
print("=" * 50)

print(f"Total Employees: {total_employees}")
print(f"Active Employees: {active_employees}")
print(f"Terminated Employees: {terminated_employees}")
print(f"Overall Termination Rate: {termination_rate:.2f}%")
print(f"Average Employee Age: {average_age:.1f} years")
print(f"Average Employee Tenure: {average_tenure:.1f} years")


# ============================================================
# 8. EMPLOYEES BY DEPARTMENT
# ============================================================

department_count = (
    df["department"]
    .value_counts()
)

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
    os.path.join(
        RESULTS_DIR,
        "employees_by_department.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 9. EMPLOYEE STATUS DISTRIBUTION
# ============================================================

status_count = (
    df["status"]
    .value_counts()
)

plt.figure(figsize=(8, 5))

status_count.plot(kind="bar")

plt.title("Employee Status Distribution")
plt.xlabel("Employee Status")
plt.ylabel("Number of Employees")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "employee_status_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 10. TERMINATION RATE BY DEPARTMENT
# ============================================================

department_termination = (
    df.groupby("department")["status"]
    .apply(
        lambda x:
        (x == "Terminated").mean() * 100
    )
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
    os.path.join(
        RESULTS_DIR,
        "termination_rate_by_department.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 11. EMPLOYEES BY GENDER
# ============================================================

gender_count = (
    df["gender"]
    .value_counts()
)

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
    os.path.join(
        RESULTS_DIR,
        "employees_by_gender.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 12. TERMINATION RATE BY GENDER
# ============================================================

gender_termination = (
    df.groupby("gender")["status"]
    .apply(
        lambda x:
        (x == "Terminated").mean() * 100
    )
    .sort_values(ascending=False)
)

print("\nTermination Rate by Gender:")
print(gender_termination)


plt.figure(figsize=(8, 5))

gender_termination.plot(kind="bar")

plt.title("Termination Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Termination Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "termination_rate_by_gender.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 13. AGE DISTRIBUTION
# ============================================================

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
    os.path.join(
        RESULTS_DIR,
        "employee_age_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 14. TOP JOB TITLES
# ============================================================

job_title_count = (
    df["jobtitle"]
    .value_counts()
    .head(15)
    .sort_values()
)

print("\nTop 15 Job Titles:")
print(job_title_count.sort_values(ascending=False))


plt.figure(figsize=(10, 7))

job_title_count.plot(kind="barh")

plt.title(
    "Top 15 Job Titles by Employee Count"
)

plt.xlabel("Number of Employees")
plt.ylabel("Job Title")
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "top_job_titles.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 15. HIRING TREND
# ============================================================

hire_year = (
    df["hire_date"]
    .dt.year
    .value_counts()
    .sort_index()
)

print("\nEmployees Hired by Year:")
print(hire_year)


plt.figure(figsize=(10, 5))

hire_year.plot(
    kind="line",
    marker="o"
)

plt.title("Employee Hiring Trend")
plt.xlabel("Hire Year")
plt.ylabel("Number of Employees Hired")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "hiring_trend.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 16. TERMINATION TREND
# ============================================================

termination_year = (
    df.loc[
        df["status"] == "Terminated",
        "termdate"
    ]
    .dt.year
    .value_counts()
    .sort_index()
)

print("\nTerminations by Year:")
print(termination_year)


plt.figure(figsize=(10, 5))

termination_year.plot(
    kind="line",
    marker="o"
)

plt.title("Employee Termination Trend")
plt.xlabel("Termination Year")
plt.ylabel("Number of Terminations")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "termination_trend.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 17. HR FINDINGS
# ============================================================

highest_termination_department = (
    department_termination.index[0]
)

highest_termination_rate = (
    department_termination.iloc[0]
)

most_common_job_title = (
    df["jobtitle"]
    .value_counts()
    .index[0]
)


print("\n")
print("=" * 50)
print("KEY HR FINDINGS")
print("=" * 50)

print(
    f"Total Employees: "
    f"{total_employees}"
)

print(
    f"Active Employees: "
    f"{active_employees}"
)

print(
    f"Terminated Employees: "
    f"{terminated_employees}"
)

print(
    f"Overall Termination Rate: "
    f"{termination_rate:.2f}%"
)

print(
    f"Average Employee Age: "
    f"{average_age:.1f} years"
)

print(
    f"Average Employee Tenure: "
    f"{average_tenure:.1f} years"
)

print(
    f"Department with Highest Termination Rate: "
    f"{highest_termination_department}"
)

print(
    f"Highest Department Termination Rate: "
    f"{highest_termination_rate:.2f}%"
)

print(
    f"Most Common Job Title: "
    f"{most_common_job_title}"
)


# ============================================================
# 18. SAVE CLEANED DATA
# ============================================================

cleaned_data_path = os.path.join(
    RESULTS_DIR,
    "cleaned_hr_data.csv"
)

df.to_csv(
    cleaned_data_path,
    index=False
)


# ============================================================
# 19. COMPLETION MESSAGE
# ============================================================

print("\n")
print("=" * 50)
print("HR ANALYTICS COMPLETED SUCCESSFULLY")
print("=" * 50)

print("\nResults saved in:")

print(
    os.path.join(
        RESULTS_DIR,
        "employees_by_department.png"
    )
)

print(
    os.path.join(
        RESULTS_DIR,
        "employee_status_distribution.png"
    )
)

print(
    os.path.join(
        RESULTS_DIR,
        "termination_rate_by_department.png"
    )
)

print(
    os.path.join(
        RESULTS_DIR,
        "employees_by_gender.png"
    )
)

print(
    os.path.join(
        RESULTS_DIR,
        "termination_rate_by_gender.png"
    )
)

print(
    os.path.join(
        RESULTS_DIR,
        "employee_age_distribution.png"
    )
)

print(
    os.path.join(
        RESULTS_DIR,
        "top_job_titles.png"
    )
)

print(
    os.path.join(
        RESULTS_DIR,
        "hiring_trend.png"
    )
)

print(
    os.path.join(
        RESULTS_DIR,
        "termination_trend.png"
    )
)

print(
    cleaned_data_path
)
