# ==========================================
# PERSONALISED LEARNING RECOMMENDATION SYSTEM
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set_theme(style="whitegrid")

# ------------------------------------------
# 1. CREATE DATASET
# ------------------------------------------

data = {
    "marks": [85, 62, 74, 48, 90, 55, 78, 67, 45, 88,
              72, 59, 81, 64, 92, 52, 76, 69, 58, 87],

    "attendance": [92, 75, 84, 65, 95, 70, 88, 80, 60, 93,
                   85, 68, 90, 78, 96, 64, 86, 82, 72, 94],

    "study_hours": [5, 2, 3, 1, 6, 2, 4, 3, 1, 5,
                    4, 2, 5, 3, 6, 1, 4, 3, 2, 5],

    "interest": [
        "Programming", "Database", "Programming", "Web Development",
        "Data Science", "Database", "Programming", "Web Development",
        "Programming", "Data Science", "Programming", "Database",
        "Data Science", "Web Development", "Data Science", "Database",
        "Programming", "Web Development", "Database", "Data Science"
    ],

    "learning_style": [
        "Video", "Notes", "Video", "Practical", "Video",
        "Notes", "Practical", "Video", "Notes", "Practical",
        "Video", "Notes", "Video", "Practical", "Video",
        "Notes", "Practical", "Video", "Notes", "Practical"
    ]
}

df = pd.DataFrame(data)

# ------------------------------------------
# 2. CREATE LEARNING LEVEL
# ------------------------------------------

def get_level(marks):

    if marks < 60:
        return "Beginner"

    elif marks < 80:
        return "Intermediate"

    else:
        return "Advanced"


df["level"] = df["marks"].apply(get_level)

print("\n========== PERSONALISED LEARNING RECOMMENDATION SYSTEM DATASET ==========")
print(df.to_string(index=False))

# ------------------------------------------
# 3. DATA PREPROCESSING
# ------------------------------------------

encoder_interest = LabelEncoder()
encoder_style = LabelEncoder()

df["interest_code"] = encoder_interest.fit_transform(df["interest"])

df["style_code"] = encoder_style.fit_transform(df["learning_style"])

X = df[
    ["marks", "attendance", "study_hours",
     "interest_code", "style_code"]
]

y = df["level"]

# ------------------------------------------
# 4. TRAIN AND TEST DATA
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# ------------------------------------------
# 5. MACHINE LEARNING MODEL
# ------------------------------------------

model = DecisionTreeClassifier(
    random_state=42,
    max_depth=4
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ------------------------------------------
# 6. MODEL EVALUATION
# ------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL RESULT ==========")

print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# ------------------------------------------
# 7. CONFUSION MATRIX HEATMAP
# ------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Beginner", "Intermediate", "Advanced"]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Beginner", "Intermediate", "Advanced"],
    yticklabels=["Beginner", "Intermediate", "Advanced"]
)

plt.title("Personalised Learning Recommendation System - Confusion Matrix")
plt.xlabel("Predicted Level")
plt.ylabel("Actual Level")
plt.tight_layout()
plt.show()

# ------------------------------------------
# 8. MARKS DISTRIBUTION
# ------------------------------------------

plt.figure(figsize=(7, 5))

sns.histplot(
    df["marks"],
    bins=8,
    kde=True
)

plt.title("Student Marks Distribution")
plt.xlabel("Marks")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.show()

# ------------------------------------------
# 9. STUDY HOURS VS MARKS
# ------------------------------------------

plt.figure(figsize=(7, 5))

sns.scatterplot(
    x="study_hours",
    y="marks",
    data=df,
    hue="level",
    s=100
)

plt.title("Study Hours vs Marks")
plt.xlabel("Daily Study Hours")
plt.ylabel("Marks")

plt.tight_layout()
plt.show()

# ------------------------------------------
# 10. STUDENT INTEREST
# ------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    x="interest",
    data=df,
    hue="interest",
    legend=False
)

plt.title("Student Interests")
plt.xlabel("Interest")
plt.ylabel("Number of Students")

plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# ------------------------------------------
# 11. LEARNING LEVEL DISTRIBUTION
# ------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    x="level",
    data=df,
    hue="level",
    order=["Beginner", "Intermediate", "Advanced"],
    legend=False
)

plt.title("Learning Level Distribution")
plt.xlabel("Learning Level")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.show()

# ------------------------------------------
# 12. CORRELATION HEATMAP
# ------------------------------------------

plt.figure(figsize=(8, 6))

numeric_df = df[
    ["marks", "attendance", "study_hours",
     "interest_code", "style_code"]
]

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Student Performance Correlation Heatmap")

plt.tight_layout()
plt.show()

# ------------------------------------------
# 13. RECOMMENDATION FUNCTION
# ------------------------------------------

def recommend(interest, level, learning_style):

    recommendations = {

        "Programming": {
            "Beginner": [
                "Python Basics",
                "Variables and Data Types",
                "Loops",
                "Functions"
            ],
            "Intermediate": [
                "Object Oriented Programming",
                "Data Structures",
                "Algorithms",
                "File Handling"
            ],
            "Advanced": [
                "Machine Learning",
                "Advanced Algorithms",
                "Artificial Intelligence",
                "System Design"
            ]
        },

        "Database": {
            "Beginner": [
                "DBMS Basics",
                "SQL Basics",
                "Database Keys",
                "ER Diagram"
            ],
            "Intermediate": [
                "Normalization",
                "SQL Joins",
                "Subqueries",
                "Transactions"
            ],
            "Advanced": [
                "Database Optimization",
                "Indexing",
                "Distributed Databases",
                "Database Security"
            ]
        },

        "Web Development": {
            "Beginner": [
                "HTML Basics",
                "CSS Basics",
                "JavaScript Basics",
                "Web Page Design"
            ],
            "Intermediate": [
                "JavaScript DOM",
                "React Basics",
                "Node.js",
                "REST API"
            ],
            "Advanced": [
                "Full Stack Development",
                "Authentication",
                "Cloud Deployment",
                "Web Security"
            ]
        },

        "Data Science": {
            "Beginner": [
                "Python for Data Science",
                "Statistics Basics",
                "Pandas",
                "Data Visualization"
            ],
            "Intermediate": [
                "Data Preprocessing",
                "Exploratory Data Analysis",
                "Machine Learning",
                "Regression"
            ],
            "Advanced": [
                "Deep Learning",
                "Neural Networks",
                "Natural Language Processing",
                "Advanced Machine Learning"
            ]
        }
    }

    print("\n========== LEARNING RECOMMENDATIONS ==========")

    topics = recommendations.get(interest, {}).get(level, [])

    for topic in topics:
        print("✔", topic)

    print("\nPreferred Learning Style:", learning_style)

    if learning_style == "Video":

        print("Recommended Resource: Video Tutorials")

    elif learning_style == "Notes":

        print("Recommended Resource: PDF Notes / Study Notes")

    else:

        print("Recommended Resource: Practical Exercises / Projects")


# ------------------------------------------
# 14. USER INPUT
# ------------------------------------------

print("\n\n====================================")
print("       PERSONALISED LEARNING RECOMMENDATION SYSTEM")
print("====================================")

print("\nEnter Your Academic Details")

marks = float(input("\nEnter your marks (0-100): "))

attendance = float(input("Enter attendance percentage (0-100): "))

study_hours = float(input("Enter daily study hours: "))

if not (0 <= marks <= 100 and 0 <= attendance <= 100 and study_hours >= 0):
    print("\nInvalid input! Please enter valid academic details.")

else:

    print("\nChoose your interest:")

    print("1. Programming")
    print("2. Database")
    print("3. Web Development")
    print("4. Data Science")

    choice = int(input("Enter choice (1-4): "))

    interest_list = {
        1: "Programming",
        2: "Database",
        3: "Web Development",
        4: "Data Science"
    }

    if choice not in interest_list:
        print("Invalid interest choice!")

    else:

        interest = interest_list[choice]

        print("\nChoose your learning style:")

        print("1. Video")
        print("2. Notes")
        print("3. Practical")

        style_choice = int(input("Enter choice (1-3): "))

        style_list = {
            1: "Video",
            2: "Notes",
            3: "Practical"
        }

        if style_choice not in style_list:

            print("Invalid learning style choice!")

        else:

            learning_style = style_list[style_choice]

            # ------------------------------------------
            # 15. PREDICT STUDENT LEVEL
            # ------------------------------------------

            interest_code = encoder_interest.transform([interest])[0]

            style_code = encoder_style.transform([learning_style])[0]

            user_data = pd.DataFrame(
                [[marks, attendance, study_hours,
                  interest_code, style_code]],
                columns=X.columns
            )

            predicted_level = model.predict(user_data)[0]

            # ------------------------------------------
            # 16. DISPLAY STUDENT RESULT
            # ------------------------------------------

            print("\n====================================")
            print("          STUDENT RESULT")
            print("====================================")

            print("Marks:", marks)

            print("Attendance:", attendance, "%")

            print("Study Hours:", study_hours)

            print("Interest:", interest)

            print("Learning Style:", learning_style)

            print("Predicted Learning Level:", predicted_level)

            # ------------------------------------------
            # 17. LEARNING RECOMMENDATIONS
            # ------------------------------------------

            recommend(
                interest,
                predicted_level,
                learning_style
            )

            print("\n====================================")
            print("       PERSONALISED LEARNING RECOMMENDATION SYSTEM COMPLETED")
            print("====================================")
