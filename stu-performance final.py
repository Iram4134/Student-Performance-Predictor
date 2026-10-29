
import tkinter as tk
from tkinter import messagebox
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ==========================================
# 1. SAMPLE DATASET
# ==========================================
data = {
    "Marks": [
        45, 50, 55, 60, 65, 70, 75, 80, 85, 90,
        48, 58, 68, 78, 88, 52, 62, 72, 82, 92
    ],
    "Attendance": [
        60, 65, 70, 75, 80, 85, 90, 95, 98, 99,
        62, 72, 78, 88, 96, 68, 76, 84, 93, 100
    ],
    "Final_Score": [
        48, 52, 57, 62, 67, 73, 78, 83, 89, 95,
        50, 60, 70, 81, 92, 55, 65, 75, 86, 97
    ]
}

df = pd.DataFrame(data)


# ==========================================
# 2. TRAIN LINEAR REGRESSION MODEL
# ==========================================
X = df[["Marks", "Attendance"]]
y = df["Final_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# RMSE without NumPy
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)


# ==========================================
# 3. COLOURS AND WINDOW
# ==========================================
BG = "#F3F6FB"
NAVY = "#172554"
BLUE = "#3563E9"
GREEN = "#13805C"
TEXT = "#334155"
MUTED = "#64748B"
WHITE = "#FFFFFF"
BORDER = "#E2E8F0"

root = tk.Tk()
root.title("Student Performance Predictor")
root.geometry("1050x720")
root.minsize(900, 650)
root.configure(bg=BG)


# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def make_label(parent, text, size=11, color=TEXT,
               bold=False, bg=WHITE):
    return tk.Label(
        parent,
        text=text,
        font=("Arial", size, "bold" if bold else "normal"),
        fg=color,
        bg=bg
    )


def draw_graph(marks, attendance, score):
    graph.delete("all")

    width = max(graph.winfo_width(), 420)
    height = max(graph.winfo_height(), 300)

    left = 52
    right = width - 25
    top = 48
    bottom = height - 65

    graph.create_text(
        width / 2, 20,
        text="STUDENT PERFORMANCE OVERVIEW",
        font=("Arial", 12, "bold"),
        fill=NAVY
    )

    # Grid and y-axis labels
    for value in range(0, 101, 20):
        y_pos = bottom - (value / 100) * (bottom - top)

        graph.create_line(
            left, y_pos, right, y_pos,
            fill=BORDER, dash=(3, 3)
        )

        graph.create_text(
            left - 12, y_pos,
            text=str(value),
            anchor="e",
            font=("Arial", 9),
            fill=MUTED
        )

    graph.create_line(
        left, top, left, bottom,
        fill="#94A3B8"
    )
    graph.create_line(
        left, bottom, right, bottom,
        fill="#94A3B8"
    )

    items = [
        ("Marks", marks, "#60A5FA"),
        ("Attendance", attendance, "#F5B942"),
        ("Predicted", score, "#34B78A")
    ]

    slot = (right - left) / 3
    bar_width = min(55, slot * 0.48)

    for index, (label, value, color) in enumerate(items):
        center = left + slot * (index + 0.5)
        bar_height = (value / 100) * (bottom - top)

        x1 = center - bar_width / 2
        x2 = center + bar_width / 2
        y1 = bottom - bar_height

        graph.create_rectangle(
            x1, y1, x2, bottom,
            fill=color,
            outline=""
        )

        graph.create_text(
            center, max(y1 - 15, top - 5),
            text=f"{value:.1f}",
            font=("Arial", 10, "bold"),
            fill=NAVY
        )

        graph.create_text(
            center, bottom + 22,
            text=label,
            font=("Arial", 9, "bold"),
            fill=TEXT
        )


def predict_score():
    try:
        marks = float(marks_entry.get().strip())
        attendance = float(attendance_entry.get().strip())

        if not 0 <= marks <= 100:
            messagebox.showerror(
                "Invalid Marks",
                "Enter marks between 0 and 100."
            )
            return

        if not 0 <= attendance <= 100:
            messagebox.showerror(
                "Invalid Attendance",
                "Enter attendance between 0 and 100."
            )
            return

        student = pd.DataFrame({
            "Marks": [marks],
            "Attendance": [attendance]
        })

        score = model.predict(student)[0]
        score = max(0.0, min(100.0, float(score)))

        result_value.config(text=f"{score:.2f}")
        result_status.config(
            text="Prediction completed successfully",
            fg=GREEN
        )

        draw_graph(marks, attendance, score)

    except ValueError:
        messagebox.showerror(
            "Invalid Input",
            "Please enter valid numeric values."
        )


def clear_fields():
    marks_entry.delete(0, tk.END)
    attendance_entry.delete(0, tk.END)

    result_value.config(text="--")
    result_status.config(
        text="Enter student details to begin",
        fg=MUTED
    )

    graph.delete("all")
    graph.create_text(
        max(graph.winfo_width(), 420) / 2,
        max(graph.winfo_height(), 300) / 2,
        text="Your performance graph will appear here",
        font=("Arial", 11),
        fill=MUTED
    )

    marks_entry.focus()


# ==========================================
# 5. HEADER
# ==========================================
header = tk.Frame(root, bg=NAVY, height=105)
header.pack(fill="x")
header.pack_propagate(False)

make_label(
    header,
    "STUDENT PERFORMANCE",
    21, WHITE, True, NAVY
).pack(anchor="w", padx=28, pady=(20, 3))

make_label(
    header,
    "Predict final scores using machine learning",
    11, "#CBD5E1", False, NAVY
).pack(anchor="w", padx=29)


# Main content area
content = tk.Frame(root, bg=BG)
content.pack(fill="both", expand=True, padx=22, pady=20)

content.grid_columnconfigure(0, weight=1, uniform="panel")
content.grid_columnconfigure(1, weight=2, uniform="panel")
content.grid_rowconfigure(0, weight=1)


# ==========================================
# 6. LEFT PANEL: INPUT AND RESULT
# ==========================================
left_panel = tk.Frame(
    content, bg=WHITE,
    highlightbackground=BORDER,
    highlightthickness=1
)
left_panel.grid(
    row=0, column=0,
    sticky="nsew", padx=(0, 12)
)

make_label(
    left_panel, "Student Details",
    15, NAVY, True
).pack(anchor="w", padx=20, pady=(22, 5))

make_label(
    left_panel,
    "Enter the student's information below.",
    10, MUTED
).pack(anchor="w", padx=20, pady=(0, 20))


# Marks input
make_label(
    left_panel, "EXAM MARKS (0-100)",
    9, MUTED, True
).pack(anchor="w", padx=20, pady=(0, 7))

marks_entry = tk.Entry(
    left_panel,
    font=("Arial", 13),
    bg="#F8FAFC",
    fg=NAVY,
    relief="flat",
    highlightthickness=1,
    highlightbackground=BORDER,
    highlightcolor=BLUE
)
marks_entry.pack(fill="x", padx=20, ipady=10)


# Attendance input
make_label(
    left_panel, "ATTENDANCE (0-100%)",
    9, MUTED, True
).pack(anchor="w", padx=20, pady=(18, 7))

attendance_entry = tk.Entry(
    left_panel,
    font=("Arial", 13),
    bg="#F8FAFC",
    fg=NAVY,
    relief="flat",
    highlightthickness=1,
    highlightbackground=BORDER,
    highlightcolor=BLUE
)
attendance_entry.pack(fill="x", padx=20, ipady=10)


# Buttons
tk.Button(
    left_panel,
    text="Predict Final Score",
    command=predict_score,
    bg=BLUE,
    fg=WHITE,
    activebackground="#244AC5",
    activeforeground=WHITE,
    font=("Arial", 11, "bold"),
    relief="flat",
    cursor="hand2"
).pack(fill="x", padx=20, pady=(22, 9), ipady=9)

tk.Button(
    left_panel,
    text="Clear / Reset",
    command=clear_fields,
    bg="#EAF0FF",
    fg=BLUE,
    activebackground="#DCE6FF",
    font=("Arial", 10, "bold"),
    relief="flat",
    cursor="hand2"
).pack(fill="x", padx=20, ipady=8)


# Result card
result_card = tk.Frame(left_panel, bg="#ECFDF5")
result_card.pack(fill="x", padx=20, pady=(22, 8))

make_label(
    result_card, "PREDICTED FINAL SCORE",
    9, GREEN, True, "#ECFDF5"
).pack(anchor="w", padx=13, pady=(12, 2))

result_value = make_label(
    result_card, "--",
    28, GREEN, True, "#ECFDF5"
)
result_value.pack(anchor="w", padx=13)

make_label(
    result_card, "Out of 100",
    9, MUTED, False, "#ECFDF5"
).pack(anchor="w", padx=13, pady=(0, 5))

result_status = make_label(
    result_card, "Enter student details to begin",
    9, MUTED, False, "#ECFDF5"
)
result_status.pack(anchor="w", padx=13, pady=(0, 12))


# Model metrics
make_label(
    left_panel, "MODEL EVALUATION",
    10, NAVY, True
).pack(anchor="w", padx=20, pady=(12, 7))

metrics_text = (
    f"RMSE: {rmse:.2f}\n"
    f"R-squared: {r2:.3f}"
)

make_label(
    left_panel, metrics_text,
    10, MUTED, False
).pack(anchor="w", padx=20, pady=(0, 15))


# ==========================================
# 7. RIGHT PANEL: GRAPH
# ==========================================
right_panel = tk.Frame(
    content, bg=WHITE,
    highlightbackground=BORDER,
    highlightthickness=1
)
right_panel.grid(row=0, column=1, sticky="nsew")

make_label(
    right_panel, "Performance Analytics",
    15, NAVY, True
).pack(anchor="w", padx=20, pady=(22, 4))

make_label(
    right_panel,
    "Compare marks, attendance and predicted score.",
    10, MUTED
).pack(anchor="w", padx=20, pady=(0, 10))


# Built-in Tkinter canvas graph (no Matplotlib)
graph = tk.Canvas(
    right_panel,
    bg=WHITE,
    highlightthickness=0
)
graph.pack(fill="both", expand=True, padx=10, pady=10)

graph.create_text(
    250, 160,
    text="Your performance graph will appear here",
    font=("Arial", 11),
    fill=MUTED
)


# Redraw the graph when the window is resized
def on_graph_resize(event):
    # Preserve current graph data when resizing.
    if result_value.cget("text") != "--":
        try:
            marks = float(marks_entry.get())
            attendance = float(attendance_entry.get())
            score = float(result_value.cget("text"))
            draw_graph(marks, attendance, score)
        except ValueError:
            pass


graph.bind("<Configure>", on_graph_resize)


# ==========================================
# 8. FOOTER
# ==========================================
footer = tk.Label(
    root,
    text="STUDENT PERFORMANCE PREDICTOR  |  LINEAR REGRESSION",
    bg=BG,
    fg=MUTED,
    font=("Arial", 9)
)
footer.pack(pady=(0, 12))


# Start application
marks_entry.focus()
root.mainloop() 