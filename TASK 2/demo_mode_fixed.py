"""
Senior Citizen Identification System - Built-in Photo Demo
-----------------------------------------------------------
All four small portrait photographs are embedded in this Python file.
No UTKFace dataset download and no model training are needed.

Run in VS Code:
    python demo_mode.py

Important: visitor names, ages and gender labels are illustrative sample
records. This demo shows the GUI and age >= 60 classification rule; it does
not perform real AI face, age or gender prediction.
"""
import base64
import csv
import io
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit(
        "Pillow is required for displaying photos. Install it with:\n"
        "python -m pip install pillow"
    )

# Valid JPEG data embedded directly in this file.
# Do not edit these strings.
PHOTOS = {
    "young_woman": (
        "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"
        "MxYEQvz+lSpSN//Z"
    ),
    "young_man": (
        "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"
        "OxQnwzRt+9FFVMpQo4/mX60FTkcV+tFFER//2Q=="
    ),
    "older_man": (
        "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"
        "McgJBo3SS6kA8xUqUDVhH//Z"
    ),
    "older_woman": (
        "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"
        "Y5RLlvs9w//Z"
    ),
}

SAMPLE_VISITORS = [
    {"id": "VIS-001", "name": "Aditi Sharma", "age": 23, "gender": "Female", "photo": "young_woman", "time": "10:02:15"},
    {"id": "VIS-002", "name": "Rahul Mehta", "age": 27, "gender": "Male", "photo": "young_man", "time": "10:05:30"},
    {"id": "VIS-003", "name": "Kamla Devi", "age": 72, "gender": "Female", "photo": "older_woman", "time": "10:08:12"},
    {"id": "VIS-004", "name": "Vivek Patel", "age": 34, "gender": "Male", "photo": "young_man", "time": "10:11:45"},
    {"id": "VIS-005", "name": "Ramesh Shah", "age": 68, "gender": "Male", "photo": "older_man", "time": "10:15:08"},
    {"id": "VIS-006", "name": "Neha Gupta", "age": 29, "gender": "Female", "photo": "young_woman", "time": "10:18:37"},
    {"id": "VIS-007", "name": "Sarojben Patel", "age": 63, "gender": "Female", "photo": "older_woman", "time": "10:22:19"},
    {"id": "VIS-008", "name": "Karan Joshi", "age": 25, "gender": "Male", "photo": "young_man", "time": "10:25:41"},
]

class SeniorCitizenDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Senior Citizen Identification System")
        self.root.geometry("1120x700")
        self.root.minsize(900, 560)
        self.rows = []
        self.photo_refs = {}
        self._load_photos()
        self._build_window()
        self.load_records()

    def _load_photos(self):
        """Decode the embedded JPEGs and keep Tk image references alive."""
        for key, encoded in PHOTOS.items():
            try:
                image_bytes = base64.b64decode(encoded, validate=True)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image = image.resize((112, 72), Image.Resampling.LANCZOS)
                self.photo_refs[key] = ImageTk.PhotoImage(image)
            except Exception as exc:
                raise RuntimeError(f"Could not load embedded photo '{key}': {exc}") from exc

    def _build_window(self):
        self.root.configure(bg="#edf3f8")
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Treeview", rowheight=78, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

        header = tk.Frame(self.root, bg="#124b73")
        header.pack(fill="x")
        tk.Label(
            header, text="Senior Citizen Identification System",
            bg="#124b73", fg="white", padx=20, pady=16,
            font=("Segoe UI", 20, "bold")
        ).pack(side="left")
        tk.Label(
            header, text=datetime.now().strftime("%d-%m-%Y  %I:%M:%S %p"),
            bg="#124b73", fg="white", padx=18,
            font=("Segoe UI", 10)
        ).pack(side="right")

        stats = tk.Frame(self.root, bg="#edf3f8", padx=12, pady=10)
        stats.pack(fill="x")
        self.stat_vars = [tk.StringVar(value="0") for _ in range(4)]
        stat_titles = ["Total Visitors", "Senior Citizens (60+)", "Male", "Female"]
        for title, var in zip(stat_titles, self.stat_vars):
            card = tk.Frame(stats, bg="white", bd=1, relief="solid", padx=14, pady=8)
            card.pack(side="left", fill="x", expand=True, padx=5)
            tk.Label(card, text=title, bg="white", fg="#183a57",
                     font=("Segoe UI", 10, "bold")).pack(anchor="w")
            tk.Label(card, textvariable=var, bg="white", fg="#124b73",
                     font=("Segoe UI", 20, "bold")).pack(anchor="w")

        body = tk.Frame(self.root, bg="#edf3f8", padx=14, pady=5)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg="white", bd=1, relief="solid", padx=10, pady=10)
        left.pack(side="left", fill="y", padx=(0, 10))
        tk.Label(left, text="Selected Visitor", bg="#124b73", fg="white",
                 font=("Segoe UI", 12, "bold"), padx=12, pady=8).pack(fill="x")

        self.preview = tk.Label(left, bg="#e7eef5", text="Select a visitor",
                                width=34, height=12)
        self.preview.pack(pady=10)
        self.details = tk.StringVar(value="Select a row to view visitor details.")
        tk.Label(left, textvariable=self.details, bg="white", fg="#183a57",
                 font=("Segoe UI", 11), justify="left", wraplength=255).pack(pady=8)

        ttk.Button(left, text="Load Sample Visitors",
                   command=self.load_records).pack(fill="x", pady=5)
        ttk.Button(left, text="Export CSV",
                   command=self.export_csv).pack(fill="x", pady=5)
        ttk.Button(left, text="Clear Table",
                   command=self.clear_table).pack(fill="x", pady=5)

        right = tk.Frame(body, bg="white", bd=1, relief="solid", padx=8, pady=8)
        right.pack(side="left", fill="both", expand=True)
        tk.Label(right, text="Visitor Records", bg="#124b73", fg="white",
                 font=("Segoe UI", 12, "bold"), padx=12, pady=8).pack(fill="x")

        table_area = tk.Frame(right)
        table_area.pack(fill="both", expand=True, pady=(8, 0))
        columns = ("No.", "Visitor ID", "Name", "Age", "Gender", "Senior Citizen", "Visit Time")
        self.table = ttk.Treeview(table_area, columns=columns, show="tree headings")
        self.table.heading("#0", text="Photo")
        self.table.column("#0", width=120, minwidth=100, anchor="center")
        widths = (45, 90, 130, 55, 85, 120, 90)
        for col, width in zip(columns, widths):
            self.table.heading(col, text=col)
            self.table.column(col, width=width, minwidth=45, anchor="center")
        self.table.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(table_area, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.table.bind("<<TreeviewSelect>>", self.on_select)

        footer = tk.Label(
            self.root,
            text="Built-in sample photos and records • Senior citizen status uses the age 60+ rule",
            bg="#edf3f8", fg="#52677a", anchor="w", padx=18, pady=8
        )
        footer.pack(fill="x")

    def load_records(self):
        self.clear_table()
        for index, person in enumerate(SAMPLE_VISITORS, start=1):
            row = dict(person)
            row["senior"] = "Yes" if row["age"] >= 60 else "No"
            self.rows.append(row)
            self.table.insert(
                "", "end", iid=row["id"],
                image=self.photo_refs[row["photo"]],
                values=(index, row["id"], row["name"], row["age"],
                        row["gender"], row["senior"], row["time"])
            )

        total = len(self.rows)
        seniors = sum(r["senior"] == "Yes" for r in self.rows)
        males = sum(r["gender"] == "Male" for r in self.rows)
        for variable, value in zip(self.stat_vars, (total, seniors, males, total - males)):
            variable.set(str(value))
        self.details.set("Select a visitor row to view their photograph and details.")

    def on_select(self, _event=None):
        selected = self.table.selection()
        if not selected:
            return
        row = next((r for r in self.rows if r["id"] == selected[0]), None)
        if row is None:
            return
        image = self.photo_refs[row["photo"]]
        self.preview.configure(image=image, text="")
        self.preview.image = image
        self.details.set(
            f"Name: {row['name']}\\n"
            f"Age: {row['age']}\\n"
            f"Gender: {row['gender']}\\n"
            f"Senior Citizen: {row['senior']}"
        )

    def clear_table(self):
        for item in self.table.get_children():
            self.table.delete(item)
        self.rows = []
        for variable in self.stat_vars:
            variable.set("0")
        self.preview.configure(image="", text="Select a visitor")
        self.preview.image = None
        self.details.set("Table cleared. Click Load Sample Visitors.")

    def export_csv(self):
        if not self.rows:
            messagebox.showinfo("No records", "Load the sample visitors first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export visitor records",
            defaultextension=".csv",
            initialfile="visitor_records.csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return
        fields = ["id", "name", "age", "gender", "senior", "time"]
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in self.rows)
        messagebox.showinfo("Export complete", f"Saved {len(self.rows)} visitor records to:\\n{path}")

if __name__ == "__main__":
    window = tk.Tk()
    application = SeniorCitizenDemo(window)
    window.mainloop()
