import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import pandas as pd
from model import UnifiedAdModel

class AdRecommenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ad Targeting System")
        self.root.geometry("600x500")

        try:
            self.users = pd.read_csv("ad_users.csv")
            with open("ad_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open("ad_vectorizer.pkl", "rb") as f:
                self.vectorizer = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load resources:\n{str(e)}")
            self.root.destroy()
            return

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Ad Recommendation System", 
                 font=('Helvetica', 14, 'bold')).pack(pady=10)

        selection_frame = ttk.Frame(main_frame)
        selection_frame.pack(fill=tk.X, pady=10)

        ttk.Label(selection_frame, text="Select User:").pack(side=tk.LEFT)
        self.user_var = tk.StringVar()
        self.user_dropdown = ttk.Combobox(selection_frame, 
                                          textvariable=self.user_var,
                                          state="readonly")
        self.user_dropdown.pack(side=tk.LEFT, padx=10)
        self.populate_users()

        ttk.Button(selection_frame, text="Show Recommendations",
                   command=self.show_recommendations).pack(side=tk.LEFT)

        self.profile_frame = ttk.LabelFrame(main_frame, text="User Profile", padding=10)
        self.profile_frame.pack(fill=tk.X, pady=10)

        self.results_frame = ttk.LabelFrame(main_frame, text="Recommended Ads", padding=10)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

    def populate_users(self):
        user_list = [
            f"ID {row.user_id}: {row.gender}, {row.age}, {row.location}"
            for _, row in self.users.iterrows()
        ]
        self.user_dropdown['values'] = user_list
        if user_list:
            self.user_dropdown.current(0)

    def show_recommendations(self):
        for widget in self.profile_frame.winfo_children():
            widget.destroy()
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        try:
            selection = self.user_var.get()
            user_id = int(selection.split()[1].replace(":", ""))  # ✅ Fix 1

            user = self.users[self.users['user_id'] == user_id].iloc[0]

            profile_text = f"Age: {user.age}\nGender: {user.gender}\nLocation: {user.location}\nInterests: {user.interests}"
            ttk.Label(self.profile_frame, text=profile_text).pack(anchor=tk.W)

            features = f"{user.interests} {user.gender} {user.location} {user.age}"
            probs_dict = self.model.predict_proba([features])

            recommendations = [
                (ad, prob) for ad, prob in probs_dict.items() if prob[0] > 0.3
            ]
            recommendations.sort(key=lambda x: x[1], reverse=True)

            if not recommendations:
                ttk.Label(self.results_frame, 
                          text="No strong recommendations found",
                          style='Warning.TLabel').pack()
                return

            for ad, prob in recommendations[:10]:
                frame = ttk.Frame(self.results_frame)
                frame.pack(fill=tk.X, pady=2)

                ttk.Label(frame, text=f"{ad.title()}:", width=20, anchor=tk.W).pack(side=tk.LEFT)

                meter = ttk.Progressbar(frame, length=200, value=int(prob[0] * 100), maximum=100)
                meter.pack(side=tk.LEFT, padx=5)

                ttk.Label(frame, text=f"{float(prob[0]):.2f}", width=6).pack(side=tk.LEFT)  # ✅ Fix 2

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate recommendations:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AdRecommenderApp(root)
    root.mainloop()
