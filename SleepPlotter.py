
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class SleepPlotter:
    def __init__(self, df):
        self.df = df

    def plot_barplot(self):
        # Gerçek uyku sürelerini gösteren barplot
        plt.figure(figsize=(15, 8))
        plt.bar(self.df["name"], self.df["sleep_total"], color="skyblue")
        plt.xticks(rotation=90)
        plt.title("Animal Sleep Duration", fontsize=16)
        plt.xlabel("Animal Names", fontsize=12)
        plt.ylabel("Sleep Duration (hours)", fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_grouped_barplot(self):
        # Tahmini ve gerçek uyku sürelerini karşılaştıran grouped barplot
        df_long = pd.melt(self.df, id_vars="name", value_vars=["sleep_total", "predicted_sleep_total"],
                          var_name="Sleep_Type", value_name="Duration")

        plt.figure(figsize=(15, 8))
        sns.barplot(data=df_long, x="name", y="Duration", hue="Sleep_Type", palette="coolwarm")
        plt.xticks(rotation=90)
        plt.title("Comparison of Actual and Predicted Sleep Durations", fontsize=16)
        plt.xlabel("Animal Names", fontsize=12)
        plt.ylabel("Duration (hours)", fontsize=12)
        plt.tight_layout()
        plt.show()
