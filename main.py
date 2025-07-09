import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Membuat dataset sederhana: tinggi badan & berat badan
np.random.seed(42)
height = np.random.randint(140, 190, 100)  # Tinggi badan dalam cm
weight = np.random.randint(40, 90, 100)    # Berat badan dalam kg

# Label: 1 jika tinggi + berat > 200, 0 jika tidak
label = (height + weight > 200).astype(int)

# Konversi ke DataFrame
df = pd.DataFrame({
    "Height": height,
    "Weight": weight,
    "Label": label
})
print(df.head())

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    df[["Height", "Weight"]], df["Label"], test_size=0.2, random_state=42
)

# Buat model Decision Tree
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, Y_train)

# Prediksi pada test set
Y_pred = model.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(Y_test, Y_pred))

# Visualisasi pohon keputusan
from sklearn.tree import plot_tree
plt.figure(figsize=(10, 5))
plot_tree(
    model,
    feature_names=["Height", "Weight"],
    class_names=["Not Over 200", "Over 200"],
    filled=True
)
plt.show()

# Simpan ke file (untuk ditampilkan di GitHub Codespaces / Jupyter / Markdown)
plt.savefig("decision_tree_plot.png")
print("Gambar decision tree disimpan sebagai 'decision_tree_plot.png'")

# Simpan model ke file
with open("models/decision_tree_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model berhasil disimpan sebagai 'decision_tree_model.pkl'")
