import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Iris veri setini CSV dosyasından yükleme
df = pd.read_csv('IRIS.csv')

# Özellikleri ve hedefi ayırma
X = df.drop('species', axis=1)  # 'species' dışındaki sütunlar--uzunluk ve genişlik gibi özellik değerlerini içerir (bağımsız değişkenler).
y = df['species']               # 'species' sütunu hedef değişken (bağımlı değişkenimiz)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modeli oluşturma ve eğitme
# Bu model, sınıflandırma işlemini karar ağaçları topluluğu yöntemi ile yapar.
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test verisi ile tahmin yapma
y_pred = model.predict(X_test)

# Karmaşıklık matrisini hesaplama
#Gerçek etiketler (y_test) ile tahmin edilen etiketler (y_pred)
# arasındaki karşılaştırmayı yaparak karmaşıklık matrisini oluşturur
conf_matrix = confusion_matrix(y_test, y_pred)
print("Karmaşıklık Matrix:\n", conf_matrix)


# Sınıfların hangi isimlere sahip olduğu-çıktıya yazdırmak için
unique_classes = model.classes_

# Hassasiyet, duyarlılık, F1 skoru metrikleri
report = classification_report(y_test, y_pred, target_names=unique_classes)
print("Sınıflandırma Raporu:\n", report)

