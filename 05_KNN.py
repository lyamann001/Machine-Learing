
# Bu çalışmada bir teleküminikasyon firmasının 4 farklı müşteri paketine göre var olan kullanıcları sınıflandırarak yeni gelen kullanıcıya en uygun paketi tahmin etmekye çalışacağız.

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('data/teleCust1000t.csv')
print(df)

# Veri setinde her pakette sahip kaç kullanıcı var
print(df['custcat'].value_counts())

# KNN algoritmasının bizden beklediği veri tipi numpay dizisi olduğundan ve bizim veri setimiz dataframe olduğundan veri setimizde ki belirli alanların değerlerini alıp numpay array'ine dönüştüreceğiz
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values  # burada ki values property'si buradaki core.Series tipini ndarray'e dönüştürdü.
print(type(X))
print(X[0:5])
# Yularıdaki mantıkta paketlerimizi alalım
y = df["custcat"].values
print(type(y))
print(y[0:5])

# KNN algortimasında her bir noktanın bir birlerine olan mesafesini hesaplayacağımızdan veri kümelerimizde bulunan değerleri bir standarda oturmamız gerekmektedir. Çünkü ndarray tipinde ki "X" isimli veri kümemizi incelerseniz içerisinde bulunan değerlerin matematiksel olarak bir birlerinden çok farklı olduğunu göreceksiniz. Bu yüzden bu değerleri standartLaştırmalıyız.
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print(f'Train Sets: {X_train.shape} / {y_train.shape}')
print(f'Test Sets: {X_test.shape} / {y_test.shape}')

neigh = KNeighborsClassifier(n_neighbors=4).fit(X_train, y_train)
print(neigh)

y_result = neigh.predict(X_test)
print(y_result[0:5])

# Doğruluk değerlendirmesi: Çok etiketli sınıflandırmalarda doğruluk sınıflanrıdma puanı, alt küme doğrulaması yapmak için önemli bir yapıdır. Burada istatistik alanında yoğun olarak kullanılan Jaccard Similarity Index kullanarak bunu yapacağız. Yani "y_train" ile "neigh.predict(X_train)" ve "y_test, neigh.predict(X_test)" arasında ki benzerliği ve çeşitliliği göreceğiz.
print(f'Train Set Accuracy: {metrics.accuracy_score(y_train, neigh.predict(X_train))}')
print(f'Test Set Accuracy: {metrics.accuracy_score(y_test, neigh.predict(X_test))}')

# KNN'de K, incelenecek en yakın komşu sayısıdır. Yukarıda 33. satırda modeli oluştururken 4 değerini kullanmıştık. Peki kime göre neye göre 4. K için en doğru değeri nasıl seçebiliriz. Birinci adım train ve test setleri hazırlama bunu yaptık. İkinci adım ise K'yı 1 vererek bizim merak ettiğimiz ve görmek istediğimiz değere kadar K değerini değiştirmektir yani en iyi sonucu alana kadar. Bunu için bir loop kurarak tek tek K değerini 1 arttırak görelim.

K_number = int(input("Please type K number: "))
# Aşağıda ki 2 array'i for loop içerisinde elde edeceğim sonuçlar ile doldurmak için kullanacağım.
jsi_acc = np.zeros((K_number - 1))
std_acc = np.zeros((K_number - 1))
# Her bir k değeri için modeli train ve test edeceğiz. Süreci uzun tutmamak için K=10 olarak düşünülecektir.
for k in range(1, K_number):
    neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    y_predict = neigh.predict(X_test)

    jsi_acc[k - 1] = metrics.accuracy_score(y_test, y_predict)
    std_acc[k - 1] = np.std(y_predict == y_test) / np.sqrt(y_predict.shape[0])

print(jsi_acc)

plt.plot(range(1, K_number), jsi_acc, 'g')
plt.fill_between(range(1, K_number), jsi_acc - 1 * std_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy', 'std'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbor')
plt.grid()
plt.tight_layout()
plt.show()

print(f'The best accuracy was with: {jsi_acc.max()}, with k={jsi_acc.argmax()+1}')

neigh = KNeighborsClassifier(n_neighbors=16).fit(X_train, y_train)
print(neigh)

y_result = neigh.predict(X_test)
print(y_result[0:16])