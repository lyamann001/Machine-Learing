
# Bu çalışmada hastaların farklı ilaçlara verdikleri yanıtları içeren bir veri seti üzerinden bir model kurarak, yeni gelen bir hastaya uygulanabilecek en iyi tedaviyi saptamaya çalışacağız. Amaç yeni gelen hastaya çok az test uygulayarak en iyi tedaviyi verecek ilacı belirleme.

import pandas as pd
import numpy as np
import pydotplus
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import preprocessing, tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from six import StringIO


df = pd.read_csv('data/drug.csv', delimiter=',')
print(df.head(10))

# df içerisinde ki değerleri ndarray dönüştürelim yani bir özellikler (features) matrixsi oluşturalım.
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:10])

# Veri setimizi incelediğimizde göreceğiniz gibi bazı özellikler kategoricaldır. Karar ağacı algoritmasında her bir classifier'ı yani kümeyi hesaplamak için Entropi kullanılacaktır yani matematiksel hesaplamalar yapılacaktır.
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:, 1] = le_sex.transform(X[:, 1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = le_BP.transform(X[:, 2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = le_Chol.transform(X[:, 3])

print(X[0:10])

y = df['Drug']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
print(f'Train Sets: {X_train.shape} / {y_train.shape}')
print(f'Test Sets: {X_test.shape} / {y_test.shape}')

drugTree = DecisionTreeClassifier(criterion="entropy")

drugTree.fit(X_train, y_train)

preditionTree = drugTree.predict(X_test)
print(preditionTree[0:5])
print(y_test[0:5])

print(f'Desicion Tree Accuracy: {metrics.accuracy_score(y_test, preditionTree)}')

dot_data = StringIO()
filename = "drugtree.png"  # günün sonunda oluşacak göresele isim veridik
featureNames = df.columns[0:5]  # ilk beş sütunun isimlerini aldık yani features yani özellikler
targetValue = df['Drug'].unique().tolist()  # unique() ilgili sütunda ki benzersiz değerleri alancak

out = tree.export_graphviz(drugTree,
                           feature_names=featureNames,
                           out_file=dot_data,
                           class_names=np.unique(y_train),  # kaç tane sınıfa ayıracak
                           filled=True,  # Sınıflandırma için çoğunluk sınıfını belirlemek için node'leri iligi sınıfın yoğunğuna göre boyamak için kullanılacak
                           special_characters=True)  # burada "@", "`", "!" vb. özel karakterleri kullanıma açtık
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  # burada verileri grafiğe yüklüyoruz
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img, interpolation='nearest')





