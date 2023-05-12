
# BU çalışmada kullanıcının puanladığı filmler ile veri tabanında hali hazırda aynı filimleri puanlamış kullanıcılar arasında ki benzerliği saptayarak hali hazırda bulunan kullanıcıların beğendikleri filmleri yeni gelen kullanıcıya önereceğiz.

import pandas as pd
import numpy as np
from math import sqrt

movies_df = pd.read_csv('data/movies.csv')
ratings_df = pd.read_csv('data/ratings.csv')

movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

movies_df.drop('genres', axis=1, inplace=True)
ratings_df.drop('timestamp', axis=1, inplace=True)

userInput = [
    {'title': 'Toy Story', 'rating': 4},
    {'title': 'Jumanji', 'rating': 5},
    {'title': 'Grumpier Old Men', 'rating': 2},
    {'title': 'Waiting to Exhale', 'rating': 1},
    {'title': 'Sudden Death', 'rating': 5},
]
inputMovies_df = pd.DataFrame(userInput)

merged_input = movies_df[movies_df["title"].isin(inputMovies_df["title"].tolist())]
inputMovies = pd.merge(merged_input, inputMovies_df)
inputMovies = inputMovies.drop('year', axis=1)
print(inputMovies)
# Yeni gelen kullanıcının rating verdiği fimleri izleyen kullanıcıları saptayalım.
userSubSet = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]

# Kullanıcıları id'leine göre groupladık.
userSubSetGroup = userSubSet.groupby(['userId'])
#print(userSubSetGroup.get_group(11))

# Aşağıda sort yapmamızın yeni kullanıcı ile hali hazırda veri tabanında olan kullanıcıların daha verimli bir şekilde eşleştirmektrir. Yani ortak filmleri paylaştığım kullanıcıları yakalamak istediğimiz için bu işlemi yaptık.
userSubSetGroup = sorted(userSubSetGroup, key=lambda x: len(x[1]), reverse=True)
print(userSubSetGroup[0:1])


# Var olan kullanıcı ile yeni kayıt olan kullanıcının benzerliğini saptayalım
# Bu benzerliği belirlemek için pearson korelasyonunu kullanacağız. Pearson korelasyonu ile elde edeceğimiz katsayı ile yeni üye olan kullanıcı ile hali hazırdaki kullanıcıların ne kadar benzer olduğunu belirleyeceğiz. Pearson iki değişken arasındaki doğrusal ilişkinin gücünü ölçmek için kullanılmaktadır.

# Rating veri setinde çok fazla data olduğunda groupladığımız ve sort ettiğimiz veri setindeki ilk 100 kişi için yeni gelen kullanıcının korelasyonuna bakacağız.
userSubSetGroup = userSubSetGroup[0:100]

pearsonCorrelationDict = {}

# alt kullanıcılar kümemizi (userSubSetGroup) her kullanıcı grubu için yineyeceğiz
for userId, group in userSubSetGroup:
    # Girdiyi ve mevcut kullanıcı grubunu movieId bilgisine göre sıraladım, böylece değerler daha sonra karışmayacak
    group.sort_values(by='movieId', inplace=True)
    inputMovies.sort_values(by='movieId', inplace=True)
    # Korelasyon formülü için N sayısnı alalım
    nRating = len(group)
    # Her ikisinde de ortak olan filmlerin inceleme puanlarını alacağız
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    # Yukarıdaki işlem sonucunda bize movieId, title vb. bilgilerde gelecek ama bizim ihtiyacımız sadece rating bu yüzden rating sütunun temp_df içerisinden çekip alalım.
    # Ayrıca birazdan kullanacağımız formüller içerisinde rating bilgilerini daha rahat kullanmak için bu pandas series tipindeki bilgiyi listeye dönüştürelim
    tempRatingList = temp_df['rating'].tolist()
    tempGroupList = group['rating'].tolist()
    # Pearson Korelasyonun formülü için kullanılacak bilgiler hazılandı.
    # Şimdi formül tarafından x ve y olarak adlanrılına iki kullanıcı arasındaki korelasyon yani benzerlik katsayılarını bulalım
    Sxx = sum([i ** 2 for i in tempRatingList]) - pow(sum(tempRatingList), 2) / float(nRating)
    Syy = sum([i ** 2 for i in tempGroupList]) - pow(sum(tempGroupList), 2) / float(nRating)
    Sxy = sum(i * j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList) * sum(tempGroupList) / float(nRating)

    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[userId] = Sxy / sqrt(Sxx * Syy)
    else:
        pearsonCorrelationDict[userId] = 0

print(pearsonCorrelationDict.items(), end=",")
print("\n")

pearsonDf = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDf.columns = ['similartyIndex']
pearsonDf['userId'] = pearsonDf.index
pearsonDf.index = range(len(pearsonDf))
print(pearsonDf)

topUsers = pearsonDf.sort_values(by='similartyIndex', ascending=False)[0:50]
print(topUsers)

# Yukarıda seçilen ve sort edilen top 50 kullanıcının filmlere verdiği rating puanlarını bulalım
# Bu işlemi yapmak için ağırlık olarak pearson korelasyonu kullanarak filmlerin reytinglerinin ağrılıklı ortalamsı alarak yapacağız. Ancak bunu yapmak için öncelikle pearsonDf'de kullanıcıların izledikleri fimleri decerelendirme veri çerçevesinden almamız gerekmektedir.
topUserRating = topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
print(topUserRating)

# Şimdi tek yapmamız gereken, filmleri derecelenridilmesini ağrılıklarıyla yani benzerlik index'si ile çarpmak, ardından yeni derecelnerimeleri ile toplamak ve ağırlıkların toplamına bölmek. Bunu basitçe df'in hızlı filtreleme mekanizmasıyla alıp çarpıp ardından df'in movieId ile gruplayalım ve ardından "sum_similarityIndex" ve "sum_weightedRating" olarak iki ayrı sütun olarak oluşturalım.
topUserRating['weightedRating'] = topUserRating['similartyIndex'] * topUserRating['rating']
#print(topUserRating)

tempTopUserRating = topUserRating.groupby('movieId').sum()[["similartyIndex", "weightedRating"]]
tempTopUserRating.columns = ['sum_similarityIndex', 'sum_weightedRating']
print(tempTopUserRating)

recomendation_df = pd.DataFrame()

recomendation_df['weighted average recomendation score'] = tempTopUserRating['sum_weightedRating'] / tempTopUserRating['sum_similarityIndex']
recomendation_df['movieId'] = tempTopUserRating.index
print(recomendation_df)


recomendation_df = recomendation_df.sort_values(by='weighted average recomendation score', ascending=False)
print(recomendation_df.head(20))