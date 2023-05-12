
# Bu çalışmada kullanıcıların filmlere verdikleri rating oranlarına göre bir kullanıcı profili oluşturarak filmlerin türlerine göre bir ağrırlık matrix'si oluşturup kullanıcıya film önerisinde bulunmaya çalışacağız.

import pandas as pd
import numpy as np
from math import sqrt

movies_df = pd.read_csv('data/movies.csv')
# print(movies_df.head())

ratings_df = pd.read_csv('data/ratings.csv')
# print(ratings_df.head())
#
# print(movies_df.shape)
# print(ratings_df.shape)

movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
# print(movies_df[["movieId", "title", "year"]])

# 14. satırada title sütununda bulunan (1999) şeklindeki string ifadeyi extract ederek movies_df içerisinde yeni oluşturulan ['year'] sütununa yazdırdık. burada ilgili yapıtı ifade etmek için regx kullanarak ifade ettik.
# 15. satırada (1999) halinde olan verilerimizin sağında ve solundaki parantezlerden kurtulduk.
# 16. satırda ['title'] sütununa bulunan (1999) ifadesinin her bir karakterini empty yaptık.
# 17. satırada yukarıda empty yapılan değerleri strip ederek yok ettik. burada daha önceden kullandığım lamda ve apply yapısından faydalandık. Yani her bir empty değer için strip işlemi tekrar edildi.

# Aşağıda ki kod bloğunun amacı filmlerin içerdikleri film türleri için 1 içermediği türler için 0 değeri basmaktır. Diğer adımalrda ana  movies_df veri setine ihtiyacım olduğundan ilk adımda onu kopyaladım.
moviesWithGenres_df = movies_df.copy()  # movies_df'i yedekledik

# 2. adımda: Kullanılan df (movies_df) her bir title satırı için, bütün türleri tek tek dolaşacak ve ilgili film bir türü içeriyorsa 1 değerini ekrana basacak. Böylelikle her bir filme ait olan tür saptanacak ve saptandığında 1 değeri atanacak. Şayet ilgili filmin türler listesinde bulunan türe karşılık gelen bir değeri yoksa NaN değeri basılacak. Örneğin Jumanji filmi için "Adventure, Children, Fantasy" bu türler için bir diğer türler için NaN değerini basacak.
for index, row in movies_df.iterrows():  # iterrows() fonsiyonu sütun ve satır şeklinde bize ikili değer gönerir. Yani satır ve o satır içerisinde ki değeri. Kendi dokümantasyonuna bakarak burada ki index ve colum ifadelerini düzeltelim.
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1  # burada kullanılan "at" yapısı daha önce öğrendiğimiz "loc" yapısı ile benzer çalışmaktadır. Bir satır yada sütun için çalışır ve burada değerler eşleşirse 1 değerini eşleşmezse NaN değerini basar. BUrada etiket tabanlı (label) bir arama işlemi yaptık varsa yada içeriyorsa 1 içermiyorsa NaN bastık.

# 3. adımda: ikinci adımda oluşan NaN değerleri için bu NaN değeri yerine 0 değeri ile doldurduk. Çünkü matematiksel olarak burada ki değerler ile işlem yapıp film ağırlık matrix'si çıkaracağız.
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
# print(moviesWithGenres_df)

# Filim izleme platformumuza yeni üye olan kişinin web arayüzünden girdiği datayı temsilen bir veri oluşturalım. Bu veri film isimleri ve rating oranlarını içerecek.

userInput = [
    {'title': 'Toy Story', 'rating': 4},
    {'title': 'Jumanji', 'rating': 5},
    {'title': 'Grumpier Old Men', 'rating': 2},
    {'title': 'Waiting to Exhale', 'rating': 1},
    {'title': 'Sudden Death', 'rating': 5},
]
inputMovies_df = pd.DataFrame(userInput)
print(inputMovies_df[["title", "rating"]])

# Aşağıdaki kod bloğunda kullancı girdisinde bulunan "title" alanı ile ana film veri setimizde bulunan title'ların eşlemesi sonucunda yakalanan data "merged_input" içerisine atıldı. Burada eşleşen data'dan kastımız movies_df içerisinde ki satırlar
# Burada ki ana amacımız kullanıcı girdisine bizim veri tabanımızda ki id bilgilerini vermektir.
merged_input = movies_df[movies_df["title"].isin(inputMovies_df["title"].tolist())]
inputMovies = pd.merge(merged_input, inputMovies_df)  # burada concat yada join'de kullanılır.

# inputMovies, movies_df içerisinde ki bilgilere sahip olduğundan merge işleminde ihtiyacımız olmayan sütunlarıda içerisinde barındırır durumda bu sebepten ötürü burada gereksiz sütunları atıyoruz.

inputMovies = inputMovies.drop('genres', axis=1).drop('year', axis=1)
print(inputMovies)

# 56. satırdaki çıktıyı incelediğimizde kullanıcının girdisine bizim veri tabanında ki id bilgilerini getirmiş olduk. yani artık elimizde kullancıı girdisinin yanında fimlerein id'leride var


userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
userGenre_df = userMovies.drop('movieId', axis=1).drop('title', axis=1).drop('genres', axis=1).drop('year', axis=1)
userGenre_df = userGenre_df.reset_index(drop=True)  # ilgili df üzerinde bunlanan index yapısını kaldırdık.
print(userGenre_df)

# 50. ve 61. satırlarda yapılan işlmelerin amacı filmilerin rating'lerine kullanarak türlerine göre onlara ağırlık vermekti. Bir öneri sisteminde bu tabloları ve ağırlık matrixsini çıkarmak en önemli adımdır.

userProfile = userGenre_df.transpose().dot(inputMovies_df["rating"])
print(userProfile)

# 29. satırda oluşturduğumuz for loop içerisinde filmlerin ait oldukları türlere 1 sahip olmadıkları türlere 0 değeri vermiştik. Bu df içerisinde aşağıda drop ettiğimiz sütunlarda yer almaktaydı. Recomendation system'a göre yaratılması gereken movie matrix mantığına göre burada sadece film türleri ve puanlarını bırakacak şekilde bir dizayn yaptık.
movie_matrix = moviesWithGenres_df.set_index(moviesWithGenres_df["movieId"])
movie_matrix = movie_matrix.drop(["movieId", "title", "genres", "year"], axis=1)
print(movie_matrix)

recomendation_matrix = ((userProfile * movie_matrix).sum(axis=1))/(userProfile.sum())
print(recomendation_matrix.head())

# recomendation_matrix sort edelim ki en çok öenrdiğinden en az önerdiğine göre sıralansın
recomendation_matrix = recomendation_matrix.sort_values(ascending=False)
print(recomendation_matrix.head(20))

result = movies_df.loc[movies_df['movieId'].isin(recomendation_matrix.head(20).keys())]
print(result[["movieId", "title"]])


