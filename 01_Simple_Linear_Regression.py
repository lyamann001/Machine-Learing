import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

df = pd.read_csv("data/FuelConsumption.csv")
print(df.head())
print(df.shape)

#  Bu örnekte simple linear regression yapacağımız için bizim ihtiyacımız olan bir tane bağımsız değişken (EngineSize) ve bağımlı değişken (Co2Emmision) olacak. Bu sebepten ötürü veri setindeki diğer sütunlara ihtiyacımız yok.
# Bu bağlamda "Kümülatif Veri Seti" oluşturalım. Yani projemizde ihtiyacımız olan verileri içeren bir veri seti inşa edelim. Kümülatif adındanda anlaşılacağı gibi daraltılmış kümelenmiş veri seti anlamına gelmekedir.
cdf = df[["ENGINESIZE", "CO2EMISSIONS"]]  # ihtiyacımız olmayan sütunaları drop etmekle uğraşmaktansa ihtiyacımız olanları select edip yeni bir df'e yazdık.
print(cdf.head(10))

# Makine Öğenmesi algoritmalarının hepsinde kullanılan bir adıma sıra geldi.
# Train Test Splited Data, veri setimizi eğitim ve test olmak üzere iki ayrı veri setine böleceğiz. Burada amaç bu projemizde kullanacağımız regrasyon modelimizi eğitmek için train veri setini, modelimiz sonucunda elde ettiğimiz değerleri test etmek için ise test veri setini kullanmaktır.
# Veri setini split ederken nelere dikkat etmeliyiz:
# 1. Bu işlemde veri setimizin yüzde 80 yada en az yüzde 70'şini train olarak ayıracağız. Bunun nedeni, modelimiz ne kadar farklı veri görürse o kadar başarılı sonuçlar üretir. Yani insan için konuşmak gerekirse çok okuyan ve çok gezen iyi bilir mantığıyla burada modelimizi farklı farklı datalar ile train edeceğiz başarılı sonuçlar alalım. Yukarıda verilen oranlar best practice'dir. BU işin gurulararı tarafından öğrendiğimiz değerlerdir.
# 2. Bu işlemde diğer dikkat etmem gereken husus ise hem train hemde test veri setlerimin split işleminden sonra homojen bir yapıda kalmasıdır. Örneğin veri setim içerisinde a, b, c, d veri türleri olsun. Şayet train setimi a veri türü ile doldurursam ve ileride modelimi b veri türü için çalıştırısam başarısız olurum. Bu yüzden split işlemini yaparken tüm veri türlerinin mümkün mertebe eşit olarak dağılmasını yani veri setlerimin homojen olarak oluşmasını temin etmek zorundayım.

# Split the data
msk = np.random.randn(len(df)) <= 0.8
print(msk)  # bu sonucu incelediğimizde true false olarak bize bir array döndüğünü gördük. randn fonksiyonu saysinde true false olarak doldurduk. bu listenin yüzde sekseni true yüzde 20si false'dir
train = cdf[msk]  # burada yüzde 80'ni traine
test = cdf[~msk]  # burada ise "~" sembolü sayesinde yani değil anlamına gelir, yani true olmayanları test attık.

# print(train)
# print(test)

# Not: Bu tarz split işlemleri için sklearn içerisinde hazır methodlar bulunmaktadır.


regr = linear_model.LinearRegression()  # instance aldık
train_x = np.asanyarray(train[["ENGINESIZE"]])  # asanyarray() fonksiyonu içerisine parametre olarak verilen girdiği diziye dönüştürerek bize teslim eder. Burada bu işlemi yapma nedenimiz bizardan linear doğrunun katsayılarını hesaplamak için kullacağımız fonksiyonda matematiksel işlem yapılmasıdır.
train_y = np.asanyarray(train[["CO2EMISSIONS"]])  # asanyarray() fonsiyonu içerisine pandas.Series tipinde değer gönderdiğimzide yani train["CO2EMISSIONS"] bu tipte gönderdiğimizde hata verdi. bunu yerine df tipinde göndermek için train[["CO2EMISSIONS"]] şeklinde input verdik.
regr.fit(train_x, train_y)
print('Coefficient: %.2f' % regr.coef_[0][0])
print('Intercept:  %.2f' % regr.intercept_[0])

plt.scatter(train["ENGINESIZE"], train["CO2EMISSIONS"], color="blue")
plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], color="r")
plt.title("Scatter Grapgh")
plt.xlabel("Engine Size")
plt.ylabel("Emmison")
plt.show()

x = float(input("Please type into engine size: "))
y = regr.intercept_ + regr.coef_ * x

print(f"Carbon emmision of the vehicle with engine size of {x}: {math.floor(y[0][0])}")

test_x = np.asanyarray(test[["ENGINESIZE"]])
test_y = np.asanyarray(test[["CO2EMISSIONS"]])
test_linear_ = regr.predict(test_x)  # test veri setin içerisinde ki her bir ENGINESIZE değerine göre carbon emisyonunu tahmin ettik. bu tahmin edilen sonucu algoritmanın başarısını ölçmek için kullanılacak metrikler içerisinde kullanacağız.
print(test_linear_)

print("r-2 Score: %.2f" % r2_score(test_linear_, test_y))
# Regrasyon'da R2 belirleme katsayısı regrasyon tahminlerinin gerçek veri noktalarına ne kadar iyi yaklaştığının istatiksel olarak bir ölçümüdür. R2 scoru 1're ne kadar yaklaşırsa bulunan fit çizgisi veri setine o kadar uyduğunu gösterir.

print("Mean Sum of Squares Error: %.2f" % np.mean(test_linear_ - test_y) ** 2)
