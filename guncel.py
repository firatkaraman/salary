import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import missingno as msno

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import model_selection, metrics
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder

from scipy.stats import norm
from scipy import stats


pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

df_salary = pd.read_csv('NBA Players Salaries 1920.csv')
df_stats1718 = pd.read_csv('NBA Players Stats 201718.csv')
df_stats1819 = pd.read_csv('NBA Players Stats 201819.csv')
df_list = [df_stats1718, df_stats1819, df_salary]

##################### 1. Data Cleaning #####################

# Veri setinin ilk 5 satırını getirip verilere göz atalım

df_salary.head()
df_stats1718.head()
df_stats1819.head()

# şu anlık ihtiyacımız olan kısım oyuncu isimleri bu yüzden sadece "Player" sütununu alıyoruz

for df in df_list:
    df[['Player', 'Del']] = df.Player.str.split("\\", expand = True)

df_stats1718 = df_stats1718.drop(['Del'], axis = 1)
df_stats1819 = df_stats1819.drop(['Del'], axis = 1)
df_salary = df_salary.drop(['Del'], axis = 1)

# "Salary" deki para pirimlerine gerek yok sadece sayı almamız yeterli. Bu yüzden sütunu yeniden adlandırıyoruz.
# Bu analizde yalnızca 19/20 sezonunu kullandığımız için 19/20 sezonunun maaşına ihtiyacımız olacak. Bu nedenle sadece 19/20 sezonu maaşı için belirtilen hususları değiştireceğiz.

# $ işaretini silip sütunu floata çeviriyoruz
df_salary['2019-20'] = df_salary['2019-20'].str[:-2].astype(float)

# salary sütununu yeniden adlandırıyoruz
df_salary = df_salary.rename(columns = {'2019-20': 'Salary 19/20'})

# salary'i sütununu 1000 ile bölüyoruz
df_salary['Salary 19/20'] = df_salary['Salary 19/20']/1000

# Sezon istatistikleri veri setinden aynı oyuncu için yinelenen satırları siliyoruz
#Analizimize gelince, oyuncunun bir sezonda nerede oynadığı önemli değil, sezon boyunca takım değiştiren oyuncuların tüm yinelenen satırlarını silebilir ve satırı toplam sezon istatistikleriyle tutabiliriz.
# Toplam istatistikler her zaman en üst sırada olduğundan drop_duplicates işlevini kullanabiliriz

df_stats1718 = df_stats1718.drop_duplicates(['Player'])
df_stats1819 = df_stats1819.drop_duplicates(['Player'])

## Ayrı sezonların verilerini birleştirme(Merge Datasets)
# Öncelikle sezon istatistiklerinin her veri setinin her sütununa karşılık gelen yılı atamamız gerekiyor.

columns_renamed = [s + ' 17/18' for s in list(df_stats1718.columns)]
df_stats1718.columns = list(df_stats1718.columns)[:3] + columns_renamed[3:]

columns_renamed = [s + ' 18/19' for s in list(df_stats1819.columns)]
df_stats1819.columns = list(df_stats1819.columns)[:3] + columns_renamed[3:]

# 17/18 df'den 'Pos' sütununu silebiliriz ona yalnızca bir kez ihtiyacımız var
df_stats1718 = df_stats1718.drop('Pos', axis = 1)


# Birleştirme işlemi

df_stats = df_stats1718.merge(df_stats1819, how = 'outer',left_on = ['Player'],right_on = ['Player'])
df = df_stats.merge(df_salary, how = 'outer', left_on = ['Player'],right_on = ['Player'])

df.head()

# veri setinin uzunluğu
len(df)

# Farklı sütunların veri türleri nelerdir?

df.dtypes

# Maaş tahminlemede işimize yaramayacak sütunları çıkartalım

df.columns
df = df.drop(['Rk_x', 'Rk_y', 'Rk', 'Tm', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25',
              'Signed Using', 'Guaranteed'], axis = 1)

# Eksik veriler
# Her sütunda kaç tane eksik değerimiz var?

df.isnull().sum()


# NaN satırları çıkarma
# 17/18 sezonu için istatistikler olmadan yeni bir dataframe oluşturalım
df1 = df.dropna(subset = ['Salary 19/20', 'PTS 18/19', 'eFG% 18/19'])
df1 = df1.reset_index()
columns = list(df1.columns)
for i in columns:
    if '17/18' in i:
        df1 = df1.drop([i], axis = 1)
df1 = df1.reset_index()

df2 = df.dropna(subset = ['Salary 19/20', 'PTS 18/19', 'eFG% 18/19', 'PTS 17/18', 'eFG% 17/18'])
df2 = df2.reset_index()

# oyuncularda hangi pozisyonlara sahibiz?

print(df1.Pos.unique())
print(df2.Pos.unique())

# bizim ihtyacımız olan ana pozisyonlar yani 'C' 'PF' 'SF' 'PG' 'SG' gibi aynı 'PF-SF' 'SF-SG' 'SG-PF' 'C-PF' 'SG-SF'
# gibi birden fazla pozisyonda oynayanları alırsak aynı oyuncuları birden fazla çekebiliriz o yüzden asıl mevkilerini alıcaz.

df1 = df1.replace({'SF-SG': 'SF', 'PF-SF': 'PF', 'SG-PF': 'SG', 'C-PF': 'C', 'SG-SF': 'SG'})
df2 = df2.replace({'SF-SG': 'SF', 'PF-SF': 'PF', 'SG-PF': 'SG', 'C-PF': 'C', 'SG-SF': 'SG'})


# Analizimizde, ana istatistiklerin mutlak büyümesini elde etmek için ikinci veri çerçevemiz
# (df2) için 17/18 sezonunun istatistiklerini kullanacağız.

list_growth = ['eFG%','TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
for i in list_growth:
    df2[i + ' +-'] = df2[i + ' 18/19'] - df2[i + ' 17/18']

columns = list(df2.columns)
for i in columns:
    if '17/18' in i:
        df2 = df2.drop([i], axis = 1)

print(df1.shape)
print(df2.shape)


# iki sezona ait istatistikleri ve 19/20 maaşlarını içeren 358 satırımız (Oyuncular) var.
# 17/18 sezonunun istatistiklerini dışarıda bırakırsak, 73 ek satırımız olur.

##################### 2. Veri kümemizi keşfetme #####################

# En yüksek maaşı hangi oyuncular alıyor, bunu inceleyelim


df_sal = df1[['Player', 'Salary 19/20']]
df_sal.sort_values(by = 'Salary 19/20', ascending = False, inplace = True)

sns.catplot(x = 'Player', y = 'Salary 19/20', kind = 'bar', data = df_sal.head()).set(xlabel = None)
plt.title('Players with highest salary (in 1000)')
plt.ylim([35000, 40000])
plt.xticks(rotation = 90)
plt.show(block=True)

# Temel istatistiksel değerlerin özeti
df1['Salary 19/20'].describe()

# Dağılım nasıl görünüyor histogramı ile bakalım
sns.distplot(df1['Salary 19/20'])
plt.show(block=True)

# Bu dağılıma baktıktan sonra neler söyleyebiliriz?
# 1- Büyük standart sapma, maaşların yayıldığı anlamına gelir
# 2- Maaşlar normal dağılıma sahip değil
# 3- Sağa çarpık dağılım sahiptir
# Çoğu tahmin modeli için verilerin normal dağılıma sahip olması önemlidir.

# En önemli oyuncu istatistiklerinin analizi
# Şimdi, NBA oyuncularının maaşlarının büyük bir kısmını muhtemelen açıklayan değişkenlere bakalım.
# Odak noktamız, 18/19 sezonundaki maç başına sayı, asist, top çalma ve ribaund olacak.

# Her kategoride lider olanlar kimlerdir?

df_pts = df1[['Player', 'PTS 18/19']]
df_pts.sort_values(by = 'PTS 18/19', ascending = False, inplace = True)
df_ast = df1[['Player', 'AST 18/19']]
df_ast.sort_values(by = 'AST 18/19', ascending = False, inplace = True)
df_stl = df1[['Player', 'STL 18/19']]
df_stl.sort_values(by = 'STL 18/19', ascending = False, inplace = True)
df_trb = df1[['Player', 'TRB 18/19']]
df_trb.sort_values(by = 'TRB 18/19', ascending = False, inplace = True)

f, axes = plt.subplots(2, 2, figsize=(20, 15))
sns.despine(left=True)

sns.barplot(x = 'PTS 18/19', y = 'Player', data = df_pts.head(), color = "b", ax = axes[0, 0]).set(ylabel = None)
sns.barplot(x = 'AST 18/19', y = 'Player', data = df_ast.head(), color = "r", ax = axes[0, 1]).set(ylabel = None)
sns.barplot(x = 'STL 18/19', y = 'Player', data = df_stl.head(), color = "g", ax = axes[1, 0]).set(ylabel = None)
sns.barplot(x = 'TRB 18/19', y = 'Player', data = df_trb.head(), color = "m", ax = axes[1, 1]).set(ylabel = None)
plt.show(block=True)

# Dağılımlar nasıl görünüyor?

f, axes = plt.subplots(2, 2, figsize=(20, 15))
sns.despine(left=True)

# Histogramlar
sns.distplot(df1['PTS 18/19'], color = "b", ax = axes[0, 0])
sns.distplot(df1['AST 18/19'], color = "r", ax = axes[0, 1])
sns.distplot(df1['STL 18/19'], color = "g", ax = axes[1, 0])
sns.distplot(df1['TRB 18/19'], color = "m", ax = axes[1, 1])
plt.show(block=True)

# Basketbolun en önemli istatistikleri de genellikle normal dağılıma sahip değil, sağa çarpık bir dağılıma sahiptir.


# Olası özelliklerle ilişki

# Basketbolun en önemli istatistikleri ile maaş arasındaki ilişkiye nasıl bir görüntü ortaya çıkıyor, görelim.
# Şu an için sadece 2018/19 sezonu istatistiklerine odaklanacağız. Yine de maç başına sayı, asist, top çalma ve
# ribaund ile olan ilişkiyi inceleyeceğiz.

f, axes = plt.subplots(2, 2, figsize=(20, 15))

# Regressionplot
sns.regplot(x = df1['PTS 18/19'], y = df1['Salary 19/20'], color="b", ax=axes[0, 0])
sns.regplot(x = df1['AST 18/19'], y = df1['Salary 19/20'], color="r", ax=axes[0, 1])
sns.regplot(x = df1['STL 18/19'], y = df1['Salary 19/20'], color="g", ax=axes[1, 0])
sns.regplot(x = df1['TRB 18/19'], y = df1['Salary 19/20'], color="m", ax=axes[1, 1])
plt.show(block=True)

# Beklendiği gibi, tüm bu özellikler ile maaş arasında pozitif bir lineer ilişki bulunmaktadır

sns.regplot(x = df1['eFG% 18/19'], y = df1['Salary 19/20'])
plt.show(block=True)


# Normal saha isabet yüzdesi yerine etkili saha isabet yüzdesini seçtik, çünkü üçlük isabetleri
# üç sayıya denk gelirken, normal saha isabetleri sadece iki sayıya denk gelmektedir. Burada
# pozitif bir lineer ilişki neredeyse gözlenemez. Yüksek bir eFG%'ye sahip ancak düşük maaşa
# sahip oyuncular var, ancak düşük bir eFG%'ye sahip ve yüksek maaşa sahip oyuncular bulunmamaktadır.

# Maç başına oynanan dakikalar ile olan ilişki
sns.regplot(x = df1['MP 18/19'], y = df1['Salary 19/20'])
plt.show(block=True)

# Maç başına oynanan dakikalar ile de pozitif bir ilişki bulunmaktadır. Muhtemelen bu, üssel bir ilişki olabilir.
# Ancak bu ilişkiyi analiz ederken dikkatli olmamız gerekiyor. İyi istatistiklerin bir sonraki sezon daha yüksek
# bir maaşa ve aynı zamanda daha fazla maç başına dakikaya neden olabileceğini düşünebiliriz.
# Bu nedenle, bu pozitif ilişkinin muhtemel nedeni büyük olasılıkla istatistiklerdir.


# Yaş ile olan ilişki
sns.regplot(x = df1['Age 18/19'], y = df1['Salary 19/20'])
plt.show(block=True)

# Yaş ile maaş arasında lineer bir ilişki olmadığını söyleyebiliriz.


# Pozisyon ile olan ilişki
sns.boxplot(x = 'Pos', y = 'Salary 19/20', data = df1, order = ['PG', 'SG', 'SF', 'PF', 'C'])
plt.show(block=True)
# Herhangi bir düzen veya korelasyon belirlemenin mümkün olmadığını söyleyebiliriz.


# Correlation matrix
# Şu ana kadar yapılan analiz, sezgilerimize ve maaşı açıklamak için önemli olduğunu düşündüklerimize dayanıyordu.
# Şimdi daha objektif bir analiz yapalım ve değişkenlerimizin tüm ilişkilerine dair mükemmel bir genel bakış elde edelim.
# Bu nedenle bir ısı haritası kullanacağız.

sns.set(style = "white")
cor_matrix = df1.loc[:, 'Age 18/19': 'Salary 19/20'].corr()
mask = np.triu(np.ones_like(cor_matrix, dtype = np.bool))
plt.figure(figsize = (15, 12))
cmap = sns.diverging_palette(220, 10, as_cmap = True)
sns.heatmap(cor_matrix, mask = mask, cmap = cmap, center = 0,
            square = True, linewidths = .5, cbar_kws = {"shrink": .5})
plt.show(block=True)

# Isı haritasınıda çıkardıktan sonra lde edilen görüşler nelerdir?

# Çoklu doğrusallık:
# Bu üçgen ısı haritasında ilk dikkat çeken şey, birçok kırmızı renkli kare olduğudur.
# Bu, tahmin modelimiz için önemli bir bilgidir. Bu kadar yüksek bir korelasyona sahip değişkenler,
# tahmin modelimize neredeyse aynı bilgiyi sağlar ve sadece tahminimizin varyansını artırır.
# Modelimizi oluştururken bunu akılda tutmalıyız.

# Maaşla İlişkiler 2019/20:
# Zaten analiz ettiğimiz gibi, maç başına maaşla dört ana istatistik arasında lineer bir ilişki bulunmaktadır.
# Ancak dikkate alınması gereken diğer değişkenler de bulunmaktadır. Örneğin, maç başına fauller (PF 18/19),
# maaşı açıklamada önemli bir rol oynar.
# Maç başına oynanan dakika (MP) ve başlanan maç sayısının (GS) maaşla olan korelasyonunu analiz ederken dikkatli
# olmalıyız. Oyuncuların daha fazla oynaması otomatik olarak daha fazla kazandıkları anlamına gelmez.
# Çünkü hücum ve savunma istatistikleri, koçun oyuncuyu oyun içinde ne kadar süreyle bırakacağını belirler ve
# aynı zamanda gelecek sezonun maaşını belirler, bu ilişki nedensel değildir. Modelimizde kullanabileceğimiz bir
# istatistik ise toplam oynanan maçlardır (G). Bu istatistik, bir oyuncunun savunmasızlık derecesi ile maaşı arasındaki
# ilişkiyi ölçebilir, örneğin.
# Ayrıca, daha fazla top kaybının (TOV) yüksek korelasyon nedeniyle daha yüksek bir maaşa yol açtığını söyleyemeyiz.
# Bu elbette ki doğru değil. İyi hücum ve savunma istatistiklerine sahip bir oyuncu daha fazla süre alır ve
# aynı zamanda daha yüksek bir maaş alır. Bu nedenle, iyi oyuncuların yüksek maaşlı olarak daha fazla top kaybına
# sahip olduğunu söylemek daha mantıklıdır çünkü daha fazla süre alırlar.



# Mutlak değişimlerle maaş arasındaki ilişkiye bakalım

# Bir oyuncu, 18/19 sezonunda 17/18'e göre daha iyi performans gösterdiyse ve bu nedenle
# daha da iyi olması bekleniyorsa, maaşı daha yüksek olabilir. Buna bakalım

cor_matrix = df2.loc[:, ['eFG% +-','TRB +-', 'AST +-', 'STL +-', 'BLK +-', 'TOV +-', 'PF +-', 'PTS +-',
                        'Salary 19/20']].corr()

mask = np.triu(np.ones_like(cor_matrix, dtype = np.bool))

plt.figure(figsize = (10, 8))

cmap = sns.diverging_palette(220, 10, as_cmap = True)

sns.heatmap(cor_matrix, mask = mask, cmap = cmap, center = 0,
            square = True, linewidths = .5, cbar_kws = {"shrink": .5})
plt.show(block=True)

# Mutlak değişimlerle maaş arasında zayıf pozitif ya da neredeyse hiç bir lineer ilişkiler görünmüyor.
# Bu nedenle, mutlak değişimleri ve dolayısıyla 17/18 sezonunun istatistiklerini tahmin modelimizden çıkaracağız.


##################### 3. Veriyi Hazırlama(Data Preparation) #####################

# Hedef değişkeni ve özellikleri tanımlayalım

y = df1.loc[:, 'Salary 19/20']

x = df1.loc[:, ['Pos', 'Age 18/19', 'G 18/19', 'GS 18/19', 'MP 18/19', 'FG 18/19', 'FGA 18/19',
                'FG% 18/19', '3P 18/19', '3PA 18/19', '2P 18/19', '2PA 18/19', '2P% 18/19',
                'eFG% 18/19', 'FT 18/19', 'FTA 18/19', 'ORB 18/19', 'DRB 18/19', 'TRB 18/19',
                'AST 18/19', 'STL 18/19', 'BLK 18/19', 'TOV 18/19', 'PF 18/19', 'PTS 18/19']]

print(x.shape)
print(y.shape)



#Özellik pozisyonu için Tekli Kodlama (One-Hot encoding)

# 'Pos_x' kategorik değişkeniyle başa çıkmak için Tekli Kodlayıcı'yı kullanacağız.
# Normal Etiket Kodlayıcıyı( normal Label encoder ) kullanamayız çünkü sıralama, algoritmamıza belirli pozisyonların
# diğerlerinden daha iyi olduğunu söylerdi.


ohe = OneHotEncoder(categories = [['PG', 'SG', 'SF', 'PF', 'C']])

x_ohe = pd.DataFrame(ohe.fit_transform(x['Pos'].to_frame()).toarray())

x_ohe.columns = ohe.get_feature_names_out(['Pos'])

x_ohe.index = x.index

x = pd.concat([x, x_ohe], axis=1).drop(['Pos'], axis=1)

x.head()


# Machine learning modellemesinde kullanmak için verileri 'train' ve 'test' olarak ayırırız

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Analizimizde gördüğümüz gibi, bağımlı değişken yaklaşık olarak normal bir dağılımı takip etmiyor,
# ancak çoğu model için bu gerekli. Bu nedenle, bir sonraki adımlar için y'yi normalize etmemiz gerekiyor.


y_train = pd.DataFrame(np.cbrt([y_train])).T
y_test = pd.DataFrame(np.cbrt([y_test])).T
y = pd.DataFrame(np.cbrt([y])).T

f, axes = plt.subplots(1, 2, figsize = (10, 5), sharex = True)
sns.distplot(y_train, color = "skyblue", fit = norm, ax = axes[0], axlabel = "y_train")
sns.distplot(y_test, color = "olive",fit = norm, ax = axes[1], axlabel = "y_test")
plt.show(block=True)


# Algoritmalar için özelliklerimizin ölçeklendirilmiş olması önemlidir

scaler = RobustScaler()
x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), index = x_train.index, columns = x_train.columns)
x_test_scaled = pd.DataFrame(scaler.transform(x_test), index = x_test.index, columns = x_test.columns)

x_train_scaled.head()


##################### 4. Makine öğrenmesi modelinin kurulması  #####################

# Temel makine öğrenimi algoritmaları
# Benzer kodları tekrarlamayı önlemek için bir fonksiyon kuracağız. Bu modelleri kök ortalama kare hatası
# (root-mean-squared-error) ve R-kare ile değerlendireceğiz.

def alg_fit(alg, x_train, y_train, x_test, name, y_true, df, mse, r2):
    # Model seçme
    mod = alg.fit(x_train, y_train)

    # tahmin
    y_pred = mod.predict(x_test)

    # Kesinlik
    acc1 = round(mse(y_test, y_pred), 4)
    acc2 = round(r2(y_test, y_pred), 4)

    # Kesinlik tablosu
    x_test['y_pred'] = mod.predict(x_test)
    df_acc = pd.merge(df, x_test, how='right')
    x_test.drop(['y_pred'], axis=1, inplace=True)
    df_acc = df_acc[[name, y_true, 'y_pred']]
    df_acc.sort_values(by=y_true, ascending=False, inplace=True)
    df_acc['y_pred'] = df_acc['y_pred'] ** 3

    return y_pred, acc1, acc2, df_acc


# Doğrusal Regresyon
# Basit bir doğrusal regresyon modeli kurup test edeceğiz.

y_pred_lin, mse_lin, r2_lin, df_acc_lin = alg_fit(LinearRegression(), x_train, y_train, x_test, 'Player', 'Salary 19/20',
                                                  df1, metrics.mean_squared_error, metrics.r2_score)

print("Root Mean Squared Error: %s" % round(np.sqrt(mse_lin), 4))
print("R-squared: %s" % r2_lin)
df_acc_lin.head(10)


# Ridge Regresyon
# Doğrusal regresyon modelimizde çoklu doğrusallık sorunu yaşadığımız için uygun bir çözüm bulmamız gerekiyor.
# Yüksek çoklu doğrusallık durumunda tahminimiz muhtemelen kesin değildir ve büyük standart hatalara sahiptir.
# Ridge regresyon, bu sorunu azaltarak tahminimizde iyileşmiş verimlilik sağlar, ancak bir miktar önyargı karşılığında.

y_pred_rid, mse_rid, r2_rid, df_acc_rid = alg_fit(Ridge(alpha = 1), x_train, y_train, x_test, 'Player', 'Salary 19/20',
                                                  df1, metrics.mean_squared_error, metrics.r2_score)

print("Root Mean Squared Error: %s" % round(np.sqrt(mse_rid), 4))
print("R-squared: %s" % r2_rid)
df_acc_rid.head(10)

# Lasso Regresyon
# Lasso regresyon, kavramsal olarak ridge regresyonuna oldukça benzerdir. Ek olarak, sıfır olmayan
# katsayılar için bir ceza ekler. Ridge regresyonun aksine, katsayıların karesinin toplamını değil,
# katsayıların mutlak değerini sınırlar.

y_pred_las, mse_las, r2_las, df_acc_las = alg_fit(Lasso(alpha = 0.001), x_train, y_train, x_test, 'Player', 'Salary 19/20',
                                                  df1, metrics.mean_squared_error, metrics.r2_score)

print("Root Mean Squared Error: %s" % round(np.sqrt(mse_las), 4))
print("R-squared: %s" % r2_las)
df_acc_las.head(10)


# Çapraz Doğrulama (Cross Validation)
# Bu bize daha kesin bir doğruluk ölçüsü sağlayacak.

def alg_fit_cv(alg, x, y, mse, r2):
    # Çapraz doğrulama
    cv = KFold(shuffle=True, random_state=0, n_splits=5)

    # Kesinlik
    scores1 = cross_val_score(alg, x, y, cv=cv, scoring=mse)
    scores2 = cross_val_score(alg, x, y, cv=cv, scoring=r2)
    acc1_cv = round(scores1.mean(), 4)
    acc2_cv = round(scores2.mean(), 4)

    return acc1_cv, acc2_cv
# Doğrusal regresyon
mse_cv_lin, r2_cv_lin = alg_fit_cv(LinearRegression(), x, y, 'neg_mean_squared_error', 'r2')

print("Root Mean Squared Error: %s" % round(np.sqrt(mse_cv_lin*-1), 4))
print("R-squared: %s" % r2_cv_lin)

# Ridge regresyon

mse_cv_rid, r2_cv_rid = alg_fit_cv(Ridge(alpha = 23), x, y, 'neg_mean_squared_error', 'r2')

print("Root Mean Squared Error: %s" % round(np.sqrt(mse_cv_rid*-1), 4))
print("R-squared: %s" % r2_cv_rid)

# Lasso regresyon
mse_cv_las, r2_cv_las = alg_fit_cv(Ridge(alpha = 23), x, y, 'neg_mean_squared_error', 'r2')

print("Root Mean Squared Error: %s" % round(np.sqrt(mse_cv_las*-1), 4))
print("R-squared: %s" % r2_cv_las)


# Gelişmiş Modeller
# Daha gelişmiş yaklaşımları kullanalım.

# LightGBM
lgbm = LGBMRegressor(objective = 'regression',
                     num_leaves = 20,
                     learning_rate = 0.03,
                     n_estimators = 200,
                     max_bin = 50,
                     bagging_fraction = 0.85,
                     bagging_freq = 4,
                     bagging_seed = 6,
                     feature_fraction = 0.2,
                     feature_fraction_seed = 7,
                     verbose = -1)

mse_cv_lgbm, r2_cv_lgbm = alg_fit_cv(lgbm, x, y, 'neg_mean_squared_error', 'r2')

print("Root Mean Squared Error: %s" % round(np.sqrt(mse_cv_lgbm*-1), 4))
print("R-squared: %s" % r2_cv_lgbm)

# XGB
xgb = XGBRegressor(n_estimators = 300,
                   max_depth = 2,
                   min_child_weight = 0,
                   gamma = 8,
                   subsample = 0.6,
                   colsample_bytree = 0.9,
                   objective = 'reg:squarederror',
                   nthread = -1,
                   scale_pos_weight = 1,
                   seed = 27,
                   learning_rate = 0.02,
                   reg_alpha = 0.006)

mse_cv_xgb, r2_cv_xgb = alg_fit_cv(xgb, x, y, 'neg_mean_squared_error', 'r2')

print("Root Mean Squared Error: %s" % round(np.sqrt(mse_cv_xgb*-1), 4))
print("R-squared: %s" % r2_cv_xgb)


# Veri ve Özelliklerimizi Optimize Etme
# Aykırı Değerler (Outliers)

df_new = pd.concat([y, x], axis=1)

z = np.abs(stats.zscore(df_new))

df_new = df_new[(z < 4).all(axis = 1)].reset_index()

y_new = df_new.loc[:, 0]
x_new = df_new.loc[:, 'PTS 18/19':]
print(x_new.shape)
print(y_new.shape)


# Özellik Önem Derecelendirmesi (Feature importance)

# Şimdiye kadar en iyi doğruluk puanına sahip olduğu için XGB Regresör'ü kullanarak en önemli özellikleri belirleyeceğiz.

# Model
mod = xgb.fit(x, y)

# Feature importance
df_feature_importance = pd.DataFrame(xgb.feature_importances_, index = x.columns,
                                     columns = ['feature importance']).sort_values('feature importance',
                                                                                   ascending = False)
df_feature_importance


# Yüksek öneme sahip olan özellikleri kullanacağız. Bazı özellikler gereksiz veya başka bir özellikle yakın
# olduğu için kullanılamaz. Örneğin, maç başına saha isabetleri (FG 18/19), zaten maç başına sayıları (PTS 18/19)
# olduğu için gerekli değildir.

# Düşük öneme sahip veya gereksiz olan özellikleri çıkartalım
x_new = x.loc[:, ['PTS 18/19', 'Pos_PG', 'Pos_SG', 'Pos_SF', 'Pos_PF', 'Pos_C', 'Age 18/19', 'STL 18/19',
                  'G 18/19', 'TRB 18/19', 'AST 18/19', 'PF 18/19', 'MP 18/19']]

################################ Final Model ################################

xgb_new = XGBRegressor(n_estimators = 270,
                       max_depth = 2,
                       min_child_weight = 0,
                       gamma = 18,
                       subsample = 0.7,
                       colsample_bytree = 0.9,
                       objective = 'reg:squarederror',
                       nthread = -1,
                       scale_pos_weight = 1,
                       seed = 27,
                       learning_rate = 0.023,
                       reg_alpha = 0.02)

mse_cv_xgb, r2_cv_xgb = alg_fit_cv(xgb_new, x_new, y, 'neg_mean_squared_error', 'r2')

print("Root Mean Squared Error: %s" % round(np.sqrt(mse_cv_xgb*-1), 4))
print("R-squared: %s" % r2_cv_xgb)

#test verilerinde nasıl performans gösterdiğine bakalım


x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.2, random_state = 0)

y_pred_xgb, mse_xgb, r2_xgb, df_acc_xgb = alg_fit(xgb_new, x_train, y_train, x_test, 'Player', 'Salary 19/20',
                                                  df1, metrics.mean_squared_error, metrics.r2_score)

print("Root Mean Squared Error: %s" % round(np.sqrt(mse_xgb), 4))
print("R-squared: %s" % r2_xgb)
df_acc_xgb.head(10)

# Çıktıda gördüğümüz gibi asıl verilerde 19/20 yılında aldığı maaş ve bizim geliştrdiğimiz modelde y_pred ile gelecek
# sezonda ne kadar maaş alabileceğini gösteriyoruz.











