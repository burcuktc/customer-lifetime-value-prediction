#Customer Lifetime Value Prediction
#CLTV = (Customer Value/ Churn Rate) x Profit Margin
#Customer Value= Purschase Frequency * Average Order Value
#Tüm kitlenin satın alma davranışları ve tüm kitlenin işlem başına ortalama bırakacakları kazancı olasılıksal olarak modelleme ve bu olasılıksal modelin üzerine bir kişinin özelliklerini girerek genel kitle davranışlarından beslenerek tahmin işleminde bulunma
# CLTV = Expected Number of Transaction * Expected Average Profit
#Tüm kitlenin satın alma davranışlarını bir olasılık dağılımı ile modelleyeceğiz. Daha sonra bu olasılık dağılımı ile modellediğimiz davranış biçimlerini conditional (koşullu) yani kişi özelinde biçimlendirecek şekilde kullanarak her bir kişi için beklenen satın alma beklenen işlem sayılarını tahmin edeceğiz.
#Aynı şekilde tüm kitlenin average profit değerini olasılıksal olarak modelleyeceğiz daha sonra bu modeli kullanarak kişi özelliklerini girdiğimizde kişilerin özelindeconditional expected average profit değerini hesaplayacağız.
#Bu işlemler için iki ayrı modeli kullanacağız: BG/NBD Modeli - Gamma gamma Submodel
#CLTV = BG/NBD Model * Gamma gamma Submodel

# BG-NBD (Beta Geometric/ Negative Binomial distribution) ile Expected Number of Transaction:
#BG-NBD Modeli: Satın alma sayısını tahmin etmek için kullanılan bir modeldir.
#Expected, bir rassal değişkenin beklenen değerini ifade etmek için kullanılır. Bir rassal değişkenin beklenen değeri o rassal değişkenin ortalaması demektir.
#Rassal değişken, değerlerini bir deneyin sonuçlarından alan değişkene rassal değişken denir.
#BG/NBD: Buy Till You Die
#BG/NBD modeli, Expected Number of Transaction için iki süreci olasılıksal olarak modeller: Transaction Process (Buy) + Dropout Process (till you die)
#Transaction (Buy) Process: (Satın alma işlem süreci): Alive olduğu sürece, belirli bir zaman periyodunda, bir müşteri tarafından gerçekleştirilecek işlem sayısı transaction rate parametresi ile possion dağılır.
#Bir müşteri alive olduğu sürece kendi transation rate'i etrafında rastgele satın alma yapmaya devam eder.
#Transaciton rate'ler her bir müşteriye göre dğeişir ve tüm kitle için gamma dağılır (r,a)
#Dropout [till you die) : Her bir müşterinin p olasılığı ile dropout rate(dropout probability) vardır.
#Bir müşteri alışveriş yaptıktan sonra belirli bir olasılıkla drop olur.
#Dropout rateler her bir müşteriye göre değişir ve tüm kitle için beta dağılır. (a,b)

#Gamma gamma submodel:
#Bir müşterinin işlem başına ortalama ne kadar kar getirebileceğini tahmin etmek için kullanılır.
#Bir müşterinin işlemlerinin parasal değeri (monetary) transaction value'larının ortalaması etrafında rastgele dağılır.
#Ortalama transaction value zaman içinde kullanıcılar arasında değişebilir fakat tek bir kullanıcı için değişmez.
#Ortalama transaction value tüm müşteriler arasında gamma dağılır.


# BG-NBD ve Gamma-Gamma ile CLTV Prediction
# 1. Verinin Hazırlanması (Data Preperation)
# 2. BG-NBD Modeli ile Expected Number of Transaction
# 3. Gamma-Gamma Modeli ile Expected Average Profit
# 4. BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
# 6. Çalışmanın fonksiyonlaştırılması

# 1. Verinin Hazırlanması (Data Preperation)
# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.

# Veri Seti Hikayesi
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.

# Değişkenler
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

# Gerekli Kütüphane ve Fonksiyonlar

pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

#Kuracak olduğumuz modeller olasıksal, istatiksel modeller olduğu için bu modelleri kurarken kullanacak olduğumuz değişkenlerin dağılımları sonuçları direkt etkileyebilecektir.
#Bu yüzden elimizdeki değişkenleri oluşturduktan sonra bu değişkenlerdeki aykırı değerlere işlem yapmamız gerekir. Bu nedenle boxplot yöntemi aracılığıyla aykırı değerleri tespit edeceğiz. Aykırı değerleri baskılama yöntemi ile belirlemiş olduğumuz aykırı
#değerleri belirli bir eşik değeri ile değiştireceğiz. Bunun için iki fonksiyona ihtiyacımız var (outlier_thresholds - replace_with_thresholds)
#Aykırı değer bir değişkenin genel dağılımının oldukça dışında olan değerlerdir.

#outlier_thresholds fonksiyonu kendisine girilen değişken için eşik değer belirlemektedir :
#Eşik değer için öncelikle çeyrek değerler hesaplanacak, çeyrek değerlerin farkı hesaplandıktan sonra 3. çeyreğin 1.5 iqr üstü ve 3. çeyreğin 1.5 iqr altındaki değerler üst ve alt eşik değerleri olarak belirlenecektir.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Verinin Okunması
df_=pd.read_excel("C:/Users/asus/Desktop/miuul/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()
# Veri Ön İşleme
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

# Lifetime Veri Yapısının Hazırlanması
# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde) (her kullanıcının kendi özelinde min ve max satın alma tarihleri arasında geçen zaman)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç

cltv_df = df.groupby('Customer ID').agg(
    {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                     lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
     'Invoice': lambda Invoice: Invoice.nunique(),
     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"] #işlem başına ortalama kazanç
cltv_df.describe().T

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

cltv_df["recency"] = cltv_df["recency"] / 7 #haftalık değere çevirdik

cltv_df["T"] = cltv_df["T"] / 7

# 2. BG-NBD Modelinin Kurulması
#import ettiğimiz fonksiyonlardan BetaGeoFitter isimli fonksiyonu bir model nesnesi oluşturur. Bu model nesnesi aracılığıla fit metodunu kullanarak frequency,recency,müşteri yaşı değerleri verildiğinde modeli oluşturur.
#Bu modelde beta ve gama dağılımlarını kullanır. parametreleri bulurken en çok olabilirlik metodundan yararlanır. parametre bulma işlemleri sırasında (penalizer_coef) argümanına ihityacı vardır.
#penalizer_coef modelin parametrelerinin bulunması aşamasında katsayılara uygulanacak olan ceza katsayısıdır.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

# 1 hafta içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)
#veya
bgf.predict(1,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)
#veri setine atmak için:
cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])

# 1 ay içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])
#bir ay içerisinde ne kadar satın alma olabileceğini görebilmek için:
bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

# 3 Ayda Tüm Şirketin Beklenen Satış Sayısı Nedir?
bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

cltv_df["expected_purc_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

# Tahmin Sonuçlarının Değerlendirilmesi
plot_period_transactions(bgf)
plt.show()

# 3. GAMMA-GAMMA Modelinin Kurulması

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])
cltv_df.sort_values("expected_average_profit", ascending=False).head(10)

# 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
##############################################################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=3,  # 3 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()

cltv = cltv.reset_index()

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

# 5. CLTV'ye Göre Segmentlerin Oluşturulması
cltv_final

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(50)

cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})


# 6. Çalışmanın Fonksiyonlaştırılması

def create_cltv_p(dataframe, month=3):
    # 1. Veri Ön İşleme
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2011, 12, 11)

    cltv_df = dataframe.groupby('Customer ID').agg(
        {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
         'Invoice': lambda Invoice: Invoice.nunique(),
         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    # 2. BG-NBD Modelinin Kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])

    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    # 3. GAMMA-GAMMA Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    # 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=month,  # 3 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final


df = df_.copy()

cltv_final2 = create_cltv_p(df)

cltv_final2.to_csv("cltv_prediction.csv")
