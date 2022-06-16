#IFF-9/11 Vidas Buivydas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler


cols=['Pos','Age','G','MP','AST','PTS','FG','FGA']
data = pd.read_csv('2021-2022 NBA Player Stats.csv', usecols=cols, sep=';')
#print(data)
dataCount = data.count()
#2
print("bendras reikšmių skaičius:")
print(dataCount)

print("\ntrūkstamos reikšmės, procentais:")
missingPercent = data.isnull().sum() / dataCount * 100
print(missingPercent)

print("\nkardinalumas:")
cardinality = data.nunique()
print(cardinality)

print("\nmin reikšmės:")
min = data.min()
print(min)

print("\nmax reikšmės:")
max = data.max()
print(max)

print("\n1-asis ir 3-asis kvartiliai:")
quantiles = data.quantile([0.25,0.75])
print(quantiles)

print("\nvidurkiai:")
average = data.mean()
print(average)

print("\nmediana:")
median = data.median()
print(median)

print("\nstandartinis nuokrypis:")
standardDeviation = data.std()
print(standardDeviation)

#3

print("\nmodos:")
mode = data.mode()
print("pozicijos moda:")
posMode = mode["Pos"][0]
print(posMode)
print("amžiaus moda:")
ageMode = mode["Age"][0]
print(ageMode)
print("žaistų žaidimų moda:")
gamesMode = mode["G"][0]
print(gamesMode)


print("\nmodos dažnumas:")
print("pozicijos modos dažnumas:")
posModeCount = (len(data[data["Pos"] == posMode]))
print(posModeCount)
print("amžiaus modos dažnumas:")
ageModeCount = (len(data[data["Age"] == ageMode]))
print(ageModeCount)
print("žaistų žaidimų modos dažnumas:")
gamesModeCount = (len(data[data["G"] == gamesMode]))
print(gamesModeCount)

print("\nmoda, %:")
print("pozicijos moda, %:")
posDataCount = data["Pos"].count()
posModePercantage = posModeCount / posDataCount
print(posModePercantage*100)
print("amžiaus moda, %:")
ageDataCount = data["Age"].count()
ageModePercantage = ageModeCount / ageDataCount
print(ageModePercantage*100)
print("žaistų žaidimų moda, %:")
gamesDataCount = data["G"].count()
gamesModePercantage = gamesModeCount / gamesDataCount
print(gamesModePercantage*100)

print("\n2-osios modos:")
dataPos = data[data.Pos != posMode]
dataAge = data[data.Age != ageMode]
dataGames = data[data.G != gamesMode]

pMode2 = dataPos.mode()
print("pozicijos 2-oji moda:")
posMode2 = pMode2["Pos"][0]
print(posMode2)
aMode2 = dataAge.mode()
print("amžiaus 2-oji moda:")
ageMode2 = aMode2["Age"][0]
print(ageMode2)
gMode2 = dataGames.mode()
print("žaistų žaidimų 2-oji moda:")
gamesMode2 = gMode2["G"][0]
print(gamesMode2)

print("\n2-osios modos dažnumas:")
print("pozicijos 2-osios modos dažnumas:")
posMode2Count = (len(dataPos[dataPos["Pos"] == posMode2]))
print(posMode2Count)
print("amžiaus 2-osios modos dažnumas:")
ageMode2Count = (len(dataAge[dataAge["Age"] == ageMode2]))
print(ageMode2Count)
print("žaistų žaidimų 2-osios modos dažnumas:")
gamesMode2Count = (len(dataGames[dataGames["G"] == gamesMode2]))
print(gamesMode2Count)

print("\n2- oji moda, %:")
print("pozicijos 2- oji moda, %:")
posMode2Percantage = posMode2Count / posDataCount
print(posMode2Percantage*100)
print("amžiaus 2- oji moda, %:")
ageMode2Percantage = ageMode2Count / ageDataCount
print(ageMode2Percantage*100)
print("žaistų žaidimų 2- oji moda, %:")
gamesMode2Percantage = gamesMode2Count / gamesDataCount
print(gamesMode2Percantage*100)


#4
data["Pos"].hist()
plt.title("Krepšininko pozicija")
plt.ylabel('krepšininkų skaičius')
plt.show()
data["Age"].plot.hist(bins = 22, title='Krepšininko amžius')
plt.ylabel('krepšininkų skaičius')
plt.show()
data["G"].plot.hist(bins = 22, title='Krepšininko žaisti žaidimai')
plt.ylabel('krepšininkų skaičius')
plt.show()
data["MP"].plot.hist(bins = 22, title='Vidutinis praleistų minučių skaičius rungtynėse')
plt.ylabel('krepšininkų skaičius')
plt.show()
data["AST"].plot.hist(bins = 22, title='Vidutinis padėjimų skaičius rungtynėse')
plt.ylabel('krepšininkų skaičius')
plt.show()
data["PTS"].plot.hist(bins = 22, title='Vidutinis pelnytų taškų skaičius per rungtynes')
plt.ylabel('krepšininkų skaičius')
plt.show()
data["FG"].plot.hist(bins = 22, title='Vidutinis sėkmingų pataikymų i krepšį per rungtynes skaičius')
plt.ylabel('krepšininkų skaičius')
plt.show()
data["FGA"].plot.hist(bins = 22, title='bandymų pataikyti i krepšį per rungtynes skaičius')
plt.ylabel('krepšininkų skaičius')
plt.show()

#6
data.boxplot(column="Age", return_type='axes');
plt.show()

data.boxplot(column="G", return_type='axes');
plt.show()

data.boxplot(column="MP", return_type='axes');
plt.show()

data.boxplot(column="AST", return_type='axes');
plt.show()

data.boxplot(column="PTS", return_type='axes');
plt.show()

data.boxplot(column="FG", return_type='axes');
plt.show()

data.boxplot(column="FGA", return_type='axes');
plt.show()

#triuksmu salinimas pagal formule “Age”, “AST”, “PTS”,  “FG” ir “FGA” atributams
print(quantiles.iat[0,0]); 
print(quantiles.iat[1,0]); 
ageLow = quantiles.iat[0,0] - 1.5 * (quantiles.iat[1,0] - quantiles.iat[0,0])
ageTop = quantiles.iat[1,0] + 1.5 * (quantiles.iat[1,0] - quantiles.iat[0,0])
print("AgeLowTop", ageLow, ageTop)

print(quantiles.iat[0,3]); 
print(quantiles.iat[1,3]); 
ASTLow = quantiles.iat[0,3] - 1.5 * (quantiles.iat[1,3] - quantiles.iat[0,3])
ASTTop = quantiles.iat[1,3] + 1.5 * (quantiles.iat[1,3] - quantiles.iat[0,3])
print("ASTLowTop", ASTLow, ASTTop)

print(quantiles.iat[0,4]); 
print(quantiles.iat[1,4]); 
PTSLow = quantiles.iat[0,4] - 1.5 * (quantiles.iat[1,4] - quantiles.iat[0,4])
PTSTop = quantiles.iat[1,4] + 1.5 * (quantiles.iat[1,4] - quantiles.iat[0,4])
print("PTSLowTop", PTSLow, PTSTop)
print(quantiles)
print(quantiles.iat[0,5]); 
print(quantiles.iat[1,5]); 
FGLow = quantiles.iat[0,5] - 1.5 * (quantiles.iat[1,5] - quantiles.iat[0,5])
FGTop = quantiles.iat[1,5] + 1.5 * (quantiles.iat[1,5] - quantiles.iat[0,5])
print("FGLowTop", FGLow, FGTop)

print(quantiles.iat[0,6]); 
print(quantiles.iat[1,6]); 
FGALow = quantiles.iat[0,6] - 1.5 * (quantiles.iat[1,6] - quantiles.iat[0,6])
FGATop = quantiles.iat[1,6] + 1.5 * (quantiles.iat[1,6] - quantiles.iat[0,6])
print("FGALowTop", FGALow, FGATop)

data.loc[data['Age'] < ageLow, 'Age'] = ageLow
data.loc[data['Age'] > ageTop, 'Age'] = ageTop
data["Age"].plot.hist(bins = 17, title='Krepšininko amžius be triukšmų')
plt.ylabel('krepšininkų skaičius')
plt.show()

data.loc[data['AST'] < ASTLow, 'AST'] = ASTLow
data.loc[data['AST'] > ASTTop, 'AST'] = ASTTop
data["AST"].plot.hist(bins = 22, title='Vidutinis padėjimų skaičius rungtynėse be triukšmų')
plt.ylabel('krepšininkų skaičius')
plt.show()

data.loc[data['PTS'] < PTSLow, 'PTS'] = PTSLow
data.loc[data['PTS'] > PTSTop, 'PTS'] = PTSTop
data["PTS"].plot.hist(bins = 22, title='Vidutinis pelnytų taškų skaičius per rungtynes be triukšmų')
plt.ylabel('krepšininkų skaičius')
plt.show()

data.loc[data['FG'] < FGLow, 'FG'] = FGLow
data.loc[data['FG'] > FGTop, 'FG'] = FGTop
data["FG"].plot.hist(bins = 22, title='Vidutinis sėkmingų pataikymų i krepšį per rungtynes skaičius be triukšmų')
plt.ylabel('krepšininkų skaičius')
plt.show()

data.loc[data['FGA'] < FGALow, 'FGA'] = FGALow
data.loc[data['FGA'] > FGATop, 'FGA'] = FGATop
data["FGA"].plot.hist(bins = 22, title='bandymų pataikyti i krepšį per rungtynes skaičius be triukšmų')
plt.ylabel('krepšininkų skaičius')
plt.show()

#7 stipri koreleacija
data.plot.scatter(x='PTS',y='FGA')
plt.show()

data.plot.scatter(x='MP',y='PTS')
plt.show()

data.plot.scatter(x='FG',y='FGA')
plt.show()
#splom
pd.plotting.scatter_matrix(data)
plt.show()
#kategoriniu vizualizacija
data['Age'].value_counts().plot(kind='bar');
plt.title("Krepšininko amžius")
plt.ylabel("krepšininkų skaičius")

plt.show()

dataAge22 = data.loc[data['Age'] == 22] 
dataAge22['Pos'].value_counts().plot(kind='bar');
plt.title("Pozicijų skaičius, kai krepšininko amžius 22 metai")
plt.ylabel("krepšininkų skaičius")

plt.show()

dataAge33 = data.loc[data['Age'] == 33] 
dataAge33['Pos'].value_counts().plot(kind='bar');
plt.title("Pozicijų skaičius, kai krepšininko amžius 33 metai")
plt.ylabel("krepšininkų skaičius")
plt.show()
#kategoriniu ir tolydziuju vizualizacija

data.boxplot(by ='MP', column =['G'], grid = False, figsize = (20, 10))
plt.title("Sąryšis tarp krepšininko vidutinio praleistų minučių skaičiaus rungtynėse ir jo žaistų žaidimų skaičiaus")
plt.suptitle('')
plt.xlabel("vidutinis praleistas laikas rungtynėse")
plt.ylabel("žaistų žaidimų skaičius")
plt.show()

data.boxplot(by ='Pos', column =['MP'], grid = False)
plt.title("Sąryšis tarp krepšininko pozicijos ir jo praleistų minučių žaidime")
plt.suptitle('')
plt.xlabel("Krepšininko pozicija")
plt.ylabel("Krepšininko žaisti žaidimai")
plt.show()

#koreleacija
data.cov()
corr = data.corr()
sb.heatmap(corr, cmap="Reds", annot=True)
plt.title("Koreleacijos matricos diagrama")
plt.show()
#kategoriniu vertimas tolydziais

data['Pos'] = data['Pos'].replace(['C'],'1')
data['Pos'] = data['Pos'].replace(['PF'],'2')
data['Pos'] = data['Pos'].replace(['SG'],'3')
data['Pos'] = data['Pos'].replace(['PG'],'4')
data['Pos'] = data['Pos'].replace(['SF'],'5')
#print(data)
#df_pos = pd.get_dummies(data['Pos'])
#df_newPos = pd.concat([data, df_pos], axis=1)
#print(df_newPos)

#normalizacija
scaler = MinMaxScaler()
del data["Pos"]
scaler.fit(data)

# get max values in each column
max = scaler.data_max_

normalized = scaler.transform(data)
print(scaler.transform(data))
# kovariacija
data.cov()