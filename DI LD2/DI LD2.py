!pip install -U scikit-fuzzy
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

#starting data
DataPeopleCount = 60
DataOriginality = 9
DataTime = 30

#DataPeopleCount = 5
#DataOriginality = 2
#DataTime = 32

#DataPeopleCount = 65
#DataOriginality = 4
#DataTime = 7

x_peopleC = np.arange(0,150,0.1)
x_originality = np.arange(0,10,0.01)
x_time = np.arange(0,36,0.1)
x_popularity = np.arange(0,10,0.01)

peopleC_low = fuzz.trapmf(x_peopleC, [0, 0, 10, 35])
peopleC_mid = fuzz.trapmf(x_peopleC, [20, 60, 90, 130])
peopleC_high = fuzz.trapmf(x_peopleC, [100, 140, 150, 176.5])

originality_low = fuzz.trapmf(x_originality, [0, 0, 2, 4.5])
originality_mid = fuzz.trapmf(x_originality, [3, 5, 7, 9])
originality_high = fuzz.trapmf(x_originality, [7.5, 10, 10, 10])

time_low = fuzz.trapmf(x_time, [0, 0, 3, 8])
time_mid = fuzz.trapmf(x_time, [5, 11, 22, 29])
time_high = fuzz.trapmf(x_time, [24, 36, 51, 51])

popularity_veryLow = fuzz.trapmf(x_popularity, [0, 0, 1, 3])
popularity_low = fuzz.trapmf(x_popularity, [2, 3.5, 4.5, 6])
popularity_mid = fuzz.trapmf(x_popularity, [5, 6.5, 7,8.5])
popularity_high = fuzz.trapmf(x_popularity, [7.5, 9, 10, 10])


fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(10, 14))
ax0.plot(x_peopleC, peopleC_low, 'green', linewidth=2, label='Mažas')
ax0.plot(x_peopleC, peopleC_mid, 'red', linewidth=2, label='Vidutinis')
ax0.plot(x_peopleC, peopleC_high, 'black', linewidth=2, label='Aukštas')
ax0.set_title('Kūrėjų skaičius')
ax0.legend()
ax1.plot(x_originality, originality_low, 'green', linewidth=2, label='Neoriginalus')
ax1.plot(x_originality, originality_mid, 'red', linewidth=2, label='Originalus')
ax1.plot(x_originality, originality_high, 'black', linewidth=2, label='Labai originalus')
ax1.set_title('Originalumas')
ax1.legend()
ax2.plot(x_time, time_low, 'green', linewidth=2, label='Trumpas')
ax2.plot(x_time, time_mid, 'red', linewidth=2, label='Vidutinis')
ax2.plot(x_time, time_high, 'black', linewidth=2, label='Ilgas')
ax2.set_title('Laikas (mėnesiais)')
ax2.legend()
ax3.plot(x_popularity, popularity_veryLow, 'green', linewidth=2, label='Nepopulariarus')
ax3.plot(x_popularity, popularity_low, 'blue', linewidth=2, label='Šiek tiek populiarus')
ax3.plot(x_popularity, popularity_mid, 'red', linewidth=2, label='Popularus')
ax3.plot(x_popularity, popularity_high, 'black', linewidth=2, label='Labai populiarus')
ax3.set_title('Populiarumas')
ax3.legend()

peopleC_level_low = fuzz.interp_membership(x_peopleC, peopleC_low, DataPeopleCount)
peopleC_level_mid = fuzz.interp_membership(x_peopleC, peopleC_mid, DataPeopleCount)
peopleC_level_high = fuzz.interp_membership(x_peopleC, peopleC_high, DataPeopleCount)
originality_level_low = fuzz.interp_membership(x_originality, originality_low, DataOriginality)
originality_level_mid = fuzz.interp_membership(x_originality, originality_mid, DataOriginality)
originality_level_high = fuzz.interp_membership(x_originality, originality_high, DataOriginality)
time_level_low = fuzz.interp_membership(x_time, time_low, DataTime)
time_level_mid = fuzz.interp_membership(x_time, time_mid, DataTime)
time_level_high = fuzz.interp_membership(x_time, time_high, DataTime)

not_high_time = np.fmax(time_level_low, time_level_mid)
not_low_time =  np.fmax(time_level_mid, time_level_high)
not_low_peopleC=np.fmax(peopleC_level_mid,peopleC_level_high)

#rules of very low popularity

rule1 = np.fmin(peopleC_level_low, np.fmin(originality_level_low , not_high_time))
rule2 = np.fmin(peopleC_level_mid , np.fmax(originality_level_low ,time_level_low))

vLow_prob = np.fmax(rule1,rule2)
popularity_actv_vLow = np.fmin(vLow_prob , popularity_veryLow)
print("Nepopuliarus resultatas:")
print(np.round(vLow_prob,2))

#rules of low popularity
rule4=np.fmin(peopleC_level_low ,np.fmin(originality_level_low, time_level_high))
rule5=np.fmin(peopleC_level_low, np.fmin(originality_level_mid, not_high_time ))
rule6=np.fmin(peopleC_level_mid, np.fmin(originality_level_mid, time_level_low))
rule7=np.fmin(peopleC_level_mid, np.fmin(originality_level_mid, time_level_low))
rule8=np.fmin(peopleC_level_high,np.fmin(originality_level_low, not_high_time ))

low_prob = np.fmax(rule8, np.fmax(rule7, np.fmax(rule6, np.fmax(rule4, rule5))))
popularity_actv_low = np.fmin(low_prob , popularity_low)
print("Šiek tiek populiarus rezultatas:")
print(np.round(low_prob,2))

#rules of medium popularity
rule9=np.fmin(peopleC_level_low ,np.fmin(originality_level_mid, time_level_high))
rule10=np.fmin(peopleC_level_low, originality_level_high)
rule11=np.fmin(peopleC_level_mid, np.fmin(originality_level_mid, not_low_time))
rule12=np.fmin(peopleC_level_mid, np.fmin(originality_level_high, not_high_time))
rule13=np.fmin(peopleC_level_high,np.fmin(originality_level_low, time_level_high))
rule14=np.fmin(peopleC_level_high,np.fmin(originality_level_mid, not_high_time))
rule15=np.fmin(peopleC_level_high,np.fmin(originality_level_high, time_level_low))

mid_prob = np.fmax(rule15, np.fmax(rule14, np.fmax(rule13, np.fmax(rule12, np.fmax(rule11, np.fmax(rule9, rule10))))))
popularity_actv_mid = np.fmin(mid_prob , popularity_mid)
print("populiarus resultatas:")
print(np.round(mid_prob,2))

#rules of high popularity
rule16=np.fmin(not_low_peopleC,np.fmin(originality_level_high, time_level_high))
rule17=np.fmin(peopleC_level_high,np.fmin(originality_level_mid, time_level_high))
rule18=np.fmin(peopleC_level_high,np.fmin(originality_level_high, time_level_mid))

high_prob = np.fmax(rule18, np.fmax(rule16, rule17))
popularity_actv_high = np.fmin(high_prob , popularity_high)
print("labai populiarus result:")
print(np.round(high_prob,2))

pop = np.zeros_like(x_popularity)
fig, ax0 = plt.subplots(figsize=(8, 3))
ax0.fill_between(x_popularity, pop, popularity_actv_vLow, facecolor='yellow', alpha=0.7)
ax0.plot(x_popularity, popularity_veryLow, 'green', linewidth=1, linestyle='--', )
ax0.fill_between(x_popularity, pop, popularity_actv_low, facecolor='aqua', alpha=0.7)
ax0.plot(x_popularity, popularity_low, 'red', linewidth=1, linestyle='--', )
ax0.fill_between(x_popularity, pop, popularity_actv_mid, facecolor='red', alpha=0.7)
ax0.plot(x_popularity, popularity_mid, 'blue', linewidth=1, linestyle='--')
ax0.fill_between(x_popularity, pop, popularity_actv_high, facecolor='lime', alpha=0.7)
ax0.plot(x_popularity, popularity_high, 'black', linewidth=1, linestyle='--')
ax0.set_title('Agreguota populiarumo funkcija')

# Agregavimas
aggregated = np.fmax(popularity_actv_vLow,np.fmax(popularity_actv_low, np.fmax(popularity_actv_mid, popularity_actv_high)))

# defuzifikacija
pop_centr = fuzz.defuzz(x_popularity, aggregated, 'centroid')
pop_mom = fuzz.defuzz(x_popularity, aggregated, 'mom')

pop_activ = fuzz.interp_membership(x_popularity, aggregated, pop_centr)

fig, ax0 = plt.subplots(figsize=(8, 3))
ax0.plot(x_popularity, popularity_veryLow, 'red', linewidth=1, linestyle='--', )
ax0.plot(x_popularity, popularity_low, 'turquoise', linewidth=1, linestyle='--', )
ax0.plot(x_popularity, popularity_mid, 'blue', linewidth=1, linestyle='--')
ax0.plot(x_popularity, popularity_high, 'green', linewidth=1, linestyle='--')
ax0.fill_between(x_popularity, pop, aggregated, facecolor='aquamarine', alpha=0.7)
ax0.plot([pop_centr, pop_centr], [0, pop_activ], 'black', linewidth=1.5, alpha=0.9)
ax0.plot([pop_mom, pop_mom], [0, pop_mom], 'darkgreen', linewidth=1.5, alpha=0.9)
ax0.set_ylim(0,1.1)
ax0.set_title('Gauta populiarumo reikšmė')

print("Defuzzified results:")
print("Defuzzification: Centroid")
print(np.round(pop_centr,2))
print("Defuzzification: Mean of maximum")
print(np.round(pop_mom,2))

plt.show()