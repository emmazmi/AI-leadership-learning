import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering
import matplotlib.cm as cm
from scipy.cluster.hierarchy import ward, dendrogram
from scipy.cluster.hierarchy import fcluster
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('AI_sample_data.csv', engine='python')

df["Dateofbirth"] = pd.to_datetime(df.DOB)
df["Yearofbirth"] = df.Dateofbirth.dt.year
df["Age"] = df.Cohorts_Start_Year - df.Yearofbirth

Xy = df[['Band', 'Role', 'Age', 'Contacts_Ethnic_Origin_Desc', 'Gender', 'Disability', 'Programme']]

Xy['Band'].replace(['8a', '8b', '8c', '8d', 'VSM', 'DNWD'], ['8', '8', '8', '8', '10', np.nan], inplace=True)
Xy['Role'].replace(['Registrar', 'Consultant', 'GP', 'F1', 'Leadership Role', 'Director', 'CEO / MD', 'Performance', 'HR / Workforce', 'Finance', 'Comms', 'Admin and clerical', 'Midwife', 'Commissioning'], ['Doctor', 'Doctor', 'Doctor', 'Doctor', 'Executive', 'Executive', 'Executive', 'Organisational', 'Organisational', 'Organisational', 'Organisational', 'Organisational', 'Nurse', 'Executive'], inplace=True)
#Xy.Contacts_Ethnic_Origin_Desc = Xy.Contacts_Ethnic_Origin_Desc.apply(lambda x: 'Mixed' if 'mixed' in str(x) else x)
#Xy.Contacts_Ethnic_Origin_Desc = Xy.Contacts_Ethnic_Origin_Desc.apply(lambda x: 'Mixed' if 'Mixed' in str(x) else x)
Xy.Contacts_Ethnic_Origin_Desc = Xy.Contacts_Ethnic_Origin_Desc.apply(lambda x: 'Black' if 'Black' in str(x) else x)
Xy.Contacts_Ethnic_Origin_Desc = Xy.Contacts_Ethnic_Origin_Desc.apply(lambda x: 'Black' if 'black' in str(x) else x)
Xy.Contacts_Ethnic_Origin_Desc = Xy.Contacts_Ethnic_Origin_Desc.apply(lambda x: 'Asian' if 'Asian' in str(x) else x)
Xy.Contacts_Ethnic_Origin_Desc = Xy.Contacts_Ethnic_Origin_Desc.apply(lambda x: 'Asian' if 'asian' in str(x) else x)
Xy.Contacts_Ethnic_Origin_Desc = Xy.Contacts_Ethnic_Origin_Desc.apply(lambda x: 'White' if 'White' in str(x) else x)
Xy.Contacts_Ethnic_Origin_Desc = Xy.Contacts_Ethnic_Origin_Desc.apply(lambda x: 'Other' if 'Other' in str(x) else x)
Xy.Contacts_Ethnic_Origin_Desc = Xy.Contacts_Ethnic_Origin_Desc.apply(lambda x: 'Other' if 'other' in str(x) else x)
Xy['Contacts_Ethnic_Origin_Desc'].replace(['Do not wish to disclose'], [np.nan], inplace=True)
Xy['Disability'].replace(['No ', 'Not disclosed'], ['No', np.nan], inplace=True)

Xy.Programme = Xy.Programme.apply(lambda x: 'Seacole' if 'Seacole' in str(x) else x)
Xy.Programme = Xy.Programme.apply(lambda x: 'Stepping Up' if 'Stepping Up' in str(x) else x)

Xy.dropna(inplace=True)

#print(Xy.shape)

Xy['Band'] = Xy['Band'].astype(int)

X = Xy[['Band', 'Role', 'Age', 'Contacts_Ethnic_Origin_Desc', 'Gender', 'Disability']]
y = Xy[['Programme']]

X_rf = X
X_cl = X

#print(X['Role'].unique())
#print(X['Contacts_Ethnic_Origin_Desc'].unique())
#print(X['Gender'].unique())
#print(X['Disability'].unique())
#print(y['Programme'].unique())

################################################################################

X_rf = pd.get_dummies(X_rf, prefix=['Role'], columns=['Role'])
X_rf = pd.get_dummies(X_rf, prefix=['Ethnic'], columns=['Contacts_Ethnic_Origin_Desc'])
#X_rf = pd.get_dummies(X_rf, prefix=['Gender'], columns=['Gender'])
#X_rf = pd.get_dummies(X_rf, prefix=['Disability'], columns=['Disability'])

#cat = pd.Categorical(X_rf['Role'], categories=['Admin and clerical', 'Organisational', 'AHP', 'Midwife', 'Nurse', 'Doctor', 'Commissioning', 'Executive'])
#codes, uniques = pd.factorize(cat)
#X_rf.Role = codes
#definitions_role = uniques

#cat2 = pd.Categorical(X_rf['Contacts_Ethnic_Origin_Desc'], categories=['White', 'Asian', 'Black', 'Chinese', 'Mixed'])
#codes2, uniques2 = pd.factorize(cat2)
#X_rf.Contacts_Ethnic_Origin_Desc = codes2
#definitions_ethnic = uniques2

cat3 = pd.Categorical(X_rf['Gender'], categories=['Female', 'Male'])
codes3, uniques3 = pd.factorize(cat3)
X_rf.Gender = codes3
definitions_gender = uniques3
#print(definitions_gender)

cat4 = pd.Categorical(X_rf['Disability'], categories=['No', 'Yes'])
codes4, uniques4 = pd.factorize(cat4)
X_rf.Disability = codes4
definitions_disability = uniques4
#print(definitions_disability)

#cat5 = pd.Categorical(y['Programme'], categories=['Jenner', 'Seacole', 'Anderson', 'Bevan', 'Stepping Up', 'Ready Now', 'Coaching for Inclusion', 'Aspiring Director of Nursing Talent Scheme', 'Aspiring Chief Executive'])
#codes5, uniques5 = pd.factorize(cat5)
#y.Programme = codes5
#definitions_programme = uniques5
#print(definitions_programme)

factor = pd.factorize(y['Programme'])
y.Programme = factor[0]
definitions = factor[1]
#print(definitions)

X_rf_columns = X_rf.columns
#print(X_rf.columns)
#print(y.columns)

X_train, X_test, y_train, y_test = train_test_split(X_rf, y, test_size=0.2, random_state=0)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(min_samples_split=15, random_state=0).fit(X_train_scaled, y_train)

y_train_predict = clf.predict(X_train_scaled)
y_test_predict = clf.predict(X_test_scaled)

print(clf.score(X_train_scaled, y_train))
print(clf.score(X_test_scaled, y_test))
print(confusion_matrix(y_train, y_train_predict))
print(confusion_matrix(y_test, y_test_predict))

######################################################################################

dist_matrix = distance_matrix(X_rf, X_rf)
Z = hierarchy.linkage(dist_matrix, 'average')

#max_d = 3
#clusters = fcluster(Z, max_d, criterion='distance')

#k = 5
#clusters = fcluster(Z, k, criterion='maxclust')

dendro = hierarchy.dendrogram(Z)
plt.savefig('AI sample data dendrogram.png')

#plt.figure()
#dendrogram(ward(X))
#plt.savefig('AI sample data dendrogram 1')

scaler.fit(X_rf)
X_rf_scaled = scaler.transform(X_rf)

agglom = AgglomerativeClustering(n_clusters = 7, linkage = 'average')
agglom.fit(X_rf_scaled)
X_rf['cluster'] = agglom.labels_

print(X_rf['cluster'].value_counts())

n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

plt.figure(figsize=(16,14))
for color, label in zip(colors, cluster_labels):
    subset = X_rf[X_rf.cluster == label]
    plt.scatter(subset.Age, subset.Band, c=color, label='cluster'+str(label))
plt.legend()
plt.xlabel('Age')
plt.ylabel('Band')
plt.savefig('AI sample data clusters.png')
plt.clf()

agg_learners = X_rf.groupby(['cluster'])[list(X_rf_columns)].mean()
agg_learners.to_csv('AI sample data clusters.csv')

plt.figure(figsize=(16,14))
for color, label in zip(colors, cluster_labels):
    subset = agg_learners.loc[(label,),]
    plt.scatter(subset.Age, subset.Band, c=color, label='cluster'+str(label))
plt.legend()
plt.xlabel('Age')
plt.ylabel('Band')
plt.savefig('AI sample data cluster means.png')