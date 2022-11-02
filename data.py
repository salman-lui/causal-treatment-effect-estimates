import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
## Hello

class Dataset():
	def __init__(self, seed):
		self.seed = seed

	def load_data(self,fname, sep=","):
		df = pd.read_csv(fname, sep=sep)
		df = df.sample(frac=1,random_state=1)    
		return df 

	def get_feat_types(self, df):
		cat_feat = []
		num_feat = []
		for key in list(df):
			if df[key].dtype==object:
				cat_feat.append(key)
			elif len(set(df[key]))>2:
				num_feat.append(key)
		return cat_feat,num_feat

	def scale_num_feats(self, df, num_feat):
		#scale numerical features
		for key in num_feat:
			scaler = StandardScaler()
			df[key] = scaler.fit_transform(df[key].values.reshape(-1,1))
		return df
	
	def split_data(self, X, y):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=self.seed, stratify=y)
		return X_train, y_train, X_test, y_test


class German(Dataset):
	def __init__(self, seed):
		super().__init__(seed)

	def get_data(self, fname):
		#From Jess' utils.py
		X = self.load_data(fname)
		y = X["GoodCustomer"]

		X = X.drop(["GoodCustomer", "PurposeOfLoan"], axis=1)
		X['Gender'] = [1 if v == "Male" else 0 for v in X['Gender'].values]

		_, num_feat = self.get_feat_types(X)
		cat_feat = [X.columns.get_loc(col) for col in ['Gender', 'ForeignWorker', 'Single', 'HasTelephone', 'CheckingAccountBalance_geq_0', 'CheckingAccountBalance_geq_200', 'SavingsAccountBalance_geq_100', 'SavingsAccountBalance_geq_500', 'MissedPayments', 'NoCurrentLoan', 'CriticalAccountOrLoansElsewhere', 'OtherLoansAtBank', 'OtherLoansAtStore', 'HasCoapplicant', 'HasGuarantor', 'OwnsHouse', 'RentsHouse', 'Unemployed', 'YearsAtCurrentJob_lt_1', 'YearsAtCurrentJob_geq_4', 'JobClassIsSkilled']]
		
		X = self.scale_num_feats(X, num_feat)

		X_train, y_train, X_test, y_test = self.split_data(X,y)
		a_test = X_test["Gender"].values


		return X_train, y_train, X_test, y_test, cat_feat, a_test


class COMPAS(Dataset):
	def __init__(self):
		super(COMPAS, self).__init__()

	def get_data(self, fname):
		#From Dylan & Jess' utils.py
		PROTECTED_CLASS = 1
		UNPROTECTED_CLASS = 0
		POSITIVE_OUTCOME = 1
		NEGATIVE_OUTCOME = 0
	
		compas_df = self.load_data(fname)
		compas_df = compas_df.loc[(compas_df['days_b_screening_arrest'] <= 30) &
							  (compas_df['days_b_screening_arrest'] >= -30) &
							  (compas_df['is_recid'] != -1) &
							  (compas_df['c_charge_degree'] != "O") &
							  (compas_df['score_text'] != "NA")]

		compas_df['length_of_stay'] = (pd.to_datetime(compas_df['c_jail_out']) - pd.to_datetime(compas_df['c_jail_in'])).dt.days
		X = compas_df[['age', 'two_year_recid','c_charge_degree', 'race', 'sex', 'priors_count', 'length_of_stay']]

		# if person has high score give them the _negative_ model outcome
		y = compas_df['two_year_recid']
		sens = X.pop('race')
		X.pop('two_year_recid')
		

		# assign African-American as the protected class
		X = pd.get_dummies(X)
		sensitive_attr = np.array(pd.get_dummies(sens).pop('African-American'))
		X['race'] = sensitive_attr

		X.pop('sex_Male')

		# make sure everything is lining up
		assert all((sens == 'African-American') == (X['race'] == PROTECTED_CLASS))
		cols = [col for col in X]

		# ADDED BY JESS -- so race = 1 is ADVANTAGED
		X['race'] = 1 - X['race']

		X_train, y_train, X_test, y_test = self.split_data(X,y)

		a_test = X_test["race"].values

		cat_feat = [i for i in range(len(cols)) if cols[i] in ['c_charge_degree_F', 'c_charge_degree_M', 'sex_Female', 'race']]

		return X_train, y_train, X_test, y_test, cat_feat, a_test


class Adult(Dataset):
	def __init__(self):
		super(German).__init__()

	def get_data(self, train_fname, test_fname):
		#From https://fairmlbook.org/code/adult.html
		features = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
		"Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
		"Hours per week", "Country", "Target"] 

		X_train = pd.read_csv(train_fname, names=features, sep=r'\s*,\s*', 
									 engine='python', na_values="?")
		X_test = pd.read_csv(test_fname, names=features, sep=r'\s*,\s*', 
									engine='python', na_values="?", skiprows=1)

		num_train = len(X_train)
		X = pd.concat([X_train, X_test])
   
		y = X['Target']
		y = y.replace('<=50K', 0).replace('>50K', 1)
		y = y.replace('<=50K.', 0).replace('>50K.', 1)

		# Redundant column
		del X["Education-Num"]

		# Remove target variable
		del X["Target"]

		#added by sohini
		X["Sex"] = [0 if v == "Female" else 1 for v in X['Sex'].values]

		cat_feat, num_feat = self.get_feat_types(X)
		
		X = pd.get_dummies(X, columns=cat_feat)

		X = self.scale_num_feats(X, num_feat)

		X_train, y_train, X_test, y_test = self.split_data(X,y)

		a_test = X_test["Sex"].values

		cat_feat = [X.columns.get_loc(col) for col in X if col not in num_feat]

		return X_train, y_train, X_test, y_test, cat_feat, a_test


class Student(Dataset):
	def __init__(self):
		super(German).__init__()

	def get_data(self, fname, sep=";"):
		df = self.load_data(fname, sep)

		#Define target variable
		df["Outcome"] = (df["G3"]<10).astype(int)

		#Drop variables highly correlated with target
		df = df.drop(columns=["G1","G2","G3"])

		df["sex"] = [0 if v == "F" else 1 for v in df['sex'].values]

		cat_feat,num_feat = self.get_feat_types(df)

		#One-hot encode categorical features
		df = pd.get_dummies(df, columns=cat_feat)

		#Scale numerical features
		df = self.scale_num_feats(df, num_feat)

		X, y = df.drop(columns=["Outcome","school_GP","school_MS"]), df["Outcome"]

		X_train, y_train, X_test, y_test = self.split_data(X,y)

		a_test = X_test["sex"].values
		
		cat_feat = [X.columns.get_loc(col) for col in X if col not in num_feat]

		return X_train, y_train, X_test, y_test, cat_feat, a_test

