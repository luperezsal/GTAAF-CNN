from ctgan import CTGAN

class cGAN:


	def __init__(self,X_train, Y_train, n_epochs):
		self.X_train = X_train
		self.Y_train = Y_train
		self.n_epochs = n_epochs

	def call(self):

		transf = {
		    'Slight': 0,
		    'Assistance': 1
		}   

		X_train_temp_for_cgan = self.X_train
		Y_train_temp_for_cgan = self.Y_train
		Y_train_temp_for_cgan = Y_train_temp_for_cgan.replace(transf)

		train_temp_for_cgan = pd.concat([X_train_temp_for_cgan,Y_train_temp_for_cgan], axis=1)

		target_column = ['lesividad']

		ctgan = CTGAN(epochs=self.n_epochs, verbose=False, cuda=False)
		ctgan.fit(train_temp_for_cgan, target_column)

		cgan_samples = ctgan.sample(10000, 'lesividad', 1)

		assistance_cgan_samples = cgan_samples[cgan_samples.lesividad == 1][:700]
		X_assistance_cgan_samples = assistance_cgan_samples.iloc[:,:-1]
		Y_assistance_cgan_samples = assistance_cgan_samples.iloc[:,-1:]

		transf = {
		    0: 'Slight',
		    1: 'Assistance'
		}   


		Y_assistance_cgan_samples = Y_assistance_cgan_samples.replace(transf)

		Y_assistance_cgan_samples

		X_train = pd.concat([self.X_train, X_assistance_cgan_samples], axis=0)
		Y_train = pd.concat([self.Y_train, Y_assistance_cgan_samples.lesividad], axis=0)



		return X_train_upsampled, Y_train_upsampled