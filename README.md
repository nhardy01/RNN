# RNN

A Matlab package for training and testing recurrent neural networks (RNN). The package also contains a few helper functions to assist with various training methods and tasks.

There main files are:

**RNN.m**: Class definition for the RNN. Contains methods for instantiating, training, and running the model. Training is accomplished mainly using the innate learning rule.
For more information see:
	Laje, R., and Buonomano, D.V. (2013). Robust timing and motor patterns
 	by taming chaos in recurrent neural networks. Nat. Neurosci. 16,
 	925�933.
  Hardy, N.F., Goudar, V., Romero-Sosa, J.L., and Buonomano,
 	D. (2017). A Model of Temporal Scaling Correctly Predicts that Weber's
 	Law is Speed-dependent. BioRxiv 159590.

  **TrainRNN_TempInv.m**: Function to train RNNs to produce temporally invariant activity (temporal scaling).

  **TestRNN.m**: Function to test RNNs.

  **TrainOut.m**: Function to train the output units of the RNN.
