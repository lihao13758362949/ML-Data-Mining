def get_data(file_name):
  data=pd.read_csv(file_name)
  X = []
  Y = []
  for i,j in zip(data[0],data[1]):
    X.append([float(i)])
    Y.append([float(j)])
  return X,Y
