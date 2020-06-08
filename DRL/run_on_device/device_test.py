import sys

sys.path.append('../')
#sys.path.append('/home/Documents/deep-reinforcement-learning-for-fine-tuning/drl_algorithm/Pygor')
print("path ",sys.path)
import Pygor

pygor_mode = 'None'
pygor_xmlip = "http://129.67.86.107:8000/RPC2"
#pygor_xmlip = None

pygor = Pygor.Experiment(mode=pygor_mode, xmlip=pygor_xmlip)

pygor.setvals(gates,[-1150.91240972, -1947.66738027, -1050, -1172.60790138, -1149.99759364,  -1050, -837.49978806 ])

[c5_val,c9_val] = pygor.getvals(["c5","c9"])

n = 100

data = pygor.do2d("c5",c5_val-350,c5_val+350,n,"c9",c9_val-350,c9_val+350,n)
plt.imshow(data.data[0])
plt.colorbar()
plt.show()
