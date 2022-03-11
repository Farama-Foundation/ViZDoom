import pandas as pd
import matplotlib.pyplot as plt 


df = pd.read_csv("./logs/torch-beast-results/torchbeast-20220311-193156/logs.csv")
#df2 = pd.read_csv("./logs/torchbeast-train_test/logs.csv")
#print(df.columns)

plt.figure(1)

plt.subplot(321)
plt.plot(df['# _tick'].values, df.entropy_loss.values, 'r')
plt.title('entropy_loss') 

plt.subplot(322)
plt.plot(df['# _tick'].values, df.total_loss.values, 'r')
plt.title('total_loss') 

plt.subplot(323)
plt.plot(df['# _tick'].values, df.pg_loss.values, 'r')
plt.title('pg_loss') 

plt.subplot(324)
plt.plot(df['# _tick'].values, df.baseline_loss.values, 'r')
plt.title('baseline_loss') 

plt.subplot(325)
plt.plot(df['# _tick'].values, df.mean_episode_return.values, 'r')
plt.title('mean_episode_return') 

plt.subplot(326)
plt.plot(df['# _tick'].values, df._time.values, 'r')
plt.title('_time') 

"""

plt.figure(2)

plt.subplot(321)
plt.plot(df2['# _tick'].values, df2.entropy_loss.values, 'r')
plt.title('entropy_loss') 

plt.subplot(322)
plt.plot(df2['# _tick'].values, df2.total_loss.values, 'r')
plt.title('total_loss') 

plt.subplot(323)
plt.plot(df2['# _tick'].values, df2.pg_loss.values, 'r')
plt.title('pg_loss') 

plt.subplot(324)
plt.plot(df2['# _tick'].values, df2.baseline_loss.values, 'r')
plt.title('baseline_loss') 

plt.subplot(325)
plt.plot(df2['# _tick'].values, df2.mean_episode_return.values, 'r')
plt.title('mean_episode_return') 

plt.subplot(326)
plt.plot(df2['# _tick'].values, df2._time.values, 'r')
plt.title('_time') 

"""
plt.show()