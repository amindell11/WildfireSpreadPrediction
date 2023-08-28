# %%
all_loss_histories = []
all_val_loss_histories = []


loss_history = history.history['loss']
val_loss_history = history.history['val_loss']


all_loss_histories.append(loss_history)
all_val_loss_histories.append(val_loss_history)
 
 
def smooth_curve(points, factor=0.9):
   smoothed_points = []
   for point in points:
       if smoothed_points:
           previous = smoothed_points[-1]
           smoothed_points.append(previous * factor + point * (1 - factor))
       else:
           smoothed_points.append(point)
   return smoothed_points
 
def plot_accuracy(all_loss_histories, all_val_loss_histories):
   plt.rcParams['figure.figsize'] = (16.0, 9.0)
  
   # Averaging Total fold's accuracy into 1
   average_loss_history = [np.mean([x[i] for x in all_loss_histories]) for i in range(num_epochs)]
   average_val_loss_history = [np.mean([x[i] for x in all_val_loss_histories]) for i in range(num_epochs)]
  
   #plotting training and validation loss into 1 graph
   plt.plot(range(1, len(average_loss_history) + 1), average_loss_history, label = 'Training Loss')
   plt.plot(range(1, len(average_val_loss_history) + 1), average_val_loss_history, label = 'Validation Loss')
   plt.suptitle('Loss Graph')
   plt.xlabel('Epochs')
   plt.ylabel('Loss')
   plt.legend()
   plt.show()
  

  
   #plotting smooth curve graphs of loss graph
   print('Smooth curve graphs of above graphs')
   smooth_loss_history = smooth_curve(average_loss_history)
   plt.plot(range(1, len(smooth_loss_history) + 1), smooth_loss_history, label = 'Training Loss')
 
   smooth_val_loss_history = smooth_curve(average_val_loss_history)
   plt.plot(range(1, len(smooth_val_loss_history) + 1), smooth_val_loss_history, label = 'Validation Loss')
 
   plt.xlabel('Epochs')
   plt.ylabel('Loss')
   plt.suptitle('Loss Graph')
   plt.legend()
   plt.show()
  

  
plot_accuracy(all_loss_histories, all_val_loss_histories)
