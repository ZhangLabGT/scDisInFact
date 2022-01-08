import matplotlib.pyplot as plt
# PLOT
def plot_train(diz_loss):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x_ax = range(len(diz_loss['train_loss']))
    l1 = plt.plot(x_ax, diz_loss['train_loss'])
    l2 = plt.plot(x_ax, diz_loss['val_loss'])
    plt.legend(['train_loss', 'val_loss'])
    plt.show()