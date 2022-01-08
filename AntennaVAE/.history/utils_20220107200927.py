import matplotlib.pyplot as plt
# PLOT
def plot_train(all_loss):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x_ax = range(len(all_loss['train_loss']))
    l1 = plt.plot(x_ax, all_loss['train_loss'])
    l2 = plt.plot(x_ax, all_loss['val_loss'])
    plt.legend(['train_loss', 'val_loss'])
    plt.show()