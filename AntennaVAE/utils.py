import matplotlib.pyplot as plt
# PLOT
def plot_train(all_loss, seperate=False):
    x_rg = range(len(all_loss['train_loss']))
    fig = plt.figure()
    if not seperate:
        fig.add_subplot(1,1,1)
        plt.plot(x_rg, all_loss['train_loss'])
        plt.plot(x_rg, all_loss['test_loss'])
        plt.legend(['train_loss', 'test_loss'])
    else:
        ax1 = fig.add_subplot(2,2,1)

        ax1.plot(x_rg, all_loss['train_loss'])
        ax1.plot(x_rg, all_loss['test_loss'])
        ax1.legend(['train_loss', 'test_loss'])
        ax1.title.set_text('Combine Loss Fig')

        ax2 = fig.add_subplot(2,2,2)

        ax2.plot(x_rg, all_loss['train_loss'])
        ax2.title.set_text('Train_Loss Fig')
        
        test_loss_fig = fig.add_subplot(2,2,3)
        test_loss_fig.plot(x_rg, all_loss['test_loss'])
        test_loss_fig.title.set_text('Test_Loss Fig')
    plt.tight_layout()
    plt.show()
    