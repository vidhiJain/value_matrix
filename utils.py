import matplotlib.pyplot as plt

def plot_check(out, path_matrix):
    plt.clf()
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1,2,1)
    plt.imshow(out.detach().numpy()[0][0])
    plt.title('model')

    plt.subplot(1,2,2)
    plt.imshow(path_matrix)
    plt.title('expert')
    
    plt.suptitle('Value estimate model')

    plt.draw()
    plt.pause(0.5)
    plt.close()