from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    centered_dataset = x - np.mean(x, axis=0)
    return centered_dataset

def get_covariance(dataset):
    mat = np.array(dataset)
    dot = np.dot(np.transpose(mat), mat)
    covariance = np.divide(dot, len(mat) - 1)
    return covariance

def get_eig(S, m):
    Lambda, U = eigh(S,subset_by_index=[len(S)-m,len(S)-1])
    Lambda = np.flip(Lambda)
    Lambda = np.diag(Lambda)
    for mat in U:
        tmp = mat[0]
        mat[0] = mat[1]
        mat[1] = tmp
    return Lambda, U
    

def get_eig_prop(S, prop):
    Lambda, U = eigh(S,subset_by_value=[prop * np.trace(S),np.inf])
    Lambda = np.flip(Lambda)
    Lambda = np.diag(Lambda)
    for mat in U:
        tmp = mat[0]
        mat[0] = mat[1]
        mat[1] = tmp
    return Lambda, U
    pass

def project_image(image, U):
    projection = []
    for i in range(len(U)):
        transpose = np.transpose(U)
        a = np.dot(transpose, image)
        xi = np.dot(a, U[i])
        projection.append(xi)
    projection = np.array(projection)
    return projection

def display_image(orig, proj):
    orig_image = np.reshape(orig, (32,32)).transpose((1,0))
    proj_image = np.reshape(proj, (32,32)).transpose((1,0))
    fig, axs = plt.subplots(nrows=1, ncols=2)
    
    img1 = axs[0].imshow(orig_image, aspect='equal')
    img2 = axs[1].imshow(proj_image, aspect='equal')
    
    axs[0].set_title("Original")
    axs[1].set_title("Projection")
    
    fig.colorbar(img1, ax=axs[0])
    fig.colorbar(img2, ax=axs[1])
    plt.show()
    pass


x = load_and_center_dataset('YaleB_32x32.npy')
S = get_covariance(x)
Lambda, U = get_eig(S, 2)
projection = project_image(x[0], U)
display_image(x[0], projection)