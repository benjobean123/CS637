import scipy.io

# Load MATLAB file
mat = scipy.io.loadmat('Indian_pines_corrected.mat')

# Grab the data from the MATLAB file
data = mat['indian_pines_corrected']
print(data.shape)

# Load MATLAB file
mat = scipy.io.loadmat('Indian_pines_gt.mat')

# Grab the data from the MATLAB file
truth = mat['indian_pines_gt']
print(truth.shape)
