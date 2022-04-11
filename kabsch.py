import numpy as np
from math import sqrt

scaling = False

# Implements Kabsch algorithm - best fit.
# Supports scaling (umeyama)
# Compares well to SA results for the same data.
# Input:
#     Nominal  A Nx3 matrix of points
#     Measured B Nx3 matrix of points
# Returns s,R,t
# s = scale B to A
# R = 3x3 rotation matrix (B to A)
# t = 3x1 translation vector (B to A)
def rigid_transform_3D(A, B, scale):
    assert len(A) == len(B)

    N = A.shape[0];  # total points or total rows in matrix A

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # center the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    if scale:
        H = np.transpose(BB) * AA / N
    else:
        H = np.transpose(BB) * AA

    #calculating svd
    
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print ("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T * U.T

    if scale:
        varA = np.var(A, axis=0).sum()
        c = 1 / (1 / varA * np.sum(S))  # scale factor
        t = -R * (centroid_B.T * c) + centroid_A.T
    else:
        c = 1
        t = -R * centroid_B.T + centroid_A.T

    return c, R, t




# Test
A = np.matrix([[10.0,10.0,10.0],
               [20.0,10.0,10.0],
               [20.0,10.0,15.0]])

B = np.matrix([[18.8106,17.6222,12.8169],
               [28.6581,19.3591,12.8173],
               [28.9554, 17.6748, 17.5159]])

n = B.shape[0] #number of rows in matrix

Ttarg = np.matrix([[0.9848, 0.1737,0.0000,-11.5859],
                   [-0.1632,0.9254,0.3420, -7.621],
                   [0.0594,-0.3369,0.9400,2.7755],
                   [0.0000, 0.0000,0.0000,1.0000]])

Tstarg = np.matrix([[0.9848, 0.1737,0.0000,-11.5865],
                   [-0.1632,0.9254,0.3420, -7.621],
                   [0.0594,-0.3369,0.9400,2.7752],
                   [0.0000, 0.0000,0.0000,1.0000]])

# recover the transformation
s, ret_R, ret_t = rigid_transform_3D(A, B, scaling)
#s, ret_R, ret_t = umeyama(A, B)

# Find the error
B2 = (ret_R * B.T) + np.tile(ret_t, (1, n))
B2 = B2.T
err = A - B2
err = np.multiply(err, err)
err = np.sum(err)
rmse = sqrt(err / n)

#convert to 4x4 transform
match_target = np.zeros((4,4))
match_target[:3,:3] = ret_R
match_target[0,3] = ret_t[0]
match_target[1,3] = ret_t[1]
match_target[2,3] = ret_t[2]
match_target[3,3] = 1

print ("Points A")
print (A)
print ("")

print ("Points B")
print(B)
print ("")

print("Rotation")
print (ret_R)
print ("")

print ("Translation")
print (ret_t)
print ("")

print ("Scale")
print (s)
print ("")

print ("Homogeneous Transform")
print (match_target)
print ("")

if scaling:
    print ("Total Diff to SA matrix")
    print (np.sum(match_target - Tstarg))
    print ("")
else:
    print ("Total Diff to SA matrix")
    print (np.sum(match_target - Ttarg))
    print ("")

print ("RMSE:", rmse)
print ("If RMSE is near zero, the function is correct!")