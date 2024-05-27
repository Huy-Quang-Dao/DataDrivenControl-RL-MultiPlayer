# Code for paper: Output Feedback H1 Control for Linear
# Discrete-Time Multi-Player Systems With
# Multi-Source Disturbances Using Off-Policy
# Q-Learning.

# Programing Language : Julia 
# Purpose : Practice and Research

# Lib
using LinearAlgebra
using Plots
using Kronecker
# Model
A_c = [0.4126 0 -0.0196;
0.0333 0.5207 -0.0413;
-0.0101 0 0.2571];
B1_c=[-1.7734,0.0928,-0.0424][:,:];
B2_c=[0.0696,0.4658,-0.0930][:,:];
B3_c=[0.0734,0.1051,2.0752][:,:];
E1_c=[0.123956,-0.37895,0.12698][:,:];
E2_c=[-0.08564,0.1268,0.036984][:,:];
C=[1 0 -1;-2 1 0;1 1 1];
A = C*A_c*pinv(C'*C)*C';
B1 = C*B1_c;
B2 = C*B2_c;
B3 = C*B3_c;
E1 = C*E1_c;
E2 = C*E2_c;
# Value function parameters
Q = Matrix(Diagonal([10,10,10]));
R1=[10][:,:];
R2=R1;R3=R1;
γ=5;

# Nash equilibrium
K1_s = -[0.0295 -0.0829 0.0209];
K2_s = -[-0.1904 -0.1612 -0.1622];
K3_s = -[0.0253 -0.0240 -0.0756];
Kd1_s = -[-0.0599 -0.0546 -0.0503];
Kd2_s = -[0.0183 0.0186 0.0169];

# 
n = size(A, 2);
m1 = size(B1, 2);
m2 = size(B2, 2);
m3 = size(B3, 2);
p1 = size(E1, 2);
p2 = size(E2, 2);
q = n + m1 + m2 + m3 + p1 + p2;

# Initial control matrix
# K1_0 = [-10.3603 -8.8495 -5.3501];
# K2_0 = [7.1171 8.5319 4.8977];
# K3_0 = [0.1674 -6.2478 -4.4080];
# Kd1_0 = [-7.8473 2.1523 3.1386];
# Kd2_0 = [11.4279 5.9016 2.3473];
K1_0 = [0.1 0 -0];
K2_0 = [-0.6 0 0];
K3_0 = [0.1 -0.0 0];
Kd1_0 = [-0.5 0 0];
Kd2_0 = [-0.5 0 0];
H0 = zeros(q, q);

# Stores the control matrix
save_K1 = [K1_0];
save_K2 = [K2_0];
save_K3 = [K3_0];
save_Kd1 = [Kd1_0];
save_Kd2 = [Kd2_0];
H = [H0];
n_end = 50;
n_learn = 30;
x = zeros(n, n_end);
x[:, 1]=[0.2;-0.65;1.55];
u1 = zeros(m1, n_end);
u2 = zeros(m2, n_end);
u3 = zeros(m3, n_end);
w1 = zeros(p1, n_end);
w2 = zeros(p2, n_end);
z = zeros(q, n_end);
for i in 1:n_learn
global Φ = zeros(n_end, q*q);
global Ψ = zeros(n_end, 1);
    for k in 1:n_end-1  # Collect Data
        global e1 = 0.1 * sin(k);
        global e2 = 0.1 * sin(0.1 * k);
        global e3 = 0.1 * sin(0.3 * k);
        global u1[:, k] = -save_K1[i] * x[:, k] + [e1];
        global u2[:, k] = -save_K2[i] * x[:, k] + [e2];
        global u3[:, k] = -save_K3[i] * x[:, k] + [e3];
        global w1[:, k] = [exp(-0.01*k) * sin(0.1*k)];
        global w2[:, k] = [exp(-0.1*k) * sin(2.1*k)];
        global z[:, k] = vcat(x[:, k], u1[:, k], u2[:, k], u3[:, k], w1[:, k], w2[:, k]);
        global x[:, k+1] = A * x[:, k] + B1 * u1[:, k] + B2 * u2[:, k] + B3 * u3[:, k] + E1 * w1[:, k]+ E2 * w2[:, k];
        global z[:, k+1] = vcat(x[:, k+1], -save_K1[i] * x[:, k+1], -save_K2[i] * x[:, k+1], -save_K3[i] * x[:, k+1], -save_Kd1[i] * x[:, k+1],-save_Kd2[i] * x[:, k+1]);
        global Φ[k, :] = kron(z[:, k]', z[:, k]') - kron(z[:, k+1]', z[:, k+1]');
        global Ψ[k] = x[:, k]' * Q * x[:, k] + u1[:, k]' * R1 * u1[:, k] + u2[:, k]' * R2 * u2[:, k] + u3[:, k]' * R3 * u3[:, k] - γ^2 * (w1[:, k]' * w1[:, k]+w2[:, k]' * w2[:, k]);
    end
    global vec_H = pinv(Φ' * Φ) * Φ' * Ψ;
    push!(H, reshape(vec_H, (q, q)));  # Find H in Q-Learning
    Hxu1 = H[end][1:n, n+1:n+m1];
    Hxu2 = H[end][1:n, n+m1+1:n+m1+m2];
    Hxu3 = H[end][1:n, n+m1+m2+1:n+m1+m2+m3];
    Hxw1 = H[end][1:n, n+m1+m2+m3+1:n+m1+m2+m3+p1];
    Hxw2 = H[end][1:n, n+m1+m2+m3+p1+1:end];
    Hu1u1 = H[end][n+1:n+m1, n+1:n+m1];
    Hu1u2 = H[end][n+1:n+m1, n+m1+1:n+m1+m2];
    Hu1u3 = H[end][n+1:n+m1, n+m1+m2+1:n+m1+m2+m3];
    Hu1w1 = H[end][n+1:n+m1, n+m1+m2+m3+1:n+m1+m2+m3+p1];
    Hu1w2 = H[end][n+1:n+m1, n+m1+m2+m3+p1+1:end];
    Hu2u2 = H[end][n+m1+1:n+m1+m2, n+m1+1:n+m1+m2];
    Hu2u3 = H[end][n+m1+1:n+m1+m2, n+m1+m2+1:n+m1+m2+m3];
    Hu2w1 = H[end][n+m1+1:n+m1+m2, n+m1+m2+m3+1:n+m1+m2+m3+p1];
    Hu2w2 = H[end][n+m1+1:n+m1+m2, n+m1+m2+m3+p1+1:end];
    Hu3u3 = H[end][n+m1+m2+1:n+m1+m2+m3, n+m1+m2+1:n+m1+m2+m3];
    Hu3w1 = H[end][n+m1+m2+1:n+m1+m2+m3, n+m1+m2+m3+1:n+m1+m2+m3+p1];
    Hu3w2 = H[end][n+m1+m2+1:n+m1+m2+m3, n+m1+m2+m3+p1+1:end];
    Hw1w1 = H[end][n+m1+m2+m3+1:n+m1+m2+m3+p1, n+m1+m2+m3+1:n+m1+m2+m3+p1];
    Hw1w2 = H[end][n+m1+m2+m3+1:n+m1+m2+m3+p1, n+m1+m2+m3+p1+1:end];
    Hw2w2 = H[end][n+m1+m2+m3+p1+1:end, n+m1+m2+m3+p1+1:end];
    K1 = pinv(Hu1u1) * (Hxu1' - (Hu1u2 * save_K2[i] + Hu1u3 * save_K3[i] + Hu1w1 * save_Kd1[i] + Hu1w2 * save_Kd2[i]));
    K2 = pinv(Hu2u2) * (Hxu2' - (Hu1u2' * save_K1[i] + Hu2u3 * save_K3[i] + Hu2w1 * save_Kd1[i] + Hu2w2 * save_Kd2[i]));
    K3 = pinv(Hu3u3) * (Hxu3' - (Hu1u3' * save_K1[i] + Hu2u3' * save_K2[i] + Hu3w1 * save_Kd1[i] + Hu3w2 * save_Kd2[i]));
    Kd1 = pinv(Hw1w1) * (Hxw1' - (Hu1w1' * save_K1[i] + Hu2w1' * save_K2[i] + Hu3w1' * save_K3[i] + Hw1w2*save_Kd2[i]));
    Kd2 = pinv(Hw2w2) * (Hxw2' - (Hu1w2' * save_K1[i] + Hu2w2' * save_K2[i] + Hu3w2' * save_K3[i] + Hw1w2'*save_Kd1[i]));
    # Find Optimal Solution Step by Step
    push!(save_K1, K1);
    push!(save_K2, K2);
    push!(save_K3, K3);
    push!(save_Kd1, Kd1);
    push!(save_Kd2, Kd2);
end

# Calculate Error
dK1 = zeros(n_learn, 1);
dK2 = zeros(n_learn, 1);
dK3 = zeros(n_learn, 1);
dKd1 = zeros(n_learn, 1);
dKd2 = zeros(n_learn, 1);

for j in 1:n_learn
    dK1[j] = norm(save_K1[j] - K1_s);
    dK2[j] = norm(save_K2[j] - K2_s);
    dK3[j] = norm(save_K3[j] - K3_s);
    dKd1[j] = norm(save_Kd1[j] - Kd1_s);
    dKd2[j] = norm(save_Kd2[j] - Kd2_s);
end

