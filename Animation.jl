# Lib
using LinearAlgebra
using Plots
using Kronecker
using LaTeXStrings

# Plot
# t=1:1:n_learn;
# p1=plot(t[:,1],dK1,label=L"|| K1 - K1^*||")
# p2=plot(t[:,1],dK2,label=L"|| K2 - K2^*||")
# p3=plot(t[:,1],dK3,label=L"|| K3 - K3^*||")
# plot(p1, p2, p3, layout=(3,1))

t=collect(1:1:n_learn);
anim1 = @animate for i = 1:n_learn
p1=plot(t[1:i],dK1[1:i],label=L"|| K_1 - K_1^*||",ylim=(-0.1, 0.2))
p2=plot(t[1:i],dK2[1:i],label=L"|| K_2 - K_2^*||",ylim=(-0.1, 0.9))
p3=plot(t[1:i],dK3[1:i],label=L"|| K_3 - K_3^*||",ylim=(-0.1, 0.2))
plot(p1, p2, p3, layout=(3,1))
end
gif(anim1, "OptimalControl.gif", fps = 5)

anim2 = @animate for i = 1:n_learn
    p4=plot(t[1:i],dKd1[1:i],label=L"|| K_{d_1} - K_{d_1}^*||",ylim=(-0.1, 0.6))
    p5=plot(t[1:i],dKd2[1:i],label=L"|| K_{d_2} - K_{d_2}^*||",ylim=(-0.1, 0.5))
    plot(p4, p5, layout=(2,1))
    end
gif(anim2, "WorstDisturbance.gif", fps = 5)