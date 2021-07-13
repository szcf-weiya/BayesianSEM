using CmdStan
using Distributions

## ############################################
## parameter setup
## ############################################
b = 0.5
γ = [0.4, 0.4, 0.3, 0.2]
ψ_δ = 0.36
ψ_ε = repeat([0.3, 0.5, 0.4], inner = 3)
Φ = [1.0 0.3; 0.3 1.0]
A = repeat([0.3, 0.7, 0.9], outer = 3)
Λ = [1 0 0;
    0.9 0 0;
    0.7 0 0;
    0 1 0;
    0 0.9 0;
    0 0.7 0;
    0 0 1;
    0 0 0.9;
    0 0 0.7]
df_d = 5
df_c = 9
u = zeros(9)

## ############################################
## simulate data
## ############################################
function genData(N::Int)
    c = ones(N)
    d = ones(N)
    Y = ones(N, 9)
    for i = 1:N
        c[i] = rand(TDist(df_c))
        d[i] = rand(TDist(df_d))
        ξ = rand(MvNormal(Φ))
        η = b*d[i] + γ[1]*ξ[1] + γ[2]*ξ[2] + γ[3]*ξ[1]^2 + γ[4]*ξ[2]^2 + rand(Normal(sqrt(ψ_δ))) 
        Y[i,:] = u + A * c[i] + Λ * vcat(η, ξ) + rand(MvNormal(sqrt.(ψ_ε)))
    end
    return c, d, Y
end

monitor = vcat("u.".*string.(1:9), 
    "lam.".*string.(1:6), 
    "A.".*string.(1:9), 
    "gam.".*string.(1:4),
    "sgm2.".*string.(1:9),
    "phx.1.1", "phx.1.2", "phx.2.1", "phx.2.2",
    "sgd2",
    "b"
)


model = Stanmodel(name = "sem", model = read(open("q1.stan"), String), monitors = monitor)

## ##############################
## sensitivity analysis
## ##############################
modele1 = Stanmodel(name = "seme1", model = read(open("q1e1.stan"), String), monitors = monitor)
modele2 = Stanmodel(name = "seme2", model = read(open("q1e2.stan"), String), monitors = monitor)


N = 500
c, d, Y = genData(N)
data = Dict("N" => N, 
            "c" => c, 
            "d" => d, 
            "Phi0" => Φ,
            "Y" => Y)

rc, sim, cnames = stan(model, data, summary = true)
rce1, sime1, cnamese1 = stan(modele1, data, summary = true)
rce2, sime2, cnamese2 = stan(modele2, data, summary = true)

## ############################################
## Trace plots
## ############################################

using Plots
function traceplot(res, idx::Int, lbl::String, truth)
    p = plot(res[:,:,1][:,idx], title = lbl, label="chain 1")
    for j = 2:4
        plot!(p, res[:,:,j][:,idx], label="chain "*string(j))
    end
    hline!(p, [truth], color=:black, lw=2, label="truth")
    return p
end

#p1 = traceplot(sim, 1, "u[1]", 0);
p1 = traceplot(sim, 5, "u[5]", 0);
p2 = traceplot(sim, 10, "lam[1]", 0.9);
p3 = traceplot(sim, 16, "A[1]", 0.3);
p4 = traceplot(sim, 25, "b", 0.5);
p5 = traceplot(sim, 30, "sgm2.1", 0.3);
p6 = traceplot(sim, 39, "sgd2", 0.36);
p7 = traceplot(sim, 40, "phx.1.1", 1.0);
p8 = traceplot(sim, 41, "phx.1.2", 0.3);
p = plot(p1, p2, p3, p4, p5, p6, p7, p8, layout = (4,2), legend=false, size = (1080, 720));
savefig(p, "traceplots.png")


## sensitivity analysis

p1 = traceplot(sime1, 5, "u[5]", 0);
p2 = traceplot(sime1, 10, "lam[1]", 0.9);
p3 = traceplot(sime1, 16, "A[1]", 0.3);
p4 = traceplot(sime1, 25, "b", 0.5);
p5 = traceplot(sime1, 30, "sgm2.1", 0.3);
p6 = traceplot(sime1, 39, "sgd2", 0.36);
p7 = traceplot(sime1, 40, "phx.1.1", 1.0);
p8 = traceplot(sime1, 41, "phx.1.2", 0.3);
p = plot(p1, p2, p3, p4, p5, p6, p7, p8, layout = (4,2), legend=false, size = (1080, 720));
savefig(p, "traceplots_e1.png")
p1 = traceplot(sime2, 5, "u[5]", 0);
p2 = traceplot(sime2, 10, "lam[1]", 0.9);
p3 = traceplot(sime2, 16, "A[1]", 0.3);
p4 = traceplot(sime2, 25, "b", 0.5);
p5 = traceplot(sime2, 30, "sgm2.1", 0.3);
p6 = traceplot(sime2, 39, "sgd2", 0.36);
p7 = traceplot(sime2, 40, "phx.1.1", 1.0);
p8 = traceplot(sime2, 41, "phx.1.2", 0.3);
p = plot(p1, p2, p3, p4, p5, p6, p7, p8, layout = (4,2), legend=false, size = (1080, 720));
savefig(p, "traceplots_e2.png")

=#

## ############################################
## calculate logBF
## ############################################
function logBF(model, data; nt::Int=10)
    U = ones(nt+1)
    for s = 1:(nt+1)
        data["ts"] = 1/nt*(s-1)
        rc, sim, cnames = stan(model, data, summary = false)
        U[s] = mean(sim[:,:,1])
    end
    res = 0
    for s = 1:nt
        res += ( U[s] + U[s+1] ) * (1 / nt) / 2
    end
    return res
end

q1_model = Stanmodel(name = "sem1", model = read(open("q1.stan"),String), monitors = monitor, nchains=1)
q2a_model = Stanmodel(name = "sem2a", model = read(open("q2a.stan"),String), monitors = ["U"], nchains=1)
q2b_model = Stanmodel(name = "sem2b", model = read(open("q2b.stan"),String), monitors = ["U"], nchains=1)

# 10 replications
truth = vcat(u, repeat([0.9,0.7], outer=3), A, b, γ, ψ_ε, ψ_δ, Φ[:])
N = 500

bf_a = ones(10)
bf_b = ones(10)
bias = ones(10,43)
rms = ones(10,43)
for i = 1:10
    println("i = ", i)
    c, d, Y = genData(N)
    data = Dict("N" => N, 
            "c" => c, 
            "d" => d, 
            "Phi0" => Φ,
            "Y" => Y)
    rc, sim, cnames = stan(q1_model, data, summary = false)
    bias[i,:] = abs.(mean(sim[:,:,1], dims=1)' .- truth)
    rms[i,:] = (mean(sim[:,:,1], dims=1)' .- truth).^2
    bf_a[i] = logBF(q2a_model, data)
    bf_b[i] = logBF(q2b_model, data)
end
avg_bias = mean(bias, dims = 1)
avg_rms = mean(rms, dims = 1)