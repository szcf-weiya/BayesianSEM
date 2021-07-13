## ##################################################
## Julia Program for the final project of STAT 5020
## 
## author: WANG Lijun <ljwang@link.cuhk.edu.hk>
## date: April 9, 2019
## 
## ##################################################
using CmdStan
using DelimitedFiles
data = readdlm("prostate.data", ',')
Y = data[2:end, [9, 1, 8, 2, 4, 6]]
age = data[2:end, 3]
Z = data[2:end, [7, 5]]
N = size(Y)[1]

inputdata = Dict("N" => N, "Y" => Y, "Z" => Z, "age" => age)
monitor = vcat("mu.".*string.(1:8), 
                "lambda.".*string.(1:4), 
                "gamma.".*string.(1:3), 
                "sgm.".*string.(1:8), "sgd", 
                "phx.1.".*string.(1:3), "phx.2.".*string.(1:3), "phx.3.".*string.(1:3),
                "b",
                "c.".*string.(1:3), "c0")


# run 
model = Stanmodel(model = read(open("model.stan"), String), monitors = monitor)
rc, sim, cnames = stan(model, inputdata)

## ############################################
## save traceplots
## ############################################
using MCMCChains
using StatsPlots
p1 = plot(sim[vcat("sgm.".*string.(1:4))])
savefig(p1, "sgm1to4.png")
p2 = plot(sim[vcat("sgm.".*string.(5:8))])
savefig(p2, "sgm5to8.png")
p3 = plot(sim[vcat("lambda.".*string.(1:4))])
savefig(p3, "lambda.png")
p4 = plot(sim[vcat("gamma.".*string.(1:3))])
savefig(p4, "gamma.png")
p5 = plot(sim[["phx.1.1","phx.1.2","phx.1.3","phx.2.2","phx.2.3", "phx.3.3"]])
savefig(p5, "phx.png")
p6 = plot(sim[vcat("b","c.".*string.(1:3), "c0")])
savefig(p6, "bc.png")
p7 = plot(sim["sgd"])
savefig(p7, "sgd.png")

## ############################################
## calculate logBF
## ############################################
using StatsBase
function logBF(model, data; nt::Int=10)
    U = ones(nt+1)
    for s = 1:(nt+1)
        data["t"] = 1/nt*(s-1)
        rc, sim, cnames = stan(model, data, summary = false)
        U[s] = mean(sim[:,:,1])
    end
    res = 0
    for s = 1:nt
        res += ( U[s] + U[s+1] ) * (1 / nt) / 2
    end
    return res
end

## ############################################
## model comparisons
## ############################################

# comparison I
model_c = Stanmodel(model = read(open("model_c.stan"), String), output_format=:array, monitors = ["U"], nchains = 1)
res = ones(10)
for i=1:10
    res[i] = logBF(model_c, inputdata)
end

# comparison II
model_c13 = Stanmodel(model = read(open("model_c13.stan"), String), output_format=:array, monitors = ["U"], nchains = 1)
res = ones(10)
for i=1:10
    res[i] = logBF(model_c13, inputdata)
end

# comparison III
model_cd = Stanmodel(model = read(open("model_cd.stan"), String), output_format=:array, monitors = ["U"], nchains = 1)
res = ones(10)
for i = 1:10
    res[i] = logBF(model_cd, inputdata)
end

## ############################################
## sensitivity analysis
## ############################################
model_s1 = Stanmodel(model = read(open("model_s1.stan"), String), monitors = monitor, nchains = 4)
rc, sim, cnames = stan(model_s1, inputdata)
model_s2 = Stanmodel(model = read(open("model_s2.stan"), String), monitors = monitor, nchains = 4)
rc, sim, cnames = stan(model_s2, inputdata)
model_s3 = Stanmodel(model = read(open("model_s3.stan"), String), monitors = monitor, nchains = 4)
rc, sim, cnames = stan(model_s3, inputdata)