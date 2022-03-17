function fv(n)
    x = √2
    r = x/2
    for _=2:n
        y = √(2 + x)
        r = r * y/2
        x = y
    end
    return 2/r
end

function fv2(n)
    a = 0
    p = 1
    for _=1:n
        a = √(2 + a)
        p = p*a/2
    end
    return 2/p
end

function fv3(n)
    a = 1
    c = √(2)
    for _=1:n
        a = a * c/2
        c = √(2 + c)
    end
    return 2/a
end

using Plots
n = 10
plot(1:10, [fvs, fill(pi, 10)], xlabel="n", legend=["fv", "pi"])