# phuonglh
# Note: Use the Excel file to answer the questions instead of this code.
# Morgage-Backed Security Pricing

M0 = 400
c = 6/100/12
seasoning = 0
r = 4.5/100/12
n = 240

B = M0*c/(1 - (1 + c)^(-(n - seasoning)))

"Compute M_k for all k = 1...n"
function computeM(M0, B, c, n)
    map(k -> (1+c)^k * M0 - B*((1 + c)^k - 1)/c, 1:n)
end

function computeP(M0, B, c, n)
    M = computeM(M0, B, c, n)
    P = map(k -> B - c*M[k-1], 2:n)
    return [B - c*M0; P]
end

function computeV0(M0, c, r, n)
    a = c*M0/((1 + c)^n - 1)
    b = ((1 + r)^n - (1 + c)^n)/((r-c) * ((1 + r)^n))
    a * b
end

function computeF0(M0, c, r, n)
    a = c*(1+c)^n*M0 / ((1 + c)^n - 1)
    b = ((1 + r)^n - 1)/(r * (1 + r)^n)
    a * b
end

"Duration of the principal-only MBS"
function computeD_P(M0, B, c, r, n)
    P = computeP(M0, B, c, n)
    V0 = computeV0(M0, c, r, n)
    D = map(k -> k * P[k]/(1+r)^k, 1:n)
    sum(D)/(12*V0)
end

"Duration of the interest-only MBS"
function computeD_I(M0, B, c, r, n)
    D_P = computeD_P(M0, B, c, r, n)
    V0 = computeV0(M0, c, r, n)
    F0 = computeF0(M0, c, r, n)
    W0 = F0 - V0
    D = map(k -> k/((1+r)^k), 1:n)
    sum(D)*B / (12 * W0) - (V0/W0)*D_P
end

