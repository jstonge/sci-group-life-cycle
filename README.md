# An approximate master equations of scientific group life cycle

### Activate environment

```bash
julia> ]
(@v1.9) pkg> activate .
(SomeProject) pkg> instantiate
```

### Examples

```bash
julia src/sci-group-life-cycle.jl --mu 0.5 --nu_p 0.05 --a 3 -o "figs"
```

To know what parameter means, you can access help via

```bash
julia src/sci-group-life-cycle.jl --help

usage: sci-group-life-cycle.jl [--mu MU] [--nu_n NU_N] [--nu_p NU_P]
                        [--alpha ALPHA] [--beta BETA] [--a A] [-o O]
                        [-h]

optional arguments:
  --mu MU        inflow new students-non coders (type: Float64,
                 default: 0.1)
  --nu_n NU_N    death rate non-coders (type: Float64, default: 0.01)
  --nu_p NU_P    death rate coders (type: Float64, default: 0.01)
  --alpha ALPHA  benefits non coders (type: Float64, default: 0.01)
  --beta BETA    benefits coders (type: Float64, default: 0.02)
  --a A          Parameter cost function, which is c(n, i; a=3) = n ==
                 i == 0 ? 0.95 : 0.95 * exp(-a*i / n) (type: Float64,
                 default: 3.0)
  -o O           Output file for results
  -h, --help     show this help message and exit

p.s. Group benefits is τ(n, i, α, β) = exp(-α + β*(1 - c(n, i)))
```
