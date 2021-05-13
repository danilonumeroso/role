def normalize(x, x_min=0, x_max=1, range_=(0, 1)):
    a, b = range_
    return (b-a) * ((x - x_min) / (x_max - x_min)) + a
