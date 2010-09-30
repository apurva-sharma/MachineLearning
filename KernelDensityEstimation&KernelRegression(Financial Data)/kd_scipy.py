from pylab import plot, figure, imshow, xlabel, ylabel, cm, show
from scipy import stats, mgrid, c_, reshape, random, rot90

def measure(n):
    """ Measurement model, return two coupled measurements.
    """
    m1 = random.normal(size=n)
    m2 = random.normal(scale=0.5, size=n)
    return m1+m2, m1-m2

# Draw experiments and plot the results
m1, m2 = measure(2000)
xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m1.max()

# Perform a kernel density estimator on the results
X, Y = mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = c_[X.ravel(), Y.ravel()]
values = c_[m1, m2]
kernel = stats.kde.gaussian_kde(values.T)
Z = reshape(kernel(positions.T).T, X.T.shape)

figure(figsize=(3, 3))
imshow(     rot90(Z),
            cmap=cm.gist_earth_r,
            extent=[xmin, xmax, ymin, ymax])
plot(m1, m2, 'k.', markersize=2)

show()
