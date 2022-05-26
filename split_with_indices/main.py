from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# merge X y into 1 df
df = X.copy(deep=True)
df['Target'] = y
train_indices = set(X_train.index) # use Set to achieve O(1) lookup time
df['Train/test flag'] = ['Train' if idx in train_indices else 'Test' for idx in range(len(y))]