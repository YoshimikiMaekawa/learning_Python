from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline

dataset = load_boston()
x = dataset.data
t = dataset.target

# Sprit dataset into training data (70%) and test data (30%)
# random_state=0 means that result of separation is not different 
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)

# パイプライン化を使用した場合

pipline = Pipeline([
    ('scaler', PowerTransformer()),
    ('reg', LinearRegression())
])

pipline.fit(x_train, t_train)
linear_result = pipline.score(x_test, t_test) # なんでこの処理でx_testの標準化までできちゃうの？？
print(linear_result)

# 下のコードはパイプライン化を使用しなかった場合

# ### preprocessing ###
# # scaler = StandardScaler()
# scaler = PowerTransformer()
# scaler.fit(x_train)
# x_train_scaled = scaler.transform(x_train)
# x_test_scaled = scaler.transform(x_test)

# ### training ###
# reg_model = LinearRegression() # Declare the model
# reg_model.fit(x_train_scaled, t_train) # Train the training data

# ### evaluation ###
# print(reg_model.score(x_train_scaled, t_train))
# print(reg_model.score(x_test_scaled, t_test))